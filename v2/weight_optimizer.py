"""
v2/weight_optimizer.py
================================================
V2 —— 融合权重自动优化器
================================================

四路信号全部来自已有数据库，无需新调 LLM：
  ① LLM 语义信号   ← semantic_history.hawkish_score
  ② VAR 计量信号   ← 复用 backtest.py 的走一步VAR函数
  ③ 市场信号       ← political_quant_monthly.market_signal（收益率曲线斜率）
  ④ 政治压力信号   ← political_quant_monthly.political_score（EPU + VIX）

融合公式：
  fusion_score = w_llm·LLM + w_var·VAR + w_market·Market + w_pol·PoliticalPressure

决策规则：
  fusion > hike_thr → HIKE
  fusion < cut_thr  → CUT
  otherwise         → HOLD

数据集划分：
  训练集 2000-2020（约174次）：用于网格搜索最优参数
  验证集 2021-2024（32次）：评估泛化能力
  最终测试集 2025-至今：保留，本模块不触碰

运行方式（从项目根目录）：
  E:\\Anaconda\\envs\\fed-agent\\python.exe v2/weight_optimizer.py
"""

import sys
import sqlite3
import warnings
from datetime import timedelta
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np
import yaml

warnings.filterwarnings("ignore")

# ── sys.path：确保项目根目录可寻址 ─────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.db import DB_PATH
from v2.backtest import var_predict_direction, fetch_full_macro_series

# ── 读取配置 ────────────────────────────────────────────────────────────
_cfg = yaml.safe_load(open(str(_ROOT / "config.yml"), encoding="utf-8"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 数据加载与特征矩阵构建
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_feature_matrix() -> pd.DataFrame:
    """
    从三张表合并出统一特征矩阵，每行对应一次 FOMC 会议：
      fomc_history      → 实际决策（标签）+ 宏观快照 + train/test split
      semantic_history  → LLM hawkish_score
      political_quant_monthly → 市场信号、政治压力（月频，merge_asof 对齐）
    """
    conn = sqlite3.connect(DB_PATH)

    fomc = pd.read_sql(
        "SELECT meeting_date, chair, actual_decision, split, "
        "cpi_yoy, unemployment, fed_funds_rate "
        "FROM fomc_history ORDER BY meeting_date",
        conn,
    )

    semantic = pd.read_sql(
        "SELECT meeting_date, hawkish_score, inflation_concern, growth_concern "
        "FROM semantic_history",
        conn,
    )

    pq = pd.read_sql(
        "SELECT month, market_signal, political_score FROM political_quant_monthly "
        "ORDER BY month",
        conn,
    )
    conn.close()

    # ── 日期类型转换 ────────────────────────────────────────────────────
    fomc["meeting_ts"] = pd.to_datetime(fomc["meeting_date"])
    semantic["meeting_date"] = pd.to_datetime(semantic["meeting_date"]).dt.strftime("%Y-%m-%d")
    fomc["meeting_date"] = pd.to_datetime(fomc["meeting_date"]).dt.strftime("%Y-%m-%d")
    pq["month_ts"] = pd.to_datetime(pq["month"])

    # ── 合并 fomc + semantic ───────────────────────────────────────────
    df = fomc.merge(semantic, on="meeting_date", how="left")
    df = df.sort_values("meeting_ts").reset_index(drop=True)

    # ── merge_asof：把月频政治数据对齐到会议日（取最近月份，防前视偏差）
    pq_sorted = pq.sort_values("month_ts").rename(columns={"month_ts": "meeting_ts"})
    df = pd.merge_asof(df, pq_sorted[["meeting_ts", "market_signal", "political_score"]],
                       on="meeting_ts", direction="backward")

    # ── 丢弃 hawkish_score 为 NaN 的行（唯一缺失：2007-06-28）─────────
    before = len(df)
    df = df.dropna(subset=["hawkish_score"]).reset_index(drop=True)
    if len(df) < before:
        print(f"  [数据] 丢弃 {before - len(df)} 行（hawkish_score 缺失）")

    # ── political_score 缺失时默认 0.5（中性）──────────────────────────
    df["political_score"] = df["political_score"].fillna(0.5)

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 信号数值化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SIG_MAP = {"hike": 1.0, "hold": 0.0, "cut": -1.0}

def sig_to_num(s) -> float:
    """文字信号 → 数值：hike=+1, hold=0, cut=-1"""
    return _SIG_MAP.get(str(s).lower(), 0.0)

def pol_to_hawkish(political_score: float) -> float:
    """
    政治压力得分 [0,1] → 鹰鸽数值 [-1,+1]
    高 political_score = 高不确定性 = 鸽派压力 = 负分
    公式：pol_hawkish = -(2 * political_score - 1)
    """
    return -(2.0 * float(political_score) - 1.0)


def add_numeric_signals(df: pd.DataFrame, var_signals: list) -> pd.DataFrame:
    """
    在特征矩阵中添加数值化信号列，供向量化融合计算使用。
    var_signals: 与 df 行序一致的 VAR 信号列表（由 compute_var_signals 生成）
    """
    df = df.copy()
    df["var_signal"]  = var_signals
    df["var_num"]     = df["var_signal"].map(sig_to_num)
    df["market_num"]  = df["market_signal"].map(sig_to_num)
    df["pol_hawkish"] = df["political_score"].apply(pol_to_hawkish)
    df["label"]       = df["actual_decision"].str.lower()
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. VAR 信号预计算（走一步，无前视偏差）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_var_signals(df: pd.DataFrame, df_macro: pd.DataFrame) -> list:
    """
    为 df 中每次会议计算 VAR 走一步预测（与 backtest.py 逻辑完全一致）。
    预先计算一次，后续网格搜索直接复用结果，避免重复 VAR 拟合。
    """
    signals = []
    for _, row in df.iterrows():
        cutoff = pd.Timestamp(row["meeting_date"]) - timedelta(days=2)
        result = var_predict_direction(df_macro, cutoff)
        signals.append(result["signal"])
    return signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 单次融合预测（向量化）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fuse_predict(
    llm_v: np.ndarray,
    var_v: np.ndarray,
    market_v: np.ndarray,
    pol_v: np.ndarray,
    w_llm: float,
    w_var: float,
    w_market: float,
    w_pol: float,
    hike_thr: float,
    cut_thr: float,
) -> np.ndarray:
    """
    向量化融合预测，返回 numpy 字符串数组。
    调用一次约 1μs，适合在循环中高频调用。
    """
    fusion = w_llm * llm_v + w_var * var_v + w_market * market_v + w_pol * pol_v
    pred = np.where(fusion > hike_thr, "hike", np.where(fusion < cut_thr, "cut", "hold"))
    return pred


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 评估指标
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLASSES = ["hold", "cut", "hike"]

def compute_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    """计算 accuracy / per-class precision / recall / F1 / 混淆矩阵"""
    n = len(actual)
    accuracy = (pred == actual).mean()

    conf = {a: {p: 0 for p in CLASSES} for a in CLASSES}
    for a, p in zip(actual, pred):
        if a in conf:
            conf[a][p] = conf[a].get(p, 0) + 1

    per_class = {}
    for cls in CLASSES:
        tp = conf[cls].get(cls, 0)
        fp = sum(conf[a].get(cls, 0) for a in CLASSES if a != cls)
        fn = sum(conf[cls].get(p, 0)  for p in CLASSES if p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {
            "precision": prec, "recall": rec, "f1": f1, "support": int(tp + fn)
        }

    return {"accuracy": accuracy, "n": n, "conf_matrix": conf, "per_class": per_class}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 网格搜索
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def grid_search(df_train: pd.DataFrame) -> tuple:
    """
    在训练集上穷举所有合法参数组合，最大化准确率。

    参数空间（满足 w_llm+w_var+w_market+w_pol=1，w_pol≥0）：
      w_llm    : 7 个候选值
      w_var    : 6 个候选值
      w_market : 4 个候选值
      w_pol    : 自动推导（= 1 - 其余三者之和）
      hike_thr : 6 个候选值
      cut_thr  : 6 个候选值
    约 ~3000 个有效组合，全向量化，毫秒级完成。

    返回：(best_params dict, best_accuracy float, all_results DataFrame)
    """
    llm_v    = df_train["hawkish_score"].values.astype(float)
    var_v    = df_train["var_num"].values.astype(float)
    market_v = df_train["market_num"].values.astype(float)
    pol_v    = df_train["pol_hawkish"].values.astype(float)
    actual   = df_train["label"].values

    # ── 参数网格 ───────────────────────────────────────────────────────
    grid_w_llm    = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    grid_w_var    = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    grid_w_market = [0.05, 0.10, 0.15, 0.20]
    grid_hike_thr = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    grid_cut_thr  = [-0.35, -0.30, -0.25, -0.20, -0.15, -0.10]

    best_acc    = -1.0
    best_params = None
    records     = []

    for w_llm, w_var, w_market in product(grid_w_llm, grid_w_var, grid_w_market):
        w_pol = round(1.0 - w_llm - w_var - w_market, 4)
        if w_pol < 0:
            continue

        # 预计算本组合的 fusion 向量（与阈值无关，提前计算）
        fusion = w_llm * llm_v + w_var * var_v + w_market * market_v + w_pol * pol_v

        for h_thr, c_thr in product(grid_hike_thr, grid_cut_thr):
            if c_thr >= h_thr:
                continue  # cut阈值必须小于hike阈值

            pred = np.where(fusion > h_thr, "hike",
                   np.where(fusion < c_thr, "cut", "hold"))
            acc  = float((pred == actual).mean())

            records.append({
                "w_llm": w_llm, "w_var": w_var,
                "w_market": w_market, "w_pol": w_pol,
                "hike_thr": h_thr, "cut_thr": c_thr,
                "train_acc": round(acc, 4),
            })

            if acc > best_acc:
                best_acc    = acc
                best_params = records[-1].copy()

    return best_params, best_acc, pd.DataFrame(records).sort_values("train_acc", ascending=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 综合报告
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_report(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    best_params: dict,
    train_metrics: dict,
    val_metrics: dict,
    top_results: pd.DataFrame,
):
    W = 66

    print("\n" + "★" * W)
    print("  FedWatch AI V2 — 权重优化报告")
    print("★" * W)

    print(f"""
  ── 最优参数（训练集网格搜索结果）{'─'*30}
  w_llm    = {best_params['w_llm']:.2f}    ← LLM 语义信号权重
  w_var    = {best_params['w_var']:.2f}    ← VAR 计量信号权重
  w_market = {best_params['w_market']:.2f}    ← 市场信号（收益率曲线）权重
  w_pol    = {best_params['w_pol']:.2f}    ← 政治压力信号权重
  hike_thr = {best_params['hike_thr']:+.2f}   ← 融合分 > 此值 → HIKE
  cut_thr  = {best_params['cut_thr']:+.2f}   ← 融合分 < 此值 → CUT
""")

    # 基线（永远预测 HOLD）
    n_train = len(df_train)
    n_val   = len(df_val)
    train_baseline = (df_train["label"] == "hold").mean()
    val_baseline   = (df_val["label"]   == "hold").mean()

    print(f"  ── 准确率对比{'─'*48}")
    print(f"  {'指标':<28}  {'训练集(2000-2020)':>16}  {'验证集(2021-2024)':>16}")
    print(f"  {'─'*64}")
    print(f"  {'基线（永远预测 HOLD）':<28}  {train_baseline:>16.1%}  {val_baseline:>16.1%}")
    print(f"  {'V2 优化融合模型':<28}  {train_metrics['accuracy']:>16.1%}  {val_metrics['accuracy']:>16.1%}")
    print(f"  {'超越基线':<28}  {train_metrics['accuracy']-train_baseline:>+16.1%}  {val_metrics['accuracy']-val_baseline:>+16.1%}")

    # Per-class 指标
    for label, tag in [("训练集", train_metrics), ("验证集", val_metrics)]:
        print(f"\n  ── {label} 各类别指标{'─'*44}")
        print(f"  {'类别':6}  {'精确率':>6}  {'召回率':>6}  {'F1':>6}  {'样本数':>5}")
        for cls in CLASSES:
            pc = tag["per_class"][cls]
            print(f"  {cls.upper():6}  {pc['precision']:>6.1%}  "
                  f"{pc['recall']:>6.1%}  {pc['f1']:>6.1%}  {pc['support']:>5}")

    # 验证集混淆矩阵
    print(f"\n  ── 验证集混淆矩阵（行=实际，列=预测）{'─'*24}")
    print(f"  {'':12}  {'HOLD':>6}  {'CUT':>6}  {'HIKE':>6}")
    cm = val_metrics["conf_matrix"]
    for a in CLASSES:
        row = cm.get(a, {})
        print(f"  实际-{a.upper():4}  {row.get('hold',0):>6}  "
              f"{row.get('cut',0):>6}  {row.get('hike',0):>6}")

    # 验证集逐次明细
    print(f"\n  ── 验证集逐次预测明细{'─'*42}")
    print(f"  {'日期':12}  {'主席':10}  {'预测':6}  {'实际':6}  结果")
    print("  " + "─" * 50)
    for _, r in df_val.iterrows():
        ok = "✓" if r["pred"] == r["label"] else "✗"
        print(f"  {str(r['meeting_date'])[:10]:12}  {r['chair']:10}  "
              f"{r['pred'].upper():6}  {r['label'].upper():6}  {ok}")

    # Top-10 参数组合
    print(f"\n  ── 训练集 Top-10 参数组合{'─'*38}")
    print(f"  {'w_llm':>6}  {'w_var':>5}  {'w_mkt':>5}  {'w_pol':>5}  "
          f"{'hike_t':>6}  {'cut_t':>6}  {'acc':>6}")
    for _, r in top_results.head(10).iterrows():
        print(f"  {r['w_llm']:>6.2f}  {r['w_var']:>5.2f}  "
              f"{r['w_market']:>5.2f}  {r['w_pol']:>5.2f}  "
              f"{r['hike_thr']:>+6.2f}  {r['cut_thr']:>+6.2f}  "
              f"{r['train_acc']:>6.1%}")

    print("\n" + "★" * W)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. 保存最优权重到 config.yml
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_optimal_weights(best_params: dict, val_acc: float):
    """将最优参数写入 config.yml 的 v2_optimal_weights 节"""
    from datetime import datetime

    cfg_path = _ROOT / "config.yml"
    with open(cfg_path, encoding="utf-8") as f:
        content = f.read()

    # 构造新节内容
    new_section = f"""
# ── V2 权重优化器输出（由 v2/weight_optimizer.py 自动生成）──────────
v2_optimal_weights:
  w_llm:      {best_params['w_llm']}    # LLM 语义信号
  w_var:      {best_params['w_var']}    # VAR 计量信号
  w_market:   {best_params['w_market']}    # 市场信号（收益率曲线）
  w_pol:      {best_params['w_pol']}    # 政治压力信号
  hike_thr:   {best_params['hike_thr']}   # fusion > 此值 → HIKE
  cut_thr:    {best_params['cut_thr']}   # fusion < 此值 → CUT
  train_acc:  {best_params['train_acc']}  # 训练集准确率（2000-2020）
  val_acc:    {round(val_acc, 4)}  # 验证集准确率（2021-2024）
  optimized_at: "{datetime.now().strftime('%Y-%m-%d')}"
"""
    # 替换已有节或追加
    import re
    if "v2_optimal_weights:" in content:
        content = re.sub(
            r"\n# ── V2 权重优化器.*?(?=\n#|\Z)", new_section, content, flags=re.DOTALL
        )
    else:
        content += new_section

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [✓] 最优权重已保存到 config.yml (v2_optimal_weights)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_optimizer():
    print("\n" + "=" * 66)
    print("  FedWatch AI V2 — 融合权重优化器 启动")
    print("=" * 66)

    # ── Step 1：加载特征矩阵 ───────────────────────────────────────────
    print("\n[1/5] 加载特征矩阵（fomc_history + semantic_history + political_quant）...")
    df = load_feature_matrix()
    df_train = df[df["split"] == "train"].copy().reset_index(drop=True)
    df_val   = df[df["split"] == "test"].copy().reset_index(drop=True)
    print(f"  训练集：{len(df_train)} 次会议  |  验证集：{len(df_val)} 次会议")

    # ── Step 2：拉取宏观时间序列（VAR 所需）───────────────────────────
    print("\n[2/5] 获取宏观时间序列（供 VAR 走一步预测使用）...")
    df_macro = fetch_full_macro_series()

    # 无 FRED key 时，用 fomc_history 的存档数据兜底
    if df_macro.empty:
        print("  [降级] 使用 fomc_history 存档数据")
        conn = sqlite3.connect(DB_PATH)
        df_macro = pd.read_sql(
            "SELECT meeting_date, fed_funds_rate as fedfunds, "
            "cpi_yoy, unemployment as unrate FROM fomc_history ORDER BY meeting_date",
            conn,
        )
        conn.close()
        df_macro["meeting_date"] = pd.to_datetime(df_macro["meeting_date"])
        df_macro = df_macro.set_index("meeting_date").dropna()

    # ── Step 3：预计算所有会议的 VAR 信号（最耗时部分，约1-3分钟）──────
    print(f"\n[3/5] VAR 走一步预测（共 {len(df)} 次，约 1-3 分钟）...")
    all_var = compute_var_signals(df, df_macro)
    df = add_numeric_signals(df, all_var)
    df_train = df[df["split"] == "train"].copy().reset_index(drop=True)
    df_val   = df[df["split"] == "test"].copy().reset_index(drop=True)

    # VAR 信号分布
    var_series = pd.Series(all_var)
    print(f"  VAR 信号分布：{var_series.value_counts().to_dict()}")

    # ── Step 4：网格搜索（纯内存计算，秒级完成）────────────────────────
    print("\n[4/5] 网格搜索最优融合权重...")
    best_params, best_train_acc, results_df = grid_search(df_train)
    valid_combos = len(results_df)
    print(f"  搜索了 {valid_combos:,} 个合法参数组合")
    print(f"  最优训练集准确率：{best_train_acc:.1%}")
    print(f"  最优参数：{best_params}")

    # ── Step 5：在验证集上评估 ────────────────────────────────────────
    print("\n[5/5] 在验证集（2021-2024）上评估...")
    val_pred = fuse_predict(
        df_val["hawkish_score"].values.astype(float),
        df_val["var_num"].values.astype(float),
        df_val["market_num"].values.astype(float),
        df_val["pol_hawkish"].values.astype(float),
        best_params["w_llm"], best_params["w_var"],
        best_params["w_market"], best_params["w_pol"],
        best_params["hike_thr"], best_params["cut_thr"],
    )
    df_val["pred"] = val_pred

    train_pred = fuse_predict(
        df_train["hawkish_score"].values.astype(float),
        df_train["var_num"].values.astype(float),
        df_train["market_num"].values.astype(float),
        df_train["pol_hawkish"].values.astype(float),
        best_params["w_llm"], best_params["w_var"],
        best_params["w_market"], best_params["w_pol"],
        best_params["hike_thr"], best_params["cut_thr"],
    )
    df_train["pred"] = train_pred

    train_metrics = compute_metrics(df_train["label"].values, train_pred)
    val_metrics   = compute_metrics(df_val["label"].values,   val_pred)

    print(f"  验证集准确率：{val_metrics['accuracy']:.1%}")

    # ── 打印完整报告 ──────────────────────────────────────────────────
    print_report(df_train, df_val, best_params, train_metrics, val_metrics,
                 results_df.head(20))

    # ── 保存最优权重 ──────────────────────────────────────────────────
    save_optimal_weights(best_params, val_metrics["accuracy"])

    return best_params, train_metrics, val_metrics


if __name__ == "__main__":
    run_optimizer()
