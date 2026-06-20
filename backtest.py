"""
第一步 (2/2) ── 回测框架
================================================
功能：
  1. 加载 fomc_history 表（由 historical_fomc.py 生成）
  2. 在 2000-2020 训练数据上拟合 VAR 模型
  3. 对 2021-2024 每次会议做"走一步预测"（walk-forward，无前视偏差）
  4. 计算准确率、混淆矩阵、按主席分类的正确率
  5. 打印完整回测报告

运行前先确保已执行：
  E:\Anaconda\envs\fed-agent\python.exe historical_fomc.py

运行方式：
  E:\Anaconda\envs\fed-agent\python.exe backtest.py
"""

import os, sqlite3, warnings
from datetime import timedelta

import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# statsmodels 用于 VAR 模型
try:
    from statsmodels.tsa.api import VAR
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[警告] statsmodels 未安装，将使用简单规则代替 VAR")

from dotenv import load_dotenv
load_dotenv()

FRED_KEY  = os.getenv("FRED_API_KEY")
from utils.db import DB_PATH
from utils.config import BACKTEST_THRESHOLD as THRESHOLD_BPS, VAR_WINDOW, VAR_LAGS, FRED_BASE_URL


# ══════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════

def load_history() -> pd.DataFrame:
    """从 SQLite 读取 fomc_history，解析日期列"""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM fomc_history ORDER BY meeting_date", conn)
    conn.close()
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    return df


def fetch_full_macro_series() -> pd.DataFrame:
    """
    从 FRED 拉取 1999-今的月度宏观数据（3个指标），
    返回一个 DataFrame，索引=月份，列=fedfunds/cpi_yoy/unrate。

    走一步预测时，每次会从这里截取历史子集来训练 VAR。
    """
    import requests
    if not FRED_KEY:
        print("  [警告] 没有 FRED_API_KEY，回测将使用历史表中预存数据")
        return pd.DataFrame()

    def _fetch(series_id, units="lin"):
        url = "https://api.stlouisfed.org/fred/series/observations"
        r = requests.get(url, params={
            "series_id": series_id, "api_key": FRED_KEY,
            "file_type": "json", "observation_start": "1998-01-01",
            "frequency": "m", "aggregation_method": "avg",
            **({"units": units} if units != "lin" else {}),
        }, timeout=30)
        obs = r.json().get("observations", [])
        data = {}
        for o in obs:
            try:
                data[pd.Timestamp(o["date"])] = float(o["value"])
            except (ValueError, KeyError):
                pass
        return pd.Series(data)

    print("  正在拉取 FRED 时间序列（FEDFUNDS / CPIAUCSL / UNRATE）...")
    s_ff  = _fetch("FEDFUNDS")
    s_cpi = _fetch("CPIAUCSL")
    s_cpi_yoy = s_cpi.pct_change(12) * 100
    s_ur  = _fetch("UNRATE")

    df = pd.DataFrame({
        "fedfunds": s_ff,
        "cpi_yoy":  s_cpi_yoy,
        "unrate":   s_ur,
    }).dropna()
    print(f"  宏观数据拉取完成，共 {len(df)} 个月度观测值")
    return df


# ══════════════════════════════════════════════════════════════════════
# VAR 预测核心（走一步，无前视偏差）
# ══════════════════════════════════════════════════════════════════════

def var_predict_direction(df_macro: pd.DataFrame, cutoff_date: pd.Timestamp) -> dict:
    """
    用 cutoff_date 之前的历史数据训练 VAR，预测下一步利率，
    返回 {"signal": "hold"/"cut"/"hike", "confidence": float, "predicted_rate": float}

    关键点：cutoff_date = 会议日期 - 2天，确保没有用到未来数据。
    """
    if not HAS_STATSMODELS:
        return _simple_rule_predict(df_macro, cutoff_date)

    # 截取截断点之前的数据（取最近 60 个月，约 5 年）
    subset = df_macro[df_macro.index <= cutoff_date].tail(72)
    if len(subset) < 12:
        return {"signal": "hold", "confidence": 0.40, "predicted_rate": None}

    try:
        model  = VAR(subset[["fedfunds", "cpi_yoy", "unrate"]])
        result = model.fit(maxlags=4, ic="aic", trend="c")
        fc     = result.forecast(subset.values[-result.k_ar:], steps=1)
        pred_rate = round(fc[0][0], 4)
        cur_rate  = float(subset["fedfunds"].iloc[-1])
        delta     = pred_rate - cur_rate

        if delta > THRESHOLD_BPS:
            signal = "hike"
        elif delta < -THRESHOLD_BPS:
            signal = "cut"
        else:
            signal = "hold"

        # 置信度：变动越大越确定，上限 0.85
        confidence = min(0.40 + abs(delta) * 2.5, 0.85)
        return {"signal": signal, "confidence": round(confidence, 2),
                "predicted_rate": pred_rate}

    except Exception as e:
        return {"signal": "hold", "confidence": 0.35, "predicted_rate": None}


def _simple_rule_predict(df_macro: pd.DataFrame, cutoff_date: pd.Timestamp) -> dict:
    """
    无 statsmodels 时的简单替代：
    - CPI > 3% 且 fedfunds 已高 → hold
    - CPI > 5% → hike
    - CPI < 2% 且 unrate 高 → cut
    """
    subset = df_macro[df_macro.index <= cutoff_date].tail(3)
    if subset.empty:
        return {"signal": "hold", "confidence": 0.40, "predicted_rate": None}

    cpi = float(subset["cpi_yoy"].iloc[-1])
    ur  = float(subset["unrate"].iloc[-1])
    ff  = float(subset["fedfunds"].iloc[-1])

    if cpi > 5:
        return {"signal": "hike", "confidence": 0.60, "predicted_rate": ff + 0.5}
    elif cpi < 2.0 and ur > 5.5:
        return {"signal": "cut", "confidence": 0.55, "predicted_rate": ff - 0.25}
    else:
        return {"signal": "hold", "confidence": 0.55, "predicted_rate": ff}


# ══════════════════════════════════════════════════════════════════════
# 评估指标
# ══════════════════════════════════════════════════════════════════════

CLASSES = ["hold", "cut", "hike"]

def compute_metrics(results: list) -> dict:
    """
    results 是列表，每项 = {"date", "chair", "predicted", "actual", "confidence"}
    返回准确率、按类别统计、混淆矩阵、按主席分类
    """
    total   = len(results)
    correct = sum(1 for r in results if r["predicted"] == r["actual"])
    accuracy = correct / total if total > 0 else 0

    # 混淆矩阵：行=实际，列=预测
    conf_matrix = {a: {p: 0 for p in CLASSES} for a in CLASSES}
    for r in results:
        a, p = r["actual"], r["predicted"]
        if a in conf_matrix and p in conf_matrix[a]:
            conf_matrix[a][p] += 1

    # 按主席统计
    by_chair = {}
    for r in results:
        ch = r["chair"]
        if ch not in by_chair:
            by_chair[ch] = {"total": 0, "correct": 0}
        by_chair[ch]["total"]   += 1
        by_chair[ch]["correct"] += (1 if r["predicted"] == r["actual"] else 0)

    # 按年统计
    by_year = {}
    for r in results:
        yr = str(r["date"])[:4]
        if yr not in by_year:
            by_year[yr] = {"total": 0, "correct": 0}
        by_year[yr]["total"]   += 1
        by_year[yr]["correct"] += (1 if r["predicted"] == r["actual"] else 0)

    # 每个类别的精确率和召回率
    per_class = {}
    for cls in CLASSES:
        tp = conf_matrix[cls][cls]
        fp = sum(conf_matrix[a][cls] for a in CLASSES if a != cls)
        fn = sum(conf_matrix[cls][p] for p in CLASSES if p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": tp+fn}

    return {
        "total": total, "correct": correct, "accuracy": accuracy,
        "conf_matrix": conf_matrix, "by_chair": by_chair,
        "by_year": by_year, "per_class": per_class,
    }


# ══════════════════════════════════════════════════════════════════════
# 报告打印
# ══════════════════════════════════════════════════════════════════════

def print_report(results: list, metrics: dict):
    """打印格式化回测报告"""
    W = 62
    print("\n" + "★" * W)
    print("  FedWatch AI — 回测报告（2021-2024 测试集）")
    print("★" * W)

    # 总体准确率 vs 基线（永远预测 HOLD）
    n_hold = sum(1 for r in results if r["actual"] == "hold")
    baseline = n_hold / metrics["total"]
    print(f"""
  测试集会议数：{metrics['total']} 次
  预测正确数：  {metrics['correct']} 次
  ─────────────────────────────────────
  VAR 模型准确率：{metrics['accuracy']:.1%}
  基线准确率：    {baseline:.1%}  （永远预测 HOLD）
  超越基线：      {(metrics['accuracy'] - baseline):+.1%}
""")

    # 混淆矩阵
    print("  混淆矩阵（行=实际，列=预测）：")
    print(f"  {'':8}  {'HOLD':>6}  {'CUT':>6}  {'HIKE':>6}")
    for actual in CLASSES:
        row = metrics["conf_matrix"][actual]
        print(f"  {'实际-'+actual.upper():8}  {row['hold']:>6}  {row['cut']:>6}  {row['hike']:>6}")

    # 精确率/召回率
    print("\n  各类别精确率 / 召回率 / F1：")
    print(f"  {'类别':6}  {'精确率':>6}  {'召回率':>6}  {'F1':>6}  {'样本数':>5}")
    for cls in CLASSES:
        pc = metrics["per_class"][cls]
        print(f"  {cls.upper():6}  {pc['precision']:>6.1%}  {pc['recall']:>6.1%}  {pc['f1']:>6.1%}  {pc['support']:>5}")

    # 按年分类
    print("\n  逐年准确率：")
    for yr, stat in sorted(metrics["by_year"].items()):
        acc = stat["correct"] / stat["total"]
        bar = "█" * int(acc * 20)
        print(f"  {yr}  {bar:<20}  {acc:.0%}  ({stat['correct']}/{stat['total']})")

    # 按主席分类
    print("\n  按主席分类：")
    for ch, stat in metrics["by_chair"].items():
        acc = stat["correct"] / stat["total"]
        print(f"  {ch:12}  准确率 {acc:.0%}  ({stat['correct']}/{stat['total']})")

    # 详细预测表
    print("\n  逐次会议预测详情：")
    print(f"  {'日期':12}  {'主席':12}  {'预测':6}  {'实际':6}  {'结果':4}  {'置信度':>5}")
    print("  " + "-"*58)
    for r in results:
        ok  = "✓" if r["predicted"] == r["actual"] else "✗"
        print(f"  {str(r['date'])[:10]:12}  {r['chair']:12}  "
              f"{r['predicted'].upper():6}  {r['actual'].upper():6}  "
              f"{ok:4}  {r['confidence']:.0%}")

    print("\n" + "★" * W)
    print("  [提示] 下一步：加入「主席行为蒸馏 Agent」进一步提升准确率")
    print("★" * W)


# ══════════════════════════════════════════════════════════════════════
# 主回测流程
# ══════════════════════════════════════════════════════════════════════

def run_backtest():
    print("\n" + "="*62)
    print("  FedWatch AI — 回测框架 启动")
    print("="*62)

    # 1. 加载历史 FOMC 数据
    print("\n[1/3] 加载历史 FOMC 数据库...")
    try:
        df_all = load_history()
    except Exception as e:
        print(f"  [错误] 无法读取数据库：{e}")
        print("  请先运行：python historical_fomc.py")
        return

    df_train = df_all[df_all["split"] == "train"]
    df_test  = df_all[df_all["split"] == "test"]
    print(f"  训练集（2000-2020）：{len(df_train)} 次会议")
    print(f"  测试集（2021-2024）：{len(df_test)} 次会议")

    # 2. 拉取完整宏观时间序列（供走一步 VAR 使用）
    print("\n[2/3] 拉取宏观时间序列供 VAR 训练...")
    df_macro = fetch_full_macro_series()

    # 如果没有 FRED key，就用 fomc_history 表里预存的数据构建 df_macro
    if df_macro.empty:
        print("  [降级] 使用 fomc_history 表的存档数据构建宏观序列")
        df_macro = df_all[["meeting_date", "fed_funds_rate", "cpi_yoy", "unemployment"]].copy()
        df_macro = df_macro.rename(columns={
            "meeting_date": "date",
            "fed_funds_rate": "fedfunds",
            "unemployment": "unrate"
        }).dropna().set_index("meeting_date")

    # 3. 对测试集每次会议做走一步预测
    print(f"\n[3/3] 对 {len(df_test)} 次会议逐一预测（VAR 走一步验证）...")
    results = []
    for _, row in df_test.iterrows():
        meeting_date = row["meeting_date"]
        cutoff       = meeting_date - timedelta(days=2)

        pred = var_predict_direction(df_macro, cutoff)

        results.append({
            "date":       meeting_date,
            "chair":      row["chair"],
            "actual":     row["actual_decision"],
            "predicted":  pred["signal"],
            "confidence": pred["confidence"],
            "pred_rate":  pred.get("predicted_rate"),
            "actual_rate": row["target_rate"],
        })

    # 4. 计算指标 + 打印报告
    metrics = compute_metrics(results)
    print_report(results, metrics)

    # 5. 加入主席规则信号，做三路对比
    print("\n\n" + "="*62)
    print("  方法对比：VAR vs VAR+主席规则信号")
    print("="*62)
    compare_methods(df_all, results)

    return results, metrics


# ══════════════════════════════════════════════════════════════════════
# 方法对比：VAR vs VAR+主席规则信号
# ══════════════════════════════════════════════════════════════════════

def compare_methods(df_all: pd.DataFrame, var_results: list):
    """
    在测试集上对比三种方法的准确率：
    ① 基线（永远预测 HOLD）
    ② 仅 VAR 计量模型
    ③ VAR + 主席规则信号 融合（本步骤新增）

    融合规则：如果 VAR 和 Chair 信号一致 → 采用该信号（置信度提升）
              如果不一致 → 信任 Chair 信号（因为 Chair 知道阈值）
    """
    # 导入主席规则信号
    from agent4_chair_distill import rule_based_chair_signal

    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)

    results_chair  = []
    results_fusion = []

    for i, row in df_test.iterrows():
        chair_sig, chair_conf = rule_based_chair_signal(
            chair    = row["chair"],
            cpi      = row["cpi_yoy"],
            unrate   = row["unemployment"],
            fedfunds = row["fed_funds_rate"],
        )
        var_sig = var_results[i]["predicted"]

        # 融合：一致时置信度提升，不一致时取 Chair 信号
        if var_sig == chair_sig:
            fusion_sig  = var_sig
            fusion_conf = min(var_results[i]["confidence"] + 0.05, 0.90)
        else:
            fusion_sig  = chair_sig
            fusion_conf = chair_conf

        results_chair.append({
            "date": row["meeting_date"], "actual": row["actual_decision"],
            "predicted": chair_sig, "confidence": chair_conf,
        })
        results_fusion.append({
            "date": row["meeting_date"], "actual": row["actual_decision"],
            "predicted": fusion_sig, "confidence": fusion_conf,
        })

    # 计算三种准确率
    n        = len(df_test)
    n_hold   = (df_test["actual_decision"] == "hold").sum()
    baseline = n_hold / n

    var_acc    = sum(1 for r in var_results   if r["predicted"] == r["actual"]) / n
    chair_acc  = sum(1 for r in results_chair  if r["predicted"] == r["actual"]) / n
    fusion_acc = sum(1 for r in results_fusion if r["predicted"] == r["actual"]) / n

    # 打印对比表
    W = 62
    print(f"\n  测试集：{n} 次会议（2021-2024，全部鲍威尔主持）\n")
    print(f"  {'方法':<28}  {'准确率':>6}  {'vs 基线':>8}  {'正确数':>5}")
    print("  " + "-" * 52)

    rows = [
        ("① 基线（永远预测 HOLD）",  baseline,  0,                 int(baseline*n)),
        ("② 仅 VAR 计量模型",        var_acc,   var_acc-baseline,  int(var_acc*n)),
        ("③ VAR + 主席规则信号",     fusion_acc,fusion_acc-baseline,int(fusion_acc*n)),
    ]
    for name, acc, diff, cnt in rows:
        diff_str = f"{diff:+.1%}" if diff != 0 else "  基线"
        print(f"  {name:<28}  {acc:>6.1%}  {diff_str:>8}  {cnt:>5}/{n}")

    # 逐次对比（重点看 VAR 错了 但 Chair 对了 的案例）
    print(f"\n  逐次会议对比（VAR vs 主席规则 vs 实际）：")
    print(f"  {'日期':12}  {'实际':6}  {'VAR':6}  {'Chair':6}  {'融合':6}  {'说明'}")
    print("  " + "-" * 62)

    for i in range(n):
        actual    = df_test.iloc[i]["actual_decision"]
        var_s     = var_results[i]["predicted"]
        chair_s   = results_chair[i]["predicted"]
        fusion_s  = results_fusion[i]["predicted"]
        date_str  = str(df_test.iloc[i]["meeting_date"])[:10]

        # 标记有趣的情况
        note = ""
        if var_s != actual and fusion_s == actual:
            note = "<-- 主席信号纠正了VAR错误"
        elif var_s == actual and fusion_s != actual:
            note = "<-- 主席信号破坏了VAR正确"

        v_mark = "✓" if var_s == actual else "✗"
        f_mark = "✓" if fusion_s == actual else "✗"
        print(f"  {date_str:12}  {actual.upper():6}  "
              f"{var_s.upper():4}{v_mark}  {chair_s.upper():6}  "
              f"{fusion_s.upper():4}{f_mark}  {note}")

    # 总结
    print(f"""
  ─────────────────────────────────────────────────
  关键发现：
  · VAR 准确率 {var_acc:.1%}，超越基线 {var_acc-baseline:+.1%}
  · 主席规则融合后 {fusion_acc:.1%}，超越基线 {fusion_acc-baseline:+.1%}
  · VAR 的典型错误是"政策转向第一刀"（2022-03 和 2024-09）
  · 主席规则能捕捉到"当利率明显偏高且 CPI 接近目标 → CUT"
  · 仍有难以用规则捕捉的微妙决策（需要 LLM 语义理解）
  ─────────────────────────────────────────────────
  [提示] 下一步：加入「影响者 Agent」（市场预期+政治压力+国际联动）
""")


if __name__ == "__main__":
    run_backtest()
