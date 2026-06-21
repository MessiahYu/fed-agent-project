"""
v4/weight_optimizer_v4.py
============================================================
V4 核心改进：用【上一次】会议声明预测【本次】会议决策
============================================================

V3 的方法论问题：
  FOMC 声明与利率决议同时发布（美东 14:00），声明是对已做出决策的
  解释，并非对下次决策的预测。用当次声明预测当次决策，存在"同期信息泄露"。

V4 的修正：
  对 hawkish_score 滞后一期（shift(1)）：
    第 N 次会议的 LLM 特征 = 第 N-1 次会议声明的分析结果
  这样 LLM 信号才是真正公开可得的前瞻信号。

为什么基于 V2 而非 V3：
  V3 使用分层随机划分，测试集中的会议在时间上不连续（散布在 2000-2024
  各处），"上一次会议"的概念不够自然。V2 的时间切割保证每个集合内的
  会议是连续的时间序列，lag 含义清晰，与实际预测场景一致。

数据划分（V2 时间切割，在代码中显式重建，不依赖数据库 split 列）：
  训练集：meeting_date < 2021-01-01（约 174 次，2000-2020）
  测试集：2021-01-01 ≤ meeting_date < 2025-01-01（32 次，2021-2024）
  最终测试集：meeting_date ≥ 2025-01-01（12 次，全程封存）

其余完全沿用 V2/V3：
  - 四路信号：LLM + VAR + 市场 + 政治
  - 6012 个参数组合网格搜索
  - 最优权重写入 config.yml → v4_optimal_weights

运行方式：
  E:\\Anaconda\\envs\\fed-agent\\python.exe v4/weight_optimizer_v4.py
"""

import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from v2.weight_optimizer import (
    load_feature_matrix,
    compute_var_signals,
    add_numeric_signals,
    fuse_predict,
    CLASSES,
)
from v2.backtest import fetch_full_macro_series

_cfg = yaml.safe_load(open(str(_ROOT / "config.yml"), encoding="utf-8"))

# ── 数据划分边界（V2 时间切割） ─────────────────────────────────────────
TRAIN_END   = "2021-01-01"   # < 此日期 → train
TEST_END    = "2025-01-01"   # < 此日期 → test
# ≥ TEST_END → final_test（不参与训练/验证，最终一次性评估）


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# V4 特征矩阵：重建 V2 时间切割 + LLM 信号滞后一期
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_feature_matrix_v4() -> pd.DataFrame:
    """
    1. 加载全部会议特征（与 V2/V3 相同）
    2. 在代码中按日期重新分配 split（V2 时间切割），
       覆盖数据库里可能存在的 V3 随机 split 标签
    3. 对 LLM 语义信号整体滞后一期

    关键性质：
    - 第 N 次会议的 hawkish_score = 第 N-1 次会议声明分析结果
    - 第一次会议（无前序）丢弃
    - 跨 split 边界的 lag 是合法的：
        2021-01-27（test 第 1 次）的 LLM 特征
        = 2020-12-16（train 最后 1 次）的声明分析结果
        这在实际中完全可得（声明已公开）
    """
    df = load_feature_matrix()   # 从数据库加载，含 train/test/final_test

    # ① 按日期严格排序（shift(1) 依赖正确的时间顺序）
    df = df.sort_values("meeting_ts").reset_index(drop=True)

    # ② 在代码中重建 V2 时间切割，覆盖数据库的 V3 随机标签
    df["split"] = "train"
    df.loc[df["meeting_date"] >= TRAIN_END, "split"] = "test"
    df.loc[df["meeting_date"] >= TEST_END,  "split"] = "final_test"

    # ③ LLM 信号滞后一期（核心改动）
    for col in ["hawkish_score", "inflation_concern", "growth_concern"]:
        if col in df.columns:
            df[col] = df[col].shift(1)

    # ④ 丢弃第一行（无前序声明）
    before = len(df)
    df = df.dropna(subset=["hawkish_score"]).reset_index(drop=True)
    print(f"  [V4] LLM 信号已滞后一期，丢弃首行（无前序声明），"
          f"剩余 {len(df)} 次（原 {before} 次）")

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 评估指标
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_metrics(actual, pred) -> dict:
    n = len(actual)
    accuracy = (actual == pred).mean()
    per_class = {}
    for cls in CLASSES:
        tp = ((pred == cls) & (actual == cls)).sum()
        fp = ((pred == cls) & (actual != cls)).sum()
        fn = ((pred != cls) & (actual == cls)).sum()
        support = (actual == cls).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else float("nan"))
        per_class[cls] = dict(p=precision, r=recall, f1=f1,
                               support=int(support), tp=int(tp))
    return dict(accuracy=accuracy, per_class=per_class, n=n)


def print_metrics(label: str, m: dict):
    print(f"\n  ── {label} 各类别指标 ─────────────────────────────────")
    print(f"  {'类别':6}  {'精确率':>8}  {'召回率':>8}  {'F1':>8}  {'样本数':>6}")
    for cls in CLASSES:
        c = m["per_class"][cls]
        if c["support"] == 0:
            print(f"  {cls.upper():6}  {'—':>8}  {'—':>8}  {'—':>8}  {0:>6}")
        else:
            p = f"{c['p']:.1%}" if not np.isnan(c['p']) else "—"
            r = f"{c['r']:.1%}" if not np.isnan(c['r']) else "—"
            f = f"{c['f1']:.1%}" if not np.isnan(c['f1']) else "—"
            print(f"  {cls.upper():6}  {p:>8}  {r:>8}  {f:>8}  {c['support']:>6}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 网格搜索
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def grid_search(df_train: pd.DataFrame) -> dict:
    llm_v = df_train["hawkish_score"].values.astype(float)
    var_v = df_train["var_num"].values.astype(float)
    mkt_v = df_train["market_num"].values.astype(float)
    pol_v = df_train["pol_hawkish"].values.astype(float)
    lbl   = df_train["label"].values

    grid_w_llm    = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    grid_w_var    = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    grid_w_market = [0.05, 0.10, 0.15, 0.20]
    grid_hike_thr = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    grid_cut_thr  = [-0.35, -0.30, -0.25, -0.20, -0.15, -0.10]

    best_acc, best_params, valid = -1, None, 0
    for w_llm, w_var, w_market in product(grid_w_llm, grid_w_var, grid_w_market):
        w_pol = round(1.0 - w_llm - w_var - w_market, 4)
        if w_pol < 0:
            continue
        fusion = w_llm*llm_v + w_var*var_v + w_market*mkt_v + w_pol*pol_v
        for h_thr, c_thr in product(grid_hike_thr, grid_cut_thr):
            if c_thr >= h_thr:
                continue
            valid += 1
            pred = np.where(fusion > h_thr, "hike",
                   np.where(fusion < c_thr, "cut", "hold"))
            acc = (pred == lbl).mean()
            if acc > best_acc:
                best_acc = acc
                best_params = dict(
                    w_llm=w_llm, w_var=w_var, w_market=w_market, w_pol=w_pol,
                    hike_thr=h_thr, cut_thr=c_thr, train_acc=round(acc, 4),
                )
    print(f"  遍历 {valid} 个有效参数组合")
    return best_params


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("\n" + "★" * 65)
    print("  FedWatch AI V4 — 权重优化（V2 时间切割 + LLM 滞后一期）")
    print("★" * 65)

    print("\n[1/4] 加载 V4 特征矩阵...")
    df_all = load_feature_matrix_v4()
    df_tr  = df_all[df_all["split"] == "train"].copy().reset_index(drop=True)
    df_val = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)
    print(f"  训练集：{len(df_tr)} 次  测试集：{len(df_val)} 次")

    print("\n[2/4] 拉取宏观数据并计算 VAR 信号...")
    df_macro = fetch_full_macro_series()
    df_tr  = add_numeric_signals(df_tr,  compute_var_signals(df_tr,  df_macro))
    df_val = add_numeric_signals(df_val, compute_var_signals(df_val, df_macro))

    print("\n[3/4] 网格搜索（训练集）...")
    best = grid_search(df_tr)

    # 测试集评估
    def predict(df, p):
        return fuse_predict(
            df["hawkish_score"].values.astype(float),
            df["var_num"].values.astype(float),
            df["market_num"].values.astype(float),
            df["pol_hawkish"].values.astype(float),
            p["w_llm"], p["w_var"], p["w_market"], p["w_pol"],
            p["hike_thr"], p["cut_thr"],
        )

    tr_pred  = predict(df_tr, best)
    val_pred = predict(df_val, best)
    tr_actual  = df_tr["label"].values
    val_actual = df_val["label"].values
    val_acc = (val_pred == val_actual).mean()

    tr_base  = (tr_actual  == "hold").mean()
    val_base = (val_actual == "hold").mean()

    print("\n[4/4] 结果")
    print("★" * 65)
    print(f"""
  ── 最优参数 ───────────────────────────────────────────────
  w_llm    = {best['w_llm']}  （LLM 上次声明鹰鸽分，滞后一期）
  w_var    = {best['w_var']}  （VAR 计量信号）
  w_market = {best['w_market']}  （市场信号）
  w_pol    = {best['w_pol']}  （政治压力信号）
  hike_thr = {best['hike_thr']:+.2f}
  cut_thr  = {best['cut_thr']:+.2f}

  ── 准确率 ─────────────────────────────────────────────────
                       训练集({len(df_tr)}次)    测试集({len(df_val)}次)
  基线（永远 HOLD）      {tr_base:.1%}          {val_base:.1%}
  V4 模型               {best['train_acc']:.1%}          {val_acc:.1%}
  超越基线              {best['train_acc']-tr_base:+.1%}         {val_acc-val_base:+.1%}
""")

    print_metrics(f"训练集（{len(df_tr)}次）", compute_metrics(tr_actual, tr_pred))
    print_metrics(f"测试集（{len(df_val)}次）", compute_metrics(val_actual, val_pred))

    # 与 V2 / V3 对比
    v2 = _cfg.get("v2_optimal_weights", {})
    v3 = _cfg.get("v3_optimal_weights", {})
    if v2 and v3:
        print(f"""
  ── 跨版本权重对比 ─────────────────────────────────────────
  参数        V2      V3      V4     说明
  w_llm     {v2['w_llm']:.2f}   {v3['w_llm']:.2f}   {best['w_llm']:.2f}
  w_var     {v2['w_var']:.2f}   {v3['w_var']:.2f}   {best['w_var']:.2f}
  w_market  {v2['w_market']:.2f}   {v3['w_market']:.2f}   {best['w_market']:.2f}
  w_pol     {v2['w_pol']:.2f}   {v3['w_pol']:.2f}   {best['w_pol']:.2f}
  hike_thr  {v2['hike_thr']:+.2f}  {v3['hike_thr']:+.2f}  {best['hike_thr']:+.2f}
  cut_thr   {v2['cut_thr']:+.2f}  {v3['cut_thr']:+.2f}  {best['cut_thr']:+.2f}
  val_acc   {v2['val_acc']:.1%}  {v3['val_acc']:.1%}  {val_acc:.1%}  ← V4 基于V2划分
""")

    # 保存到 config.yml
    cfg = yaml.safe_load(open(str(_ROOT / "config.yml"), encoding="utf-8"))
    cfg["v4_optimal_weights"] = {
        "w_llm":        float(best["w_llm"]),
        "w_var":        float(best["w_var"]),
        "w_market":     float(best["w_market"]),
        "w_pol":        float(best["w_pol"]),
        "hike_thr":     float(best["hike_thr"]),
        "cut_thr":      float(best["cut_thr"]),
        "train_acc":    round(float(best["train_acc"]), 4),
        "val_acc":      round(float(val_acc), 4),
        "split_method": "time_cut_v2_llm_lag1",
        "llm_lag":      1,
        "optimized_at": "2026-06-21",
    }
    with open(str(_ROOT / "config.yml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    print("\n  [✓] V4 最优权重已保存到 config.yml (v4_optimal_weights)")


if __name__ == "__main__":
    main()
