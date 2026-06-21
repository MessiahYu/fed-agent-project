"""
v3/metrics_summary.py
============================================================
汇总 V2 / V3 模型在三个数据集上的 Precision / Recall / Accuracy。
V1 为实时单次预测系统，无历史批量回测，不参与统计。

运行方式：
  E:\\Anaconda\\envs\\fed-agent\\python.exe v3/metrics_summary.py
"""

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

CLASSES = ["hold", "hike", "cut"]

# ── 通用指标计算 ──────────────────────────────────────────────────────────

def compute_all_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual = np.array([s.lower() for s in actual])
    pred   = np.array([s.lower() for s in pred])
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
        per_class[cls] = dict(precision=precision, recall=recall,
                               f1=f1, support=int(support), tp=int(tp))

    # macro 平均（只对出现过的类别计算）
    active = [m for cls, m in per_class.items() if per_class[cls]["support"] > 0]
    macro_p = np.nanmean([m["precision"] for m in active])
    macro_r = np.nanmean([m["recall"]    for m in active])
    macro_f = np.nanmean([m["f1"]        for m in active])

    return dict(accuracy=accuracy, per_class=per_class,
                macro_p=macro_p, macro_r=macro_r, macro_f=macro_f, n=n)


def print_metrics(label: str, metrics: dict):
    acc  = metrics["accuracy"]
    n    = metrics["n"]
    pc   = metrics["per_class"]
    print(f"\n  {'类别':6}  {'精确率(P)':>10}  {'召回率(R)':>10}  {'F1':>8}  {'样本数':>6}  {'猜对':>4}")
    print("  " + "─" * 54)
    for cls in CLASSES:
        m = pc[cls]
        if m["support"] == 0:
            print(f"  {cls.upper():6}  {'—':>10}  {'—':>10}  {'—':>8}  {0:>6}  {'—':>4}")
        else:
            p_str = f"{m['precision']:.1%}" if not np.isnan(m['precision']) else "—"
            r_str = f"{m['recall']:.1%}"    if not np.isnan(m['recall'])    else "—"
            f_str = f"{m['f1']:.1%}"        if not np.isnan(m['f1'])        else "—"
            print(f"  {cls.upper():6}  {p_str:>10}  {r_str:>10}  {f_str:>8}  "
                  f"{m['support']:>6}  {m['tp']:>4}")
    print("  " + "─" * 54)
    print(f"  {'Macro':6}  {metrics['macro_p']:>10.1%}  {metrics['macro_r']:>10.1%}  "
          f"{metrics['macro_f']:>8.1%}")
    print(f"  Accuracy: {acc:.1%}  ({int(round(acc*n))}/{n})")


# ── 数据加载 ──────────────────────────────────────────────────────────────

def load_historical_test(split_label: str) -> tuple:
    """
    从数据库重新计算历史测试集指标。
    split_label='test' 即当前 DB 中的 split 列（V3 随机划分后）。
    """
    from v2.weight_optimizer import (
        load_feature_matrix, compute_var_signals,
        add_numeric_signals, fuse_predict
    )
    from v2.backtest import fetch_full_macro_series
    import yaml
    cfg = yaml.safe_load(open(str(_ROOT / "config.yml"), encoding="utf-8"))

    df_all = load_feature_matrix()
    df = df_all[df_all["split"] == split_label].copy().reset_index(drop=True)
    df_macro = fetch_full_macro_series()
    var_sigs = compute_var_signals(df, df_macro)
    df = add_numeric_signals(df, var_sigs)

    llm_v = df["hawkish_score"].values.astype(float)
    var_v = df["var_num"].values.astype(float)
    mkt_v = df["market_num"].values.astype(float)
    pol_v = df["pol_hawkish"].values.astype(float)
    actual = df["actual_decision"].str.lower().values

    # V3 权重
    w3 = cfg["v3_optimal_weights"]
    pred_v3 = fuse_predict(llm_v, var_v, mkt_v, pol_v,
                           w3["w_llm"], w3["w_var"], w3["w_market"], w3["w_pol"],
                           w3["hike_thr"], w3["cut_thr"])

    # V2 权重
    w2 = cfg["v2_optimal_weights"]
    pred_v2 = fuse_predict(llm_v, var_v, mkt_v, pol_v,
                           w2["w_llm"], w2["w_var"], w2["w_market"], w2["w_pol"],
                           w2["hike_thr"], w2["cut_thr"])

    return actual, pred_v2, pred_v3


def load_final_test() -> tuple:
    """从 v3_final_test_predictions.csv 加载最终测试集。"""
    fp = _ROOT / "v3" / "v3_final_test_predictions.csv"
    df = pd.read_csv(fp)
    actual  = df["实际决策"].str.lower().values
    pred_v2 = df["V2模型预测"].str.lower().values
    pred_v3 = df["V3模型预测"].str.lower().values
    return actual, pred_v2, pred_v3


# ── 主程序 ────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  FedWatch AI ── 模型性能全面统计（Precision / Recall / Accuracy）")
    print("=" * 65)
    print("\n  注：V1 为实时单次预测系统，无历史批量回测，不纳入统计。")

    # ── 历史测试集（V3 随机 20%，41 次，跨 2000-2024）──────────────────
    print("\n" + "─" * 65)
    print("  【历史测试集 — V3 随机 20%（41 次，跨 2000-2024 各周期）】")
    print("─" * 65)
    print("\n  正在加载并计算（需拉取 FRED 数据，约 1-2 分钟）...")
    actual_h, pred_v2_h, pred_v3_h = load_historical_test("test")

    print("\n  ▶ V2 模型（时间切割权重：w_llm=0.45, w_var=0.20）")
    print_metrics("V2-历史", compute_all_metrics(actual_h, pred_v2_h))

    print("\n  ▶ V3 模型（随机划分权重：w_llm=0.35, w_var=0.35）")
    print_metrics("V3-历史", compute_all_metrics(actual_h, pred_v3_h))

    # ── 最终测试集（12 次，2025-01 至 2026-06）─────────────────────────
    print("\n" + "─" * 65)
    print("  【最终测试集（12 次，2025-01 至 2026-06，全程封存）】")
    print("─" * 65)
    actual_f, pred_v2_f, pred_v3_f = load_final_test()

    print("\n  ▶ V2 模型（w_llm=0.45, w_var=0.20, hike_thr=+0.25）")
    print_metrics("V2-最终", compute_all_metrics(actual_f, pred_v2_f))

    print("\n  ▶ V3 模型（w_llm=0.35, w_var=0.35, hike_thr=+0.20）")
    print_metrics("V3-最终", compute_all_metrics(actual_f, pred_v3_f))

    # ── 汇总对比表 ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  【汇总对比表】")
    print("=" * 65)

    rows = [
        ("V2", "历史测试集(41次)", compute_all_metrics(actual_h, pred_v2_h)),
        ("V3", "历史测试集(41次)", compute_all_metrics(actual_h, pred_v3_h)),
        ("V2", "最终测试集(12次)", compute_all_metrics(actual_f, pred_v2_f)),
        ("V3", "最终测试集(12次)", compute_all_metrics(actual_f, pred_v3_f)),
    ]

    print(f"\n  {'模型':4}  {'数据集':18}  {'Accuracy':>9}  "
          f"{'P(macro)':>9}  {'R(macro)':>9}  {'F1(macro)':>9}")
    print("  " + "─" * 65)
    for model, dataset, m in rows:
        print(f"  {model:4}  {dataset:18}  {m['accuracy']:>9.1%}  "
              f"{m['macro_p']:>9.1%}  {m['macro_r']:>9.1%}  {m['macro_f']:>9.1%}")

    print(f"\n  {'模型':4}  {'数据集':18}  "
          f"{'HOLD-P':>7} {'HOLD-R':>7}  "
          f"{'HIKE-P':>7} {'HIKE-R':>7}  "
          f"{'CUT-P':>7} {'CUT-R':>7}")
    print("  " + "─" * 75)
    for model, dataset, m in rows:
        pc = m["per_class"]
        def fmt(cls, key):
            v = pc[cls][key]
            return f"{v:.1%}" if not np.isnan(v) else "  —  "
        print(f"  {model:4}  {dataset:18}  "
              f"{fmt('hold','precision'):>7} {fmt('hold','recall'):>7}  "
              f"{fmt('hike','precision'):>7} {fmt('hike','recall'):>7}  "
              f"{fmt('cut','precision'):>7} {fmt('cut','recall'):>7}")


if __name__ == "__main__":
    main()
