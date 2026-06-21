"""
v3/final_test_eval_v3.py
============================================================
用 V3 最优权重对 2025-至今 12 次 FOMC 会议进行评估。

V3 权重（分层随机 80/20，seed=42）：
  w_llm=0.35  w_var=0.35  w_market=0.10  w_pol=0.20
  hike_thr=+0.20  cut_thr=-0.35

与 V2 权重的对比：
  V2: w_llm=0.45  w_var=0.20  w_market=0.15  w_pol=0.20
      hike_thr=+0.25  cut_thr=-0.35

运行方式：
  E:\\Anaconda\\envs\\fed-agent\\python.exe v3/final_test_eval_v3.py
"""

import sys
import warnings
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
)
from v2.backtest import fetch_full_macro_series

# ── 读取 V3 最优权重 ──────────────────────────────────────────────────────
_cfg = yaml.safe_load(open(str(_ROOT / "config.yml"), encoding="utf-8"))
W = _cfg["v3_optimal_weights"]

W_LLM    = W["w_llm"]
W_VAR    = W["w_var"]
W_MARKET = W["w_market"]
W_POL    = W["w_pol"]
HIKE_THR = W["hike_thr"]
CUT_THR  = W["cut_thr"]

# V2 权重（用于对比）
W2 = _cfg["v2_optimal_weights"]


def sig_label(s: str) -> str:
    return {"hike": "HIKE", "hold": "HOLD", "cut": "CUT"}.get(str(s).lower(), str(s))


def fuse(llm_v, var_v, market_v, pol_v, w_llm, w_var, w_market, w_pol, hike_thr, cut_thr):
    fusion = w_llm * llm_v + w_var * var_v + w_market * market_v + w_pol * pol_v
    if fusion > hike_thr:
        return fusion, "HIKE"
    elif fusion < cut_thr:
        return fusion, "CUT"
    else:
        return fusion, "HOLD"


def evaluate_v3_final_test():
    print("正在加载特征矩阵...")
    df_all = load_feature_matrix()
    df_ft  = df_all[df_all["split"] == "final_test"].copy().reset_index(drop=True)

    if df_ft.empty:
        print("未找到 final_test 数据，请先运行 v2/final_test_eval.py 确保数据库已填充。")
        return

    print(f"拉取宏观时序数据...")
    df_macro = fetch_full_macro_series()

    print(f"计算 VAR 走一步预测信号（{len(df_ft)} 次）...")
    var_signals = compute_var_signals(df_ft, df_macro)
    df_ft = add_numeric_signals(df_ft, var_signals)

    # ── 逐行计算 V3 和 V2 融合分 ─────────────────────────────────────────
    rows = []
    for _, r in df_ft.iterrows():
        llm_v    = float(r["hawkish_score"])
        var_v    = float(r["var_num"])
        market_v = float(r["market_num"])
        pol_v    = float(r["pol_hawkish"])

        # V3 预测
        v3_score, v3_pred = fuse(
            llm_v, var_v, market_v, pol_v,
            W_LLM, W_VAR, W_MARKET, W_POL, HIKE_THR, CUT_THR
        )

        # V2 预测（对比用）
        v2_score, v2_pred = fuse(
            llm_v, var_v, market_v, pol_v,
            W2["w_llm"], W2["w_var"], W2["w_market"], W2["w_pol"],
            W2["hike_thr"], W2["cut_thr"]
        )

        actual = r["actual_decision"].upper()
        rows.append({
            "meeting_date": r["meeting_date"],
            "chair":        r["chair"],
            "llm_score":    llm_v,
            "var_signal":   sig_label(r["var_signal"]),
            "market_signal": sig_label(r["market_signal"]),
            "v3_fusion":    v3_score,
            "v3_pred":      v3_pred,
            "v2_pred":      v2_pred,
            "actual":       actual,
            "v3_correct":   v3_pred == actual,
            "v2_correct":   v2_pred == actual,
        })

    df_res = pd.DataFrame(rows)

    # ── 准确率汇总 ───────────────────────────────────────────────────────
    n = len(df_res)
    baseline_acc = (df_res["actual"] == "HOLD").mean()
    v3_acc = df_res["v3_correct"].mean()
    v2_acc = df_res["v2_correct"].mean()

    # ── 打印结果表 ───────────────────────────────────────────────────────
    print()
    print("★" * 65)
    print("  V3 最终测试集评估（2025-01 至 2026-06，共 12 次 FOMC 会议）")
    print("★" * 65)

    print(f"""
  V3 权重：w_llm={W_LLM}  w_var={W_VAR}  w_market={W_MARKET}  w_pol={W_POL}
           hike_thr={HIKE_THR:+.2f}  cut_thr={CUT_THR:+.2f}
""")

    # 表头
    print(f"  {'日期':12}  {'主席':9}  {'LLM分':>7}  {'VAR':5}  {'市场':5}  "
          f"{'V3融合分':>8}  {'V3预测':6}  {'V2预测':6}  {'实际':6}  {'V3':3}  {'V2':3}")
    print("  " + "─" * 89)

    for _, r in df_res.iterrows():
        v3_mark = "✓" if r["v3_correct"] else "✗"
        v2_mark = "✓" if r["v2_correct"] else "✗"
        print(
            f"  {r['meeting_date']:12}  {r['chair']:9}  "
            f"{r['llm_score']:+7.2f}  {r['var_signal']:5}  {r['market_signal']:5}  "
            f"{r['v3_fusion']:+8.3f}  {r['v3_pred']:6}  {r['v2_pred']:6}  "
            f"{r['actual']:6}  {v3_mark:3}  {v2_mark:3}"
        )

    print("  " + "─" * 89)
    print(f"""
  【准确率对比】
  基线（永远预测 HOLD）：{baseline_acc:.1%}
  V2 模型：             {v2_acc:.1%}   ({df_res['v2_correct'].sum()}/{n} 次正确)
  V3 模型：             {v3_acc:.1%}   ({df_res['v3_correct'].sum()}/{n} 次正确)
  V3 超越 V2：          {'+' if v3_acc >= v2_acc else ''}{(v3_acc - v2_acc)*100:+.1f}pp
""")

    # 错误明细
    wrong_v3 = df_res[~df_res["v3_correct"]]
    if len(wrong_v3) > 0:
        print("  【V3 预测错误明细】")
        for _, r in wrong_v3.iterrows():
            print(f"    {r['meeting_date']}  预测={r['v3_pred']}  实际={r['actual']}  "
                  f"融合分={r['v3_fusion']:+.3f}")

    # V3 相比 V2 改变的预测
    changed = df_res[df_res["v3_pred"] != df_res["v2_pred"]]
    if len(changed) > 0:
        print(f"\n  【V3 vs V2 预测发生变化的会议（共 {len(changed)} 次）】")
        for _, r in changed.iterrows():
            v2m = "✓" if r["v2_correct"] else "✗"
            v3m = "✓" if r["v3_correct"] else "✗"
            print(f"    {r['meeting_date']}  V2={r['v2_pred']}({v2m}) → V3={r['v3_pred']}({v3m})  "
                  f"实际={r['actual']}  融合分V3={r['v3_fusion']:+.3f}")
    else:
        print("\n  V3 与 V2 在最终测试集上的所有预测结果完全一致。")

    # ── 保存 CSV ─────────────────────────────────────────────────────────
    out_cols = {
        "meeting_date": "会议日期",
        "chair":        "主持主席",
        "llm_score":    "LLM鹰鸽分",
        "var_signal":   "VAR信号",
        "market_signal":"市场信号",
        "v3_fusion":    "V3融合得分",
        "v3_pred":      "V3模型预测",
        "v2_pred":      "V2模型预测",
        "actual":       "实际决策",
    }
    df_out = df_res[list(out_cols.keys())].copy()
    df_out.rename(columns=out_cols, inplace=True)
    df_out["LLM鹰鸽分"] = df_out["LLM鹰鸽分"].apply(lambda x: f"{x:+.2f}")
    df_out["V3融合得分"] = df_out["V3融合得分"].apply(lambda x: f"{x:+.3f}")
    df_out["V3预测结果"] = df_res["v3_correct"].map({True: "✓", False: "✗"}).values
    df_out["V2预测结果"] = df_res["v2_correct"].map({True: "✓", False: "✗"}).values

    out_path = _ROOT / "v3" / "v3_final_test_predictions.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  预测明细已保存至: {out_path.relative_to(_ROOT)}")

    return df_res


if __name__ == "__main__":
    evaluate_v3_final_test()
