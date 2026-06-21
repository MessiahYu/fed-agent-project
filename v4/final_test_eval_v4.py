"""
v4/final_test_eval_v4.py
用 V4 权重对 2025-01 至今 12 次 FOMC 会议进行最终测试集评估。
V4 改进：hawkish_score 滞后一期（上次声明预测本次决策）。

运行方式：
  E:\\Anaconda\\envs\\fed-agent\\python.exe v4/final_test_eval_v4.py
"""

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from v2.weight_optimizer import compute_var_signals, add_numeric_signals
from v2.backtest import fetch_full_macro_series
from v4.weight_optimizer_v4 import load_feature_matrix_v4

_cfg = yaml.safe_load(open(str(_ROOT / "config.yml"), encoding="utf-8"))
W  = _cfg["v4_optimal_weights"]
W2 = _cfg["v2_optimal_weights"]
W3 = _cfg["v3_optimal_weights"]


def fuse(llm_v, var_v, mkt_v, pol_v, w):
    score = w["w_llm"]*llm_v + w["w_var"]*var_v + w["w_market"]*mkt_v + w["w_pol"]*pol_v
    if   score > w["hike_thr"]: return score, "HIKE"
    elif score < w["cut_thr"]:  return score, "CUT"
    else:                        return score, "HOLD"


def main():
    print("\n正在加载 V4 特征矩阵（含 LLM 滞后一期）...")
    df_all = load_feature_matrix_v4()
    df_ft  = df_all[df_all["split"] == "final_test"].copy().reset_index(drop=True)

    print(f"拉取宏观数据...")
    df_macro = fetch_full_macro_series()
    print(f"计算 VAR 信号（{len(df_ft)} 次）...")
    df_ft = add_numeric_signals(df_ft, compute_var_signals(df_ft, df_macro))

    rows = []
    for _, r in df_ft.iterrows():
        llm_v = float(r["hawkish_score"])   # ← 上次会议声明的鹰鸽分
        var_v = float(r["var_num"])
        mkt_v = float(r["market_num"])
        pol_v = float(r["pol_hawkish"])
        actual = r["actual_decision"].upper()

        v4_score, v4_pred = fuse(llm_v, var_v, mkt_v, pol_v, W)
        v2_score, v2_pred = fuse(llm_v, var_v, mkt_v, pol_v, W2)  # 对比用

        rows.append(dict(
            date=r["meeting_date"], chair=r["chair"],
            llm=llm_v, var=r["var_signal"].upper(),
            market=r["market_signal"].upper(),
            v4_score=v4_score, v4_pred=v4_pred,
            v2_pred=v2_pred, actual=actual,
            v4_ok=(v4_pred == actual), v2_ok=(v2_pred == actual),
        ))

    df = pd.DataFrame(rows)
    n  = len(df)
    baseline = (df["actual"] == "HOLD").mean()
    v4_acc   = df["v4_ok"].mean()
    v2_acc   = df["v2_ok"].mean()

    print("\n" + "★" * 65)
    print("  V4 最终测试集评估（2025-01 至 2026-06，12 次会议）")
    print("★" * 65)
    print(f"\n  V4 权重（滞后一期）：w_llm={W['w_llm']} w_var={W['w_var']} "
          f"w_market={W['w_market']} w_pol={W['w_pol']}")
    print(f"  阈值：hike>{W['hike_thr']:+.2f}  cut<{W['cut_thr']:+.2f}\n")
    print(f"  {'日期':12}  {'主席':9}  {'LLM分(上次)':>11}  {'VAR':5}  "
          f"{'V4融合分':>8}  {'V4预测':6}  {'V2预测':6}  {'实际':5}  V4  V2")
    print("  " + "─" * 85)

    for _, r in df.iterrows():
        v4m = "✓" if r["v4_ok"] else "✗"
        v2m = "✓" if r["v2_ok"] else "✗"
        print(f"  {r['date']:12}  {r['chair']:9}  {r['llm']:+11.2f}  "
              f"{r['var']:5}  {r['v4_score']:+8.3f}  {r['v4_pred']:6}  "
              f"{r['v2_pred']:6}  {r['actual']:5}  {v4m}   {v2m}")

    print("  " + "─" * 85)
    print(f"\n  基线（永远 HOLD）：{baseline:.1%}")
    print(f"  V2 准确率：       {v2_acc:.1%}  ({df['v2_ok'].sum()}/{n})")
    print(f"  V4 准确率：       {v4_acc:.1%}  ({df['v4_ok'].sum()}/{n})")
    print(f"  V4 超越 V2：      {v4_acc - v2_acc:+.1%}")

    # 转折点识别（CUT 事件）
    cut_rows = df[df["actual"] == "CUT"]
    print(f"\n  【政策转变识别（{len(cut_rows)} 次 CUT）】")
    print(f"  V2 CUT 召回率：{df[df['actual']=='CUT']['v2_ok'].mean():.1%}  "
          f"（{df[df['actual']=='CUT']['v2_ok'].sum()}/{len(cut_rows)}）")
    print(f"  V4 CUT 召回率：{df[df['actual']=='CUT']['v4_ok'].mean():.1%}  "
          f"（{df[df['actual']=='CUT']['v4_ok'].sum()}/{len(cut_rows)}）")

    # 变化明细
    changed = df[df["v4_pred"] != df["v2_pred"]]
    if len(changed):
        print(f"\n  【V4 vs V2 预测差异（{len(changed)} 次）】")
        for _, r in changed.iterrows():
            print(f"    {r['date']}  V2={r['v2_pred']}({'✓' if r['v2_ok'] else '✗'}) "
                  f"→ V4={r['v4_pred']}({'✓' if r['v4_ok'] else '✗'})  实际={r['actual']}")

    # 保存 CSV
    out = _ROOT / "v4" / "v4_final_test_predictions.csv"
    df_out = df.rename(columns={
        "date":"会议日期","chair":"主席","llm":"LLM鹰鸽分(上次声明)",
        "var":"VAR信号","v4_score":"V4融合分","v4_pred":"V4预测",
        "v2_pred":"V2预测","actual":"实际决策"
    })
    df_out["LLM鹰鸽分(上次声明)"] = df_out["LLM鹰鸽分(上次声明)"].apply(lambda x: f"{x:+.2f}")
    df_out["V4融合分"] = df_out["V4融合分"].apply(lambda x: f"{x:+.3f}")
    df_out["V4结果"] = df["v4_ok"].map({True:"✓", False:"✗"}).values
    df_out["V2结果"] = df["v2_ok"].map({True:"✓", False:"✗"}).values
    keep = ["会议日期","主席","LLM鹰鸽分(上次声明)","VAR信号","V4融合分",
            "V4预测","V2预测","实际决策","V4结果","V2结果"]
    df_out[keep].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  预测明细已保存：{out.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
