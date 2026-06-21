"""
v5/final_test_eval_v5.py
用 V5 权重对 2025-01 至今 12 次 FOMC 会议进行最终测试集评估。
V5 = LLM 滞后一期（V4）+ 分层随机 80/20 划分（V3）

运行方式：
  E:\\Anaconda\\envs\\fed-agent\\python.exe v5/final_test_eval_v5.py
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
from v5.weight_optimizer_v5 import load_feature_matrix_v5

_cfg = yaml.safe_load(open(str(_ROOT / "config.yml"), encoding="utf-8"))


def fuse(llm_v, var_v, mkt_v, pol_v, w):
    score = (w["w_llm"] * llm_v + w["w_var"] * var_v
             + w["w_market"] * mkt_v + w["w_pol"] * pol_v)
    if   score > w["hike_thr"]: return score, "HIKE"
    elif score < w["cut_thr"]:  return score, "CUT"
    else:                        return score, "HOLD"


def main():
    print("\n正在加载 V5 特征矩阵...")
    df_all = load_feature_matrix_v5()
    df_ft  = df_all[df_all["split"] == "final_test"].copy().reset_index(drop=True)

    print("正在拉取宏观数据...")
    df_macro = fetch_full_macro_series()
    print(f"计算 VAR 信号（{len(df_ft)} 次）...")
    df_ft = add_numeric_signals(df_ft, compute_var_signals(df_ft, df_macro))

    W5 = _cfg["v5_optimal_weights"]
    W4 = _cfg["v4_optimal_weights"]
    W3 = _cfg["v3_optimal_weights"]  # 对比用（V3 权重也作用于滞后 LLM，仅为权重对比）

    rows = []
    for _, r in df_ft.iterrows():
        llm_v  = float(r["hawkish_score"])   # 已是上次声明的分数（V5/V4 均滞后）
        var_v  = float(r["var_num"])
        mkt_v  = float(r["market_num"])
        pol_v  = float(r["pol_hawkish"])
        actual = r["actual_decision"].upper()

        v5_score, v5_pred = fuse(llm_v, var_v, mkt_v, pol_v, W5)
        v4_score, v4_pred = fuse(llm_v, var_v, mkt_v, pol_v, W4)

        rows.append(dict(
            date=r["meeting_date"], chair=r["chair"],
            llm=llm_v, var=r["var_signal"].upper(),
            v5_score=v5_score, v5_pred=v5_pred,
            v4_pred=v4_pred, actual=actual,
            v5_ok=(v5_pred == actual), v4_ok=(v4_pred == actual),
        ))

    df = pd.DataFrame(rows)
    n  = len(df)
    baseline = (df["actual"] == "HOLD").mean()
    v5_acc   = df["v5_ok"].mean()
    v4_acc   = df["v4_ok"].mean()

    print("\n" + "★" * 65)
    print("  V5 最终测试集评估（2025-01 至 2026-06，12 次会议）")
    print("★" * 65)
    print(f"\n  V5 权重：w_llm={W5['w_llm']} w_var={W5['w_var']} "
          f"w_market={W5['w_market']} w_pol={W5['w_pol']}")
    print(f"  阈值：hike>{W5['hike_thr']:+.2f}  cut<{W5['cut_thr']:+.2f}\n")

    print(f"  {'日期':12}  {'主席':9}  {'LLM(上次)':>10}  {'VAR':5}  "
          f"{'V5融合分':>8}  {'V5预测':6}  {'V4预测':6}  {'实际':5}  V5  V4")
    print("  " + "─" * 87)
    for _, r in df.iterrows():
        print(f"  {r['date']:12}  {r['chair']:9}  {r['llm']:+10.2f}  "
              f"{r['var']:5}  {r['v5_score']:+8.3f}  {r['v5_pred']:6}  "
              f"{r['v4_pred']:6}  {r['actual']:5}  "
              f"{'✓' if r['v5_ok'] else '✗'}   {'✓' if r['v4_ok'] else '✗'}")
    print("  " + "─" * 87)

    print(f"\n  基线（永远 HOLD）：{baseline:.1%}")
    print(f"  V4 准确率：       {v4_acc:.1%}  ({df['v4_ok'].sum()}/{n})")
    print(f"  V5 准确率：       {v5_acc:.1%}  ({df['v5_ok'].sum()}/{n})")
    print(f"  V5 超越 V4：      {v5_acc - v4_acc:+.1%}")

    # 政策转变识别
    cuts = df[df["actual"] == "CUT"]
    hikes = df[df["actual"] == "HIKE"]
    print(f"\n  【政策转变识别】")
    print(f"  CUT 事件（{len(cuts)} 次）：")
    for _, r in cuts.iterrows():
        print(f"    {r['date']}  V5={r['v5_pred']}({'✓' if r['v5_ok'] else '✗'})  "
              f"V4={r['v4_pred']}({'✓' if r['v4_ok'] else '✗'})  "
              f"融合分={r['v5_score']:+.3f}  阈值={W5['cut_thr']:+.2f}  "
              f"差距={r['v5_score'] - W5['cut_thr']:+.3f}")
    print(f"  V4 CUT 召回率：{df[df['actual']=='CUT']['v4_ok'].mean():.1%}  "
          f"({df[df['actual']=='CUT']['v4_ok'].sum()}/{len(cuts)})")
    print(f"  V5 CUT 召回率：{df[df['actual']=='CUT']['v5_ok'].mean():.1%}  "
          f"({df[df['actual']=='CUT']['v5_ok'].sum()}/{len(cuts)})")

    # 预测差异
    changed = df[df["v5_pred"] != df["v4_pred"]]
    if len(changed):
        print(f"\n  【V5 vs V4 预测差异（{len(changed)} 次）】")
        for _, r in changed.iterrows():
            print(f"    {r['date']}  V4={r['v4_pred']}({'✓' if r['v4_ok'] else '✗'}) "
                  f"→ V5={r['v5_pred']}({'✓' if r['v5_ok'] else '✗'})  实际={r['actual']}")
    else:
        print(f"\n  V5 与 V4 预测完全一致，权重差异不影响最终测试集结论。")

    # 汇总对比
    print(f"\n  {'─'*50}")
    print(f"  【四版本最终测试集汇总】")
    print(f"  {'版本':4}  {'准确率':>8}  {'超基线':>8}  {'CUT召回':>8}  {'方法特点'}")
    print(f"  {'─'*60}")
    v3_ft = 9/12; v3_cut = 2/3
    v4_ft = v4_acc; v4_cut = df[df["actual"]=="CUT"]["v4_ok"].mean()
    rows_sum = [
        ("V2", 9/12, 0.0,           1/3,    "时间切割，同期 LLM"),
        ("V3", v3_ft, v3_ft-0.75,   v3_cut, "随机划分，同期 LLM（修复时间固定效应）"),
        ("V4", v4_ft, v4_ft-0.75,   v4_cut, "时间切割，滞后 LLM（修复同期泄露）"),
        ("V5", v5_acc, v5_acc-0.75, df[df["actual"]=="CUT"]["v5_ok"].mean(),
         "随机划分，滞后 LLM（两者同时修复）"),
    ]
    for ver, acc, delta, cut_r, note in rows_sum:
        print(f"  {ver:4}  {acc:>8.1%}  {delta:>+8.1%}  {cut_r:>8.1%}  {note}")

    # 保存 CSV
    out = _ROOT / "v5" / "v5_final_test_predictions.csv"
    df_out = df.rename(columns={
        "date":"会议日期","chair":"主席","llm":"LLM鹰鸽分(上次声明)",
        "var":"VAR信号","v5_score":"V5融合分","v5_pred":"V5预测",
        "v4_pred":"V4预测","actual":"实际决策"
    })
    df_out["LLM鹰鸽分(上次声明)"] = df_out["LLM鹰鸽分(上次声明)"].apply(lambda x: f"{x:+.2f}")
    df_out["V5融合分"] = df_out["V5融合分"].apply(lambda x: f"{x:+.3f}")
    df_out["V5结果"] = df["v5_ok"].map({True:"✓", False:"✗"}).values
    df_out["V4结果"] = df["v4_ok"].map({True:"✓", False:"✗"}).values
    keep = ["会议日期","主席","LLM鹰鸽分(上次声明)","VAR信号","V5融合分",
            "V5预测","V4预测","实际决策","V5结果","V4结果"]
    df_out[keep].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  预测明细已保存：{out.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
