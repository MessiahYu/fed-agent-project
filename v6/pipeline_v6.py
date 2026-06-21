"""
V6：并行信号 Agent + 裁决 Agent
4 个并行 CrewAI Agent（async_execution=True）同时分析不同维度，
最终由裁决 Agent 汇聚四路信号做出 FOMC 决策预测。

运行方式（项目根目录）：
  python v6/pipeline_v6.py           # 仅跑最近一次会议（沃什 2026-06-17）
  python v6/pipeline_v6.py --all     # 跑 12 次最终测试集全部会议
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import yaml
import pandas as pd
from dotenv import load_dotenv

load_dotenv(str(_ROOT / ".env"))

from crewai import Agent, Task, Crew, Process, LLM

from v2.weight_optimizer import compute_var_signals, add_numeric_signals
from v2.backtest import fetch_full_macro_series
from v5.weight_optimizer_v5 import load_feature_matrix_v5

_cfg = yaml.safe_load(open(str(_ROOT / "config.yml"), encoding="utf-8"))


# ── 工具函数 ───────────────────────────────────────────────────────────────

def _llm() -> LLM:
    return LLM(
        model=_cfg["llm"]["model"],
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=_cfg["llm"]["base_url"],
    )


def _macro_snap(df_macro: pd.DataFrame, meeting_date) -> dict:
    """从 df_macro 截取会议日前最近一个月的宏观快照（无前视偏差）"""
    if df_macro.empty:
        return {"fedfunds": "N/A", "cpi_yoy": "N/A", "unrate": "N/A"}
    cutoff = pd.Timestamp(meeting_date) - pd.offsets.MonthBegin(1)
    hist = df_macro[df_macro.index <= cutoff]
    if hist.empty:
        return {"fedfunds": "N/A", "cpi_yoy": "N/A", "unrate": "N/A"}
    row = hist.iloc[-1]
    return {
        "fedfunds": round(float(row["fedfunds"]), 2),
        "cpi_yoy":  round(float(row["cpi_yoy"]),  2),
        "unrate":   round(float(row["unrate"]),    2),
    }


def _parse_direction(text: str) -> str:
    """从裁决 Agent 输出中提取最终预测方向（HIKE / HOLD / CUT）。
    优先匹配"最终预测:"行，避免分析过程中提及对立方向时误触发。
    """
    import re
    m = re.search(r'最终预测[：:]\s*(HIKE|HOLD|CUT)', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 兜底：全文扫描（顺序 HOLD 优先，降低误判）
    text_upper = text.upper()
    for kw in ("HOLD", "CUT", "HIKE"):
        if kw in text_upper:
            return kw
    return "HOLD"


# ── 核心流水线 ─────────────────────────────────────────────────────────────

def run_v6_meeting(row: pd.Series, macro: dict, w: dict) -> dict:
    """
    对单次 FOMC 会议运行 V6 五智能体流水线。
    4 个信号 Agent 并行分析，裁决 Agent 汇聚四路结果。

    row:   final_test DataFrame 的一行（含 V5 特征列）
    macro: 宏观快照 {"fedfunds": ..., "cpi_yoy": ..., "unrate": ...}
    w:     V5 权重 dict（w_llm / w_var / w_market / w_pol / hike_thr / cut_thr）
    """
    ag_cfg = _cfg["v6_agents"]
    tpl    = _cfg["v6_task_templates"]
    llm    = _llm()

    date      = str(row["meeting_date"])[:10]
    chair     = str(row["chair"])
    llm_score = float(row["hawkish_score"])   # 已滞后一期
    var_sig   = str(row["var_signal"]).upper()
    var_num   = float(row["var_num"])
    mkt_num   = float(row["market_num"])
    pol_num   = float(row["pol_hawkish"])
    actual    = str(row["actual_decision"]).upper()

    # V5 量化融合分（作为裁决 Agent 的参考基准）
    fusion = (w["w_llm"]    * llm_score
              + w["w_var"]  * var_num
              + w["w_market"] * mkt_num
              + w["w_pol"]  * pol_num)
    if   fusion > w["hike_thr"]: v5_pred = "HIKE"
    elif fusion < w["cut_thr"]:  v5_pred = "CUT"
    else:                         v5_pred = "HOLD"

    # ── 五个 Agent（4 + 1）──────────────────────────────────────────
    a_sem = Agent(llm=llm, verbose=False, **ag_cfg["semantic"])
    a_var = Agent(llm=llm, verbose=False, **ag_cfg["var"])
    a_mkt = Agent(llm=llm, verbose=False, **ag_cfg["market"])
    a_pol = Agent(llm=llm, verbose=False, **ag_cfg["political"])
    a_arb = Agent(llm=llm, verbose=False, **ag_cfg["arbiter"])

    # ── 四个并行 Task（async_execution=True）────────────────────────
    t_sem = Task(
        description=tpl["semantic"].format(
            date=date, chair=chair, llm_score=f"{llm_score:+.2f}",
        ),
        expected_output=tpl["expected_signal"],
        agent=a_sem,
        async_execution=True,
    )
    t_var = Task(
        description=tpl["var"].format(
            date=date,
            var_signal=var_sig,
            var_num=f"{var_num:+.3f}",
            fed_funds=macro["fedfunds"],
            cpi=macro["cpi_yoy"],
            unrate=macro["unrate"],
        ),
        expected_output=tpl["expected_signal"],
        agent=a_var,
        async_execution=True,
    )
    t_mkt = Task(
        description=tpl["market"].format(
            date=date, mkt_num=f"{mkt_num:+.3f}",
        ),
        expected_output=tpl["expected_signal"],
        agent=a_mkt,
        async_execution=True,
    )
    t_pol = Task(
        description=tpl["political"].format(
            date=date, pol_num=f"{pol_num:+.3f}",
        ),
        expected_output=tpl["expected_signal"],
        agent=a_pol,
        async_execution=True,
    )

    # ── 裁决 Task（context 等待四路并行完成）──────────────────────
    t_arb = Task(
        description=tpl["arbiter"].format(
            date=date,
            chair=chair,
            w_llm=w["w_llm"],
            w_var=w["w_var"],
            w_mkt=w["w_market"],
            w_pol=w["w_pol"],
            fusion=f"{fusion:+.3f}",
            hike_thr=w["hike_thr"],
            cut_thr=w["cut_thr"],
            v5_pred=v5_pred,
        ),
        expected_output=tpl["expected_arbiter"],
        agent=a_arb,
        context=[t_sem, t_var, t_mkt, t_pol],
    )

    crew = Crew(
        agents=[a_sem, a_var, a_mkt, a_pol, a_arb],
        tasks=[t_sem, t_var, t_mkt, t_pol, t_arb],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    arbiter_raw = str(result.raw) if hasattr(result, "raw") else str(result)

    return {
        "date":           date,
        "chair":          chair,
        "actual":         actual,
        "v5_pred":        v5_pred,
        "fusion":         fusion,
        "llm_score":      llm_score,
        "var_sig":        var_sig,
        "var_num":        var_num,
        "mkt_num":        mkt_num,
        "pol_num":        pol_num,
        "arbiter_output": arbiter_raw,
    }


# ── 报告输出 ───────────────────────────────────────────────────────────────

def _print_meeting_report(res: dict, idx: int, total: int):
    W = 65
    v6_pred = _parse_direction(res["arbiter_output"])
    v6_ok   = (v6_pred == res["actual"])
    v5_ok   = (res["v5_pred"] == res["actual"])

    print(f"\n{'─'*W}")
    print(f"  [{idx}/{total}]  {res['date']}   主席：{res['chair']}")
    print(f"{'─'*W}")
    print(f"  输入信号（四路量化特征）：")
    print(f"    [A] 语义信号（LLM 鹰鸽分，滞后一期）：{res['llm_score']:+.2f}")
    print(f"    [B] 计量信号（VAR）：{res['var_sig']}  ({res['var_num']:+.3f})")
    print(f"    [C] 市场信号（收益率曲线百分位）：{res['mkt_num']:+.3f}")
    print(f"    [D] 政治信号（EPU/VIX 百分位）：{res['pol_num']:+.3f}")
    print(f"  V5 量化融合分：{res['fusion']:+.3f}  →  V5 参考预测：{res['v5_pred']}")
    print(f"\n  裁决 Agent 分析：")
    for line in res["arbiter_output"].strip().split("\n"):
        print(f"    {line}")
    print(f"\n  对比：")
    print(f"    V6 裁决 = {v6_pred}   V5 量化 = {res['v5_pred']}   实际 = {res['actual']}")
    print(f"    V6 {'✓ 正确' if v6_ok else '✗ 错误'}   V5 {'✓ 正确' if v5_ok else '✗ 错误'}")


# ── 主函数 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V6 并行信号Agent + 裁决Agent")
    parser.add_argument("--all", action="store_true",
                        help="跑全部 12 次最终测试集（默认仅跑最近一次）")
    args = parser.parse_args()

    print("\n" + "★" * 65)
    print("  V6 — 并行信号 Agent + 裁决 Agent")
    print("  架构：4 并行信号 Agent → 1 裁决 Agent（CrewAI）")
    print("★" * 65)

    print("\n正在加载 V5 特征矩阵...")
    df_all = load_feature_matrix_v5()
    df_ft  = df_all[df_all["split"] == "final_test"].copy().reset_index(drop=True)

    print("正在拉取宏观时间序列（FRED）...")
    df_macro = fetch_full_macro_series()

    print("计算 VAR 信号...")
    df_ft = add_numeric_signals(df_ft, compute_var_signals(df_ft, df_macro))

    w = _cfg["v5_optimal_weights"]

    if args.all:
        rows_to_run = list(df_ft.iterrows())
        print(f"\n将对全部 {len(rows_to_run)} 次会议运行 V6（每次 5 个 LLM 调用）")
        print("预计耗时约 5-10 分钟，请耐心等待...\n")
    else:
        last = df_ft.iloc[-1]
        rows_to_run = [(0, last)]
        print(f"\n仅运行最近一次会议：{str(last['meeting_date'])[:10]}（{last['chair']}）")
        print("使用 --all 可跑全部 12 次\n")

    total = len(rows_to_run)
    results = []

    for idx, (_, row) in enumerate(rows_to_run, 1):
        date_str = str(row["meeting_date"])[:10]
        print(f"  [{idx}/{total}] 正在分析 {date_str}（{row['chair']}）...")
        macro = _macro_snap(df_macro, pd.Timestamp(row["meeting_date"]))
        res   = run_v6_meeting(row, macro, w)
        results.append(res)
        _print_meeting_report(res, idx, total)

    # 汇总统计（--all 模式）
    if total > 1:
        print(f"\n{'★'*65}")
        print("  V6 最终测试集汇总")
        print(f"{'★'*65}")
        v6_correct = sum(1 for r in results
                         if _parse_direction(r["arbiter_output"]) == r["actual"])
        v5_correct = sum(1 for r in results if r["v5_pred"] == r["actual"])
        baseline   = sum(1 for r in results if r["actual"] == "HOLD")
        print(f"  基线（永远 HOLD）：{baseline}/{total} = {baseline/total:.1%}")
        print(f"  V5 量化准确率：   {v5_correct}/{total} = {v5_correct/total:.1%}")
        print(f"  V6 裁决准确率：   {v6_correct}/{total} = {v6_correct/total:.1%}")
        print(f"  V6 vs V5：        {(v6_correct - v5_correct):+d} 次")

        print(f"\n  {'日期':12}  {'主席':6}  {'V5':5}  {'V6':5}  {'实际':5}  V5  V6")
        print(f"  {'─'*55}")
        for r in results:
            v6_p = _parse_direction(r["arbiter_output"])
            print(f"  {r['date']:12}  {r['chair']:6}  {r['v5_pred']:5}  "
                  f"{v6_p:5}  {r['actual']:5}  "
                  f"{'✓' if r['v5_pred']==r['actual'] else '✗'}   "
                  f"{'✓' if v6_p==r['actual'] else '✗'}")
        print(f"  {'─'*55}")


if __name__ == "__main__":
    main()
