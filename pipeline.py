"""
FedWatch AI — 完整五阶段流水线
===============================
一键运行完成：
  Agent 1  自动抓取美联储声明 + FRED 宏观数据 → MeetingPackage
  Agent 2  语义分析声明文本 → SemanticFeature（鹰鸽评分等）
  Agent 3  VAR 计量 + LLM 链式草稿 → FOMCPrediction
  Agent 4  五位主席行为蒸馏 → DistillResult
  Agent 5  市场预期+政治压力+国际联动 → InfluencerResult
  全部结果写入 SQLite（完整数据血统链）

运行方式：
  E:\\Anaconda\\envs\\fed-agent\\python.exe pipeline.py
"""

from agent1_data_perception import main as run_agent1, save_to_db
from agent2_semantic        import run_semantic_analysis, save_semantic_to_db
from agent3_decision        import run_decision_agent
from agent4_chair_distill   import run_chair_distill, load_chair_stats
from agent5_influencers     import run_influencer_agents


def run_pipeline():
    print("\n" + "★" * 64)
    print("  FedWatch AI — 完整五智能体流水线启动")
    print("★" * 64)

    # ── 阶段 1：Agent 1 数据感知 ───────────────────────────────────
    print("\n【阶段 1/5】数据感知智能体\n")
    package   = run_agent1()
    pkg_db_id = save_to_db(package)
    print(f"\n✓ Agent 1 完成  →  meeting_packages.id={pkg_db_id}")

    macro_context = {
        "cpi_yoy":           package.macro.cpi_yoy,
        "unemployment_rate": package.macro.unemployment_rate,
        "fed_funds_rate":    package.macro.fed_funds_rate,
        "treasury_10y":      package.macro.treasury_10y,
    }

    # ── 阶段 2：Agent 2 语义提取 ───────────────────────────────────
    print("\n【阶段 2/5】语义提取智能体\n")
    feature   = run_semantic_analysis(
        statement_text = package.statement_text,
        meeting_date   = package.meeting_date,
        macro_context  = macro_context,
    )
    sem_db_id = save_semantic_to_db(pkg_db_id, feature)
    print(f"\n✓ Agent 2 完成  →  semantic_results.id={sem_db_id}")

    semantic_dict = {
        "hawkish_score":     feature.hawkish_score,
        "inflation_concern": feature.inflation_concern,
        "growth_concern":    feature.growth_concern,
        "warsh_signal":      feature.warsh_signal,
    }

    # ── 阶段 3：Agent 3 决策推演 ───────────────────────────────────
    print("\n【阶段 3/5】决策推演智能体（VAR + LLM CoD）\n")
    prediction = run_decision_agent(
        semantic_id      = sem_db_id,
        semantic_feature = semantic_dict,
        macro_context    = macro_context,
    )
    print(f"\n✓ Agent 3 完成  →  fomc_predictions.id 已写入")

    # ── 阶段 4：Agent 4 主席行为蒸馏 ──────────────────────────────
    print("\n【阶段 4/5】主席行为蒸馏智能体（五位主席 ICL）\n")
    distill = run_chair_distill(macro_context, package.meeting_date)
    print(f"\n✓ Agent 4 完成  →  沃什预测: {distill.warsh_prediction.upper()}"
          f"  置信度: {distill.warsh_confidence:.0%}")

    # ── 阶段 5：Agent 5 影响者联合 ────────────────────────────────
    print("\n【阶段 5/5】影响者联合 Agent 群（市场+政治+国际）\n")
    influencer = run_influencer_agents(macro_context, package.meeting_date)
    print(f"\n✓ Agent 5 完成  →  综合信号: {influencer.combined_signal.upper()}"
          f"  置信度: {influencer.combined_confidence:.0%}")

    # ── 最终综合报告 ───────────────────────────────────────────────
    _print_final_report(package, feature, prediction, distill, influencer)

    return package, feature, prediction, distill, influencer


def _print_final_report(package, feature, prediction, distill, influencer):
    W = 64

    # 五路信号汇总（用于最终共识计算）
    signals = [
        ("Agent3-VAR",  prediction.var_signal,           0.35),
        ("Agent3-LLM",  prediction.llm_signal,           0.35),
        ("Agent4-蒸馏", distill.warsh_prediction,        0.15),
        ("Agent5-市场", influencer.curve_signal,         0.08),
        ("Agent5-综合", influencer.combined_signal,      0.07),
    ]
    vote = {"hold": 0.0, "cut": 0.0, "hike": 0.0}
    for _, sig, w in signals:
        if sig in vote:
            vote[sig] += w
    consensus = max(vote, key=vote.get)
    consensus_conf = round(vote[consensus], 2)

    # 交易信号
    if "cut" in consensus:
        trade = "做多国债 / 做空美元 / 超配率敏感成长股"
    elif "hike" in consensus:
        trade = "做空国债 / 做多美元 / 规避久期风险"
    else:
        trade = "维持现有仓位 / 等待更多数据 / 关注下次会议前指引"

    print("\n" + "★" * W)
    print("  FedWatch AI — 最终综合报告")
    print("★" * W)
    print(f"""
┌─ 数据基础（Agent 1）{'─'*38}
│  会议日期   {package.meeting_date}
│  数据截断   {package.data_cutoff_date}（防前视偏差）
│  声明字数   {len(package.statement_text)} 字符（实时抓取）
│
│  CPI 同比      {package.macro.cpi_yoy}%   （目标 2%，偏高）
│  失业率        {package.macro.unemployment_rate}%
│  联邦基金利率  {package.macro.fed_funds_rate}%
│  10Y 国债      {package.macro.treasury_10y}%

┌─ 语义特征（Agent 2）{'─'*38}
│  鹰鸽评分   {feature.hawkish_score:+.2f}   通胀担忧 {feature.inflation_concern:.2f}   增长担忧 {feature.growth_concern:.2f}
│  沃什信号   {feature.warsh_signal}
│  关键证据：""")
    for i, ev in enumerate(feature.key_evidence, 1):
        print(f"│    {i}. {ev}")

    print(f"""│
┌─ 量化+LLM 决策（Agent 3）{'─'*32}
│  VAR 信号   {prediction.var_signal.upper():6}  预测利率 {prediction.var_forecast_rate}%
│  LLM 信号   {prediction.llm_signal.upper():6}  5步CoD推理已完成
│  Agent3预测 ► {prediction.predicted_action.upper()} ◄  置信度 {prediction.confidence:.0%}

┌─ 主席行为蒸馏（Agent 4）{'─'*34}
│  四位主席投票：
│    Greenspan {distill.greenspan_vote.upper():6}  Bernanke {distill.bernanke_vote.upper():6}
│    Yellen    {distill.yellen_vote.upper():6}  Powell   {distill.powell_vote.upper():6}
│  最相似时期：{distill.most_similar_period}
│  沃什预测   ► {distill.warsh_prediction.upper()} ◄  置信度 {distill.warsh_confidence:.0%}

┌─ 影响者信号（Agent 5）{'─'*36}
│  收益率曲线  10Y-2Y = {influencer.yield_spread or '?'}%  → {influencer.curve_signal.upper()}
│  政治压力   {influencer.political_direction}  强度 {influencer.political_intensity:.0%}
│             独立性风险 {influencer.fed_independence_risk.upper()}
│  ECB 方向   {influencer.ecb_direction}  美元趋势 {influencer.dollar_trend}
│  国际信号   {influencer.international_signal.upper()}
│  Agent5综合 ► {influencer.combined_signal.upper()} ◄  置信度 {influencer.combined_confidence:.0%}

""")

    # 五路共识表
    print(f"┌─ 五路信号共识{'─'*44}")
    for name, sig, w in signals:
        bar = "█" * int(w * 40)
        icon = {"hold": "—", "cut": "▼", "hike": "▲"}.get(sig, "?")
        print(f"│  [{icon}] {name:14}  {sig.upper():6}  权重 {w:.0%}  {bar}")

    print(f"""│
│  ════════════════════════════════════
│  ► 最终共识预测：{consensus.upper()}
│  ► 加权置信度：  {consensus_conf:.0%}
│
│  ► 交易信号：{trade}
└{'─'*62}""")

    print(f"""
  Agent 3 结论：{prediction.final_conclusion}
  Agent 4 蒸馏：{distill.warsh_reasoning}
  Agent 5 综合：{influencer.synthesis}
""")

    print("★" * W)
    print("  数据血统链（SQLite 五张表）")
    print("  meeting_packages → semantic_results → fomc_predictions")
    print("                                      → chair_distill_results")
    print("                                      → influencer_signals")
    print("★" * W)


if __name__ == "__main__":
    run_pipeline()
