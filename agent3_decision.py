import os
import re
import json
import sqlite3
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from statsmodels.tsa.api import VAR
from crewai import Agent, Task, Crew, LLM

load_dotenv()

FRED_KEY  = os.getenv("FRED_API_KEY")
from utils.db import DB_PATH
from utils.config import (FRED_BASE_URL as FRED_BASE, NEXT_FOMC as NEXT_FOMC_DATE,
                           LLM_MODEL, LLM_BASE_URL, CONTEXT_HISTORICAL)

# ── 输出数据模型 ──────────────────────────────────────────────────────────────

class FOMCPrediction(BaseModel):
    next_meeting_date: str   = Field(description="下次FOMC预计日期")
    predicted_action:  str   = Field(description="预测决策：hold / cut_25bp / cut_50bp / hike_25bp")
    confidence:        float = Field(description="综合置信度，0.0-1.0", ge=0.0, le=1.0)
    var_signal:        str   = Field(description="VAR计量模型信号：hold / cut / hike")
    var_forecast_rate: Optional[float] = Field(default=None, description="VAR预测的下期联邦基金利率(%)")
    llm_signal:        str   = Field(description="LLM推演信号：hold / cut / hike")
    cod_steps:         List[str] = Field(description="Chain-of-Draft五步推理")
    risk_factors:      List[str] = Field(description="关键风险因素")
    evidence_chain:    List[str] = Field(description="支撑预测的证据链")
    final_conclusion:  str   = Field(description="综合结论，150字以内")


# ═══════════════════════════════════════════════════════════════════════
# 子模块 A：VAR 计量基线
# 从 FRED 拉5年月度历史数据，拟合向量自回归模型，预测下期利率方向
# ═══════════════════════════════════════════════════════════════════════

def fetch_fred_history(series_id: str, units: str = "lin", years: int = 5) -> Optional[pd.Series]:
    """从 FRED 拉取多年月度历史数据，返回 pandas Series（索引为日期）"""
    if not FRED_KEY:
        return None

    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    params = {
        "series_id": series_id,
        "api_key":   FRED_KEY,
        "file_type": "json",
        "observation_start": start,
        "sort_order": "asc",
        "units":      units,
        "frequency":  "m",
    }
    try:
        resp = requests.get(FRED_BASE, params=params, timeout=15)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        dates, values = [], []
        for obs in observations:
            if obs.get("value") and obs["value"] != ".":
                dates.append(pd.to_datetime(obs["date"]))
                values.append(float(obs["value"]))
        if not values:
            return None
        return pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id)
    except Exception as e:
        print(f"  [VAR] FRED {series_id} 历史数据获取失败：{e}")
        return None


def build_var_dataframe() -> Optional[pd.DataFrame]:
    """
    构建 VAR 用的 DataFrame，包含三列：
    - fedfunds : 联邦基金利率（水平值）
    - cpi_yoy  : CPI 同比增速（%）
    - unrate   : 失业率（%）
    这三个变量之间存在动态联动关系，VAR 能捕捉它们的相互影响。
    """
    print("  [VAR] 正在从 FRED 拉取5年历史数据...")
    s_ff  = fetch_fred_history("FEDFUNDS", units="lin")
    s_cpi = fetch_fred_history("CPIAUCSL", units="pc1")
    s_ur  = fetch_fred_history("UNRATE",   units="lin")

    if s_ff is None or s_cpi is None or s_ur is None:
        print("  [VAR] 历史数据不完整，跳过 VAR 模块")
        return None

    df = pd.concat([s_ff, s_cpi, s_ur], axis=1)
    df.columns = ["fedfunds", "cpi_yoy", "unrate"]
    df = df.dropna()
    print(f"  [VAR] 数据就绪：{len(df)} 个月度观测值（{df.index[0].date()} → {df.index[-1].date()}）")
    return df


def run_var_submodule() -> dict:
    """
    拟合 VAR 模型，预测1期（约6-7周后的下次会议）利率。
    返回 {"signal": "hold"/"cut"/"hike", "forecast_rate": float, "confidence": float}
    """
    df = build_var_dataframe()
    if df is None or len(df) < 24:
        return {"signal": "hold", "forecast_rate": None, "confidence": 0.40, "note": "数据不足，使用默认信号"}

    try:
        model   = VAR(df)
        results = model.fit(maxlags=4, ic="aic")
        lag_p   = results.k_ar

        # 预测下1期（下个月）
        forecast = results.forecast(df.values[-lag_p:], steps=1)
        pred_ff  = round(float(forecast[0][0]), 3)
        curr_ff  = float(df["fedfunds"].iloc[-1])
        change   = pred_ff - curr_ff

        print(f"  [VAR] 当前联邦基金利率：{curr_ff:.2f}%  →  VAR预测：{pred_ff:.2f}%  变化：{change:+.2f}bp*100")

        # 把预测的利率变化映射到方向信号
        if change < -0.12:
            signal = "cut"
        elif change > 0.12:
            signal = "hike"
        else:
            signal = "hold"

        # 置信度：变化幅度越大，模型越确定方向
        confidence = min(0.5 + abs(change) * 2, 0.80)

        return {
            "signal":        signal,
            "forecast_rate": pred_ff,
            "confidence":    round(confidence, 2),
            "lag_order":     lag_p,
        }

    except Exception as e:
        print(f"  [VAR] 模型拟合失败：{e}")
        return {"signal": "hold", "forecast_rate": None, "confidence": 0.40, "note": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# 子模块 B：LLM 链式草稿推演
# 5步 Chain-of-Draft + 历史案例情境学习（ICL）→ 输出方向+置信度
# ═══════════════════════════════════════════════════════════════════════

# 历史案例从 config.yml 加载（在 import 区已导入 CONTEXT_HISTORICAL）
HISTORICAL_CASES = CONTEXT_HISTORICAL


def run_llm_submodule(semantic_feature: dict, macro_context: dict) -> dict:
    """
    用 DeepSeek + CrewAI 做链式草稿推演，预测下次 FOMC 决策方向。
    强制5步推理（每步≤30字），最终输出 JSON。
    """
    llm = LLM(
        model=LLM_MODEL,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=LLM_BASE_URL,
    )

    reasoning_agent = Agent(
        role="美联储决策推演专家",
        goal="用链式草稿推理，从宏观数据和政策文本中推演下次 FOMC 利率决策方向",
        backstory=(
            "你是曾供职于美联储的货币政策经济学家，对 FOMC 的决策逻辑有深刻理解。"
            "你的特长是'链式草稿'推理——先把复杂问题分解为5个简短步骤，"
            "再综合得出结论，每步推理都有证据支撑。"
            "你了解沃什的政策立场：偏鹰、重价格稳定信誉、但对降息持开放态度。"
        ),
        llm=llm,
        verbose=True,
    )

    task = Task(
        description=f"""
请预测美联储下次 FOMC 会议（{NEXT_FOMC_DATE}）的利率决策方向。

当前宏观环境：
- CPI 同比：{macro_context.get('cpi_yoy', 'N/A')}%（目标 2%）
- 失业率：{macro_context.get('unemployment_rate', 'N/A')}%
- 联邦基金利率：{macro_context.get('fed_funds_rate', 'N/A')}%（区间 3.5-3.75%）
- 10年期国债收益率：{macro_context.get('treasury_10y', 'N/A')}%

最新语义分析结果：
- 鹰鸽评分：{semantic_feature.get('hawkish_score', 'N/A')}
- 通胀担忧：{semantic_feature.get('inflation_concern', 'N/A')}
- 增长担忧：{semantic_feature.get('growth_concern', 'N/A')}
- 沃什信号：{semantic_feature.get('warsh_signal', 'N/A')}

{HISTORICAL_CASES}

请用"链式草稿"方式推理，然后输出一个合法 JSON（不要加 ``` 标记）：
{{
  "cod_step1": "<≤30字：判断当前通胀形势>",
  "cod_step2": "<≤30字：判断就业与增长形势>",
  "cod_step3": "<≤30字：分析金融条件（收益率曲线/利差）>",
  "cod_step4": "<≤30字：推断沃什的政策倾向与约束条件>",
  "cod_step5": "<≤30字：参照历史案例，找最相近的前例>",
  "llm_signal": "<hold 或 cut 或 hike>",
  "predicted_action": "<hold 或 cut_25bp 或 cut_50bp 或 hike_25bp>",
  "confidence": <0.0到1.0的浮点数>,
  "risk_factors": ["<风险1>", "<风险2>", "<风险3>"],
  "evidence_chain": ["<证据1>", "<证据2>", "<证据3>"],
  "final_conclusion": "<150字以内的综合结论，含下次会议方向与条件>"
}}
""",
        expected_output="包含链式推理步骤和最终预测的 JSON 对象，字段完整，不含 markdown 标记",
        agent=reasoning_agent,
    )

    crew   = Crew(agents=[reasoning_agent], tasks=[task], verbose=True)
    result = crew.kickoff()

    raw         = result.raw
    json_match  = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"LLM 子模块输出无法解析为 JSON:\n{raw}")

    return json.loads(json_match.group())


# ═══════════════════════════════════════════════════════════════════════
# 融合层：合并 VAR 和 LLM 的信号
# ═══════════════════════════════════════════════════════════════════════

def fuse_and_build_prediction(var_result: dict, llm_result: dict) -> FOMCPrediction:
    """
    融合两个子模块的输出：
    - 两者方向一致 → 置信度加成（+5%）
    - 两者分歧     → 采用 LLM 信号（上下文更丰富），置信度打折（×0.80）
    """
    var_signal = var_result.get("signal", "hold")
    llm_signal = llm_result.get("llm_signal", "hold")
    llm_conf   = float(llm_result.get("confidence", 0.60))
    var_conf   = float(var_result.get("confidence", 0.50))

    if var_signal == llm_signal:
        final_action  = llm_result.get("predicted_action", "hold")
        final_signal  = llm_signal
        final_conf    = round(min((var_conf + llm_conf) / 2 + 0.05, 0.95), 2)
        print(f"  [融合] VAR 与 LLM 方向一致（{var_signal}），置信度加成 → {final_conf:.0%}")
    else:
        final_action  = llm_result.get("predicted_action", "hold")
        final_signal  = llm_signal
        final_conf    = round(max(var_conf, llm_conf) * 0.80, 2)
        print(f"  [融合] VAR({var_signal}) vs LLM({llm_signal}) 出现分歧，采信 LLM，置信度打折 → {final_conf:.0%}")

    cod_steps = [
        llm_result.get("cod_step1", ""),
        llm_result.get("cod_step2", ""),
        llm_result.get("cod_step3", ""),
        llm_result.get("cod_step4", ""),
        llm_result.get("cod_step5", ""),
    ]

    return FOMCPrediction(
        next_meeting_date = NEXT_FOMC_DATE,
        predicted_action  = final_action,
        confidence        = final_conf,
        var_signal        = var_signal,
        var_forecast_rate = var_result.get("forecast_rate"),
        llm_signal        = final_signal,
        cod_steps         = cod_steps,
        risk_factors      = llm_result.get("risk_factors", []),
        evidence_chain    = llm_result.get("evidence_chain", []),
        final_conclusion  = llm_result.get("final_conclusion", ""),
    )


# ═══════════════════════════════════════════════════════════════════════
# 数据库存储
# ═══════════════════════════════════════════════════════════════════════

def save_prediction_to_db(semantic_id: int, prediction: FOMCPrediction) -> int:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fomc_predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            semantic_id       INTEGER NOT NULL,
            next_meeting_date TEXT,
            predicted_action  TEXT,
            confidence        REAL,
            var_signal        TEXT,
            var_forecast_rate REAL,
            llm_signal        TEXT,
            cod_steps         TEXT,
            risk_factors      TEXT,
            evidence_chain    TEXT,
            final_conclusion  TEXT,
            raw_json          TEXT,
            predicted_at      TEXT DEFAULT (datetime('now'))
        )
    """)
    cursor = conn.execute("""
        INSERT INTO fomc_predictions
            (semantic_id, next_meeting_date, predicted_action, confidence,
             var_signal, var_forecast_rate, llm_signal,
             cod_steps, risk_factors, evidence_chain, final_conclusion, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        semantic_id,
        prediction.next_meeting_date,
        prediction.predicted_action,
        prediction.confidence,
        prediction.var_signal,
        prediction.var_forecast_rate,
        prediction.llm_signal,
        json.dumps(prediction.cod_steps,      ensure_ascii=False),
        json.dumps(prediction.risk_factors,   ensure_ascii=False),
        json.dumps(prediction.evidence_chain, ensure_ascii=False),
        prediction.final_conclusion,
        prediction.model_dump_json(),
    ))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


# ═══════════════════════════════════════════════════════════════════════
# 对外接口（供 pipeline.py 调用）
# ═══════════════════════════════════════════════════════════════════════

def run_decision_agent(
    semantic_id:      int,
    semantic_feature: dict,
    macro_context:    dict,
) -> FOMCPrediction:
    """
    主入口：运行两个子模块并融合，返回 FOMCPrediction。
    """
    print("\n── 子模块 A：VAR 计量基线 ──")
    var_result = run_var_submodule()

    print("\n── 子模块 B：LLM 链式草稿推演 ──")
    llm_result = run_llm_submodule(semantic_feature, macro_context)

    print("\n── 融合两个子模块信号 ──")
    prediction = fuse_and_build_prediction(var_result, llm_result)

    db_id = save_prediction_to_db(semantic_id, prediction)
    print(f"  ✓ FOMCPrediction 已存库（fomc_predictions.id={db_id}）")

    return prediction


# ═══════════════════════════════════════════════════════════════════════
# 单独运行入口（从数据库取最新语义结果直接运行）
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT sr.id, sr.hawkish_score, sr.inflation_concern, sr.growth_concern,
               sr.warsh_signal, sr.conclusion,
               mp.cpi_yoy, mp.unemployment_rate, mp.fed_funds_rate, mp.treasury_10y
        FROM semantic_results sr
        JOIN meeting_packages mp ON sr.package_id = mp.id
        ORDER BY sr.id DESC LIMIT 1
    """).fetchone()
    conn.close()

    if not row:
        print("数据库中无语义分析结果，请先运行 pipeline.py")
        raise SystemExit(1)

    (sem_id, h_score, inf_concern, gr_concern, w_signal, conclusion,
     cpi, unrate, fedfunds, treasury) = row

    semantic_feature = {
        "hawkish_score":     h_score,
        "inflation_concern": inf_concern,
        "growth_concern":    gr_concern,
        "warsh_signal":      w_signal,
    }
    macro_context = {
        "cpi_yoy":           cpi,
        "unemployment_rate": unrate,
        "fed_funds_rate":    fedfunds,
        "treasury_10y":      treasury,
    }

    print("=" * 60)
    print("Agent 3：决策推演智能体")
    print(f"输入：semantic_results.id={sem_id}")
    print("=" * 60)

    prediction = run_decision_agent(sem_id, semantic_feature, macro_context)

    print("\n" + "=" * 60)
    print("FOMCPrediction — 最终预测结果")
    print("=" * 60)
    print(f"下次会议日期   : {prediction.next_meeting_date}")
    print(f"预测决策       : {prediction.predicted_action}")
    print(f"综合置信度     : {prediction.confidence:.0%}")
    print(f"VAR 信号       : {prediction.var_signal}  (预测利率 {prediction.var_forecast_rate}%)")
    print(f"LLM 信号       : {prediction.llm_signal}")
    print()
    print("Chain-of-Draft 推理链：")
    for i, step in enumerate(prediction.cod_steps, 1):
        print(f"  Step{i}: {step}")
    print()
    print("关键风险因素：")
    for r in prediction.risk_factors:
        print(f"  · {r}")
    print()
    print("证据链：")
    for e in prediction.evidence_chain:
        print(f"  · {e}")
    print()
    print(f"综合结论：\n  {prediction.final_conclusion}")
    print("=" * 60)
