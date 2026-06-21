import os
import re
import json
import sqlite3
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, LLM

load_dotenv()

from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from utils.db import DB_PATH
from utils.config import LLM_MODEL, LLM_BASE_URL


# ── 输出数据模型 ──────────────────────────────────────────────────────────────

class SemanticFeature(BaseModel):
    """Agent 2 的结构化输出——把文字转成可计算的特征向量"""
    hawkish_score: float = Field(
        description="鹰鸽评分，-1.0（极鸽）到 +1.0（极鹰），0 = 中性",
        ge=-1.0, le=1.0,
    )
    inflation_concern: float = Field(
        description="通胀担忧程度，0.0 到 1.0",
        ge=0.0, le=1.0,
    )
    growth_concern: float = Field(
        description="经济增长担忧程度，0.0 到 1.0",
        ge=0.0, le=1.0,
    )
    key_evidence: List[str] = Field(
        description="3到5条关键证据，直接引用声明原文"
    )
    warsh_signal: str = Field(
        description="对沃什个人政策立场的判断：hawkish / neutral / dovish"
    )
    conclusion: str = Field(
        description="政策走向结论，100字以内"
    )


# ── 核心分析函数 ──────────────────────────────────────────────────────────────

def run_semantic_analysis(
    statement_text: str,
    meeting_date: str,
    macro_context: Optional[dict] = None,
) -> SemanticFeature:
    """
    接收声明文本 + 宏观背景，用 CrewAI + DeepSeek 分析政策情绪，
    返回结构化的 SemanticFeature。

    Parameters
    ----------
    statement_text : 美联储声明正文（来自 Agent 1）
    meeting_date   : 会议日期字符串，如 "2026-06-17"
    macro_context  : 宏观数据字典，如 {"cpi_yoy": 4.17, "unemployment": 4.3, ...}
    """
    llm = LLM(
        model=LLM_MODEL,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=LLM_BASE_URL,
    )

    analyst = Agent(
        role="美联储政策语义分析师",
        goal="从 FOMC 声明中提取可量化的政策情绪特征向量，辅以宏观数据背景",
        backstory=(
            "你是顶尖宏观对冲基金的美联储文本分析师，有15年解读FOMC文件的经验。"
            "你擅长识别声明措辞的细微变化——例如 'somewhat elevated' 比 'elevated' 更鸽，"
            "'carefully assess' 比 'patiently monitor' 更鹰。"
            "你同时结合宏观经济数据背景来校准评分，避免只看文字而忽视数据。"
            "新主席沃什偏鹰，强调通胀信誉，但曾表示有降息空间，这一矛盾是分析重点。"
        ),
        llm=llm,
        verbose=True,
    )

    macro_text = ""
    if macro_context:
        macro_text = f"""
宏观数据背景（截至会议日前2天）：
- CPI 同比：{macro_context.get('cpi_yoy', 'N/A')}%（目标：2%）
- 失业率：{macro_context.get('unemployment_rate', 'N/A')}%
- 联邦基金利率：{macro_context.get('fed_funds_rate', 'N/A')}%
- 10年期国债收益率：{macro_context.get('treasury_10y', 'N/A')}%
"""

    task = Task(
        description=f"""请分析以下 {meeting_date} FOMC 声明，结合宏观数据背景，提取政策情绪特征。
{macro_text}
声明原文：
{statement_text}

你必须只输出一个合法的 JSON 对象（不要加 ``` 标记），格式严格如下：
{{
  "hawkish_score": <-1.0到1.0的浮点数，正=鹰派，负=鸽派>,
  "inflation_concern": <0.0到1.0，通胀担忧程度>,
  "growth_concern": <0.0到1.0，增长担忧程度>,
  "key_evidence": ["<证据1>", "<证据2>", "<证据3>"],
  "warsh_signal": "<hawkish 或 neutral 或 dovish>",
  "conclusion": "<100字以内的政策走向结论>"
}}
""",
        expected_output="严格遵循格式的 JSON 对象，包含6个字段，不含任何 markdown 标记",
        agent=analyst,
    )

    crew = Crew(agents=[analyst], tasks=[task], verbose=True)
    result = crew.kickoff()

    raw = result.raw
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"无法从输出中提取 JSON：\n{raw}")

    data = json.loads(json_match.group())
    return SemanticFeature.model_validate(data)


# ── 存库 ─────────────────────────────────────────────────────────────────────

def save_semantic_to_db(package_id: int, feature: SemanticFeature) -> int:
    """把语义分析结果存入数据库，通过 package_id 关联到 MeetingPackage。"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_results (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            package_id        INTEGER NOT NULL,
            hawkish_score     REAL,
            inflation_concern REAL,
            growth_concern    REAL,
            key_evidence      TEXT,
            warsh_signal      TEXT,
            conclusion        TEXT,
            raw_json          TEXT,
            analyzed_at       TEXT DEFAULT (datetime('now'))
        )
    """)
    from datetime import datetime
    cursor = conn.execute("""
        INSERT INTO semantic_results
            (package_id, hawkish_score, inflation_concern, growth_concern,
             key_evidence, warsh_signal, conclusion, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        package_id,
        feature.hawkish_score,
        feature.inflation_concern,
        feature.growth_concern,
        json.dumps(feature.key_evidence, ensure_ascii=False),
        feature.warsh_signal,
        feature.conclusion,
        feature.model_dump_json(),
    ))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


# ── 单独运行入口（直接运行此文件时，从数据库取最新 MeetingPackage）─────────────

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT id, meeting_date, statement_text,
               cpi_yoy, unemployment_rate, fed_funds_rate, treasury_10y
        FROM meeting_packages ORDER BY id DESC LIMIT 1
    """).fetchone()
    conn.close()

    if not row:
        print("数据库中没有 MeetingPackage，请先运行 agent1_data_perception.py")
        raise SystemExit(1)

    pkg_id, meeting_date, statement_text, cpi, unrate, fedfunds, treasury = row
    macro = {
        "cpi_yoy": cpi,
        "unemployment_rate": unrate,
        "fed_funds_rate": fedfunds,
        "treasury_10y": treasury,
    }

    print("=" * 60)
    print("Agent 2：语义提取智能体")
    print(f"输入：MeetingPackage id={pkg_id}，会议日期={meeting_date}")
    print("=" * 60)

    feature = run_semantic_analysis(statement_text, meeting_date, macro)
    db_id = save_semantic_to_db(pkg_id, feature)

    print("\n" + "=" * 60)
    print("语义特征提取完成（SemanticFeature）：")
    print("=" * 60)
    print(f"鹰鸽评分         : {feature.hawkish_score:+.2f}")
    print(f"通胀担忧程度     : {feature.inflation_concern:.2f}")
    print(f"增长担忧程度     : {feature.growth_concern:.2f}")
    print(f"沃什信号         : {feature.warsh_signal}")
    print()
    print("关键证据：")
    for i, ev in enumerate(feature.key_evidence, 1):
        print(f"  {i}. {ev}")
    print()
    print(f"结论：\n  {feature.conclusion}")
    print()
    print(f"已存入数据库，semantic_results.id = {db_id}")
    print("=" * 60)
