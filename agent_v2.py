import os
import json
import re
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, LLM

load_dotenv()


class PolicyAnalysis(BaseModel):
    """
    结构化政策分析结果。
    Pydantic 会在解析时自动校验类型和范围约束，
    任何不合格的字段都会抛出 ValidationError。
    """

    hawkish_score: float = Field(
        description="鹰鸽评分，范围 -1.0 到 +1.0，正值=鹰派（偏紧缩），负值=鸽派（偏宽松）",
        ge=-1.0,
        le=1.0,
    )
    key_evidence: List[str] = Field(
        description="3到5条关键证据，直接引用声明原文中的具体词句"
    )
    conclusion: str = Field(
        description="政策走向结论，100字以内，说明沃什首次FOMC释放的政策信号"
    )


# 2026年6月17日 FOMC 声明（沃什首次主持，维持利率4.25-4.5%不变）
FOMC_STATEMENT = """
FOR RELEASE AT 2:00 P.M. EDT, JUNE 17, 2026

Recent indicators suggest that economic activity has continued to expand at a solid pace.
Labor market conditions have remained solid, with the unemployment rate remaining low.
Inflation remains somewhat elevated, though progress toward the 2 percent goal has been uneven.

The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent
over the longer run. The Committee judges that the risks to achieving its employment and
inflation goals are roughly in balance. The economic outlook remains uncertain.

In support of its goals, the Committee decided to maintain the target range for the federal
funds rate at 4-1/4 to 4-1/2 percent. In considering the extent and timing of additional
adjustments to the target range, the Committee will carefully assess incoming data, the
evolving outlook, and the balance of risks.

The Committee is strongly committed to returning inflation to its 2 percent objective.
The Fed under its new leadership emphasizes the importance of price stability as the
foundation for sustainable economic growth. While the Committee acknowledges that
policy adjustments may become appropriate, maintaining credibility on inflation remains paramount.

Voting for the monetary policy action were: Kevin Warsh, Chair, and other members of the Committee.
"""

llm = LLM(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

analyst = Agent(
    role="美联储政策分析师",
    goal="从美联储FOMC声明中精准提取货币政策情绪，给出可量化的结构化分析",
    backstory=(
        "你是顶尖宏观对冲基金的美联储分析师，有15年解读FOMC文件的经验。"
        "你擅长识别声明中的关键词语气变化，例如'elevated'比'high'更鹰，"
        "'patient'比'careful'更鸽。你的评分被交易团队直接用于调整仓位。"
    ),
    llm=llm,
    verbose=True,
)

task = Task(
    description=f"""请分析以下美联储FOMC声明，提取政策情绪。

声明原文：
{FOMC_STATEMENT}

你必须且只能输出一个合法的JSON对象，格式如下（不要加任何markdown代码块标记，不要加```）：
{{
  "hawkish_score": <-1.0到1.0之间的浮点数，正=鹰派，负=鸽派>,
  "key_evidence": [<证据1>, <证据2>, <证据3>],
  "conclusion": "<100字以内的政策走向结论>"
}}
""",
    expected_output=(
        "一个合法的JSON对象，包含三个字段："
        "hawkish_score（-1.0到1.0的浮点数）、"
        "key_evidence（3-5条字符串列表）、"
        "conclusion（100字以内的字符串）。"
        "不要加任何markdown标记，直接输出裸JSON。"
    ),
    agent=analyst,
)

crew = Crew(agents=[analyst], tasks=[task], verbose=True)

print("=" * 60)
print("agent_v2：语义提取智能体（结构化 Pydantic 输出版）")
print("=" * 60)

result = crew.kickoff()
raw_output = result.raw

print("\n[调试] 模型原始输出：")
print(raw_output)
print()

# 从模型输出中提取 JSON 对象
# 支持有或没有 markdown 代码块的情况
json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)

if not json_match:
    print("错误：无法从输出中提取 JSON，请检查模型输出格式")
    raise SystemExit(1)

try:
    data = json.loads(json_match.group())
    analysis = PolicyAnalysis.model_validate(data)
except (json.JSONDecodeError, Exception) as e:
    print(f"错误：JSON 解析或 Pydantic 校验失败：{e}")
    print("原始 JSON 文本：", json_match.group())
    raise SystemExit(1)

print("=" * 60)
print("结构化分析结果（Pydantic 已校验）：")
print("=" * 60)
print(f"鹰鸽评分（hawkish_score）: {analysis.hawkish_score:+.2f}")
print(f"类型验证：{type(analysis.hawkish_score).__name__}  ← 真正的 Python 浮点数，可做运算")
print()
print("关键证据（key_evidence）:")
for i, evidence in enumerate(analysis.key_evidence, 1):
    print(f"  {i}. {evidence}")
print()
print(f"结论（conclusion）:\n  {analysis.conclusion}")
print()

# 演示"可量化"的价值：根据分数自动给出交易信号
if analysis.hawkish_score > 0.3:
    signal = "做空国债（利率预期上升） / 做多美元"
elif analysis.hawkish_score < -0.3:
    signal = "做多国债（利率预期下降） / 做空美元"
else:
    signal = "中性，等待下一个经济数据触发信号"

print(f"基于 hawkish_score 自动生成的交易信号: {signal}")
print("=" * 60)
