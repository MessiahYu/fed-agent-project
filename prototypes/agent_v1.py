import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()

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
    goal="从美联储FOMC声明中精准提取货币政策情绪，给出量化的鹰鸽评分",
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

请输出三项内容：
1. hawkish_score：鹰鸽评分，范围 -1.0 到 +1.0
   - 正值 = 鹰派（偏紧缩、强调通胀风险）
   - 负值 = 鸽派（偏宽松、强调增长或就业风险）
   - 0 = 中性
2. key_evidence：3-5条关键证据，直接引用声明原文中的具体词句
3. conclusion：简洁结论，说明沃什首次FOMC的政策信号（100字以内）
""",
    expected_output="包含鹰鸽评分、关键证据列表和政策结论的分析报告",
    agent=analyst,
)

crew = Crew(agents=[analyst], tasks=[task], verbose=True)

print("=" * 60)
print("agent_v1：语义提取智能体（文本输出版）")
print("=" * 60)

result = crew.kickoff()

print("\n" + "=" * 60)
print("最终分析结果：")
print("=" * 60)
print(result)
