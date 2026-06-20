import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()

# 1) 配置大脑：让 CrewAI 用 DeepSeek
llm = LLM(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 2) 定义智能体：给它身份(role)、目标(goal)、背景(backstory)
analyst = Agent(
    role="美联储政策语义分析师",
    goal="读懂美联储声明的字里行间，判断其货币政策立场偏鹰还是偏鸽",
    backstory=(
        "你是一位资深宏观研究员，专门研究央行沟通的措辞艺术。"
        "你能从用词的细微变化里捕捉政策信号——"
        "比如'通胀仍然偏高'就比'通胀有所缓解'更鹰派。"
    ),
    llm=llm,
    verbose=True,
)

# 3) 准备一段真实的美联储声明（示例片段）
fed_statement = """
Recent indicators suggest that economic activity has continued to expand at a solid pace.
Job gains have remained solid, and the unemployment rate has stayed low.
Inflation has eased over the past year but remains somewhat elevated.
The Committee decided to maintain the target range for the federal funds rate at 4-1/4 to 4-1/2 percent.
In considering the extent and timing of additional adjustments, the Committee will carefully assess incoming data.
"""

# 4) 给智能体派一个具体任务(task)
task = Task(
    description=(
        f"请分析下面这段美联储 FOMC 声明的政策立场：\n\n{fed_statement}\n\n"
        "请输出三部分：\n"
        "1) 鹰鸽打分：一个 -1 到 +1 的数字（-1=极度鸽派/倾向宽松，+1=极度鹰派/倾向收紧，0=中性）；\n"
        "2) 判断依据：指出声明里最关键的 1-2 句措辞，并解释为什么它们指向这个方向；\n"
        "3) 一句话结论。"
    ),
    expected_output="包含'鹰鸽打分'、'判断依据'、'一句话结论'三部分的中文分析",
    agent=analyst,
)

# 5) 组队并启动
crew = Crew(agents=[analyst], tasks=[task], verbose=True)
result = crew.kickoff()

print("\n========= 分析结果 =========\n")
print(result)