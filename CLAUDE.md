# 项目交接简报（CLAUDE.md）

## 0. 给接手 Claude Code 的话
- 用户是编程小白，请用通俗语言解释每一步，边做边教，全程中文回复。
- 每写一段代码，先讲它干嘛、为什么这么写。
- 增量推进：每完成一小块就运行验证，跑通再继续。
- 这是同花顺"宏观研究员"岗位笔试，用户最终要能亲口讲清整个系统，所以"看懂"比"跑通"更重要。
- 严守安全：绝不打印或提交 API 密钥。

## 1. 项目目标（笔试题）
搭一个基于多智能体编排的"持续性经济事件预测系统"，预测美联储议息会议政策走向。需：
- 预测年内美联储政策走向 + 推理新主席沃什(Kevin Warsh)政策脉络 + 给出下次会议决策方向。
- 系统含三个智能体：数据感知、语义特征提取、决策推演。
- 实现"数据触发 → 实时滚动预测 → 自动风险归因"闭环。
- 从"内生性 / 因果识别"角度论证 Agent 方案优于传统计量(VAR/VECM)，强调精度、时效、证据链可解释性。

## 2. 现实背景（关键）
- Kevin Warsh 已于 2026-05-22 接任美联储主席（接替鲍威尔）。
- 其首次 FOMC 为 2026-06-16/17，维持利率不变。
- 立场偏鹰（强调通胀纪律、精简沟通）但也称有降息空间，存在矛盾，是分析重点。

## 3. 已就绪的环境
- 系统 Windows；VS Code 已装 Python 扩展。
- conda 环境名 `fed-agent`，Python 3.11.15（VS Code 解释器已指向它）。
- PowerShell 已 conda init + 执行策略 RemoteSigned，终端可自动激活 fed-agent。
- 已装库：crewai、crewai-tools、streamlit、openai、python-dotenv（均在 fed-agent 内）。

## 4. 大模型配置（用 DeepSeek，便宜）
- 模型 deepseek-chat，OpenAI 兼容接口，base_url = https://api.deepseek.com
- 密钥在项目根目录 .env，变量名 DEEPSEEK_API_KEY；.gitignore 已排除 .env 与 __pycache__/。
- CrewAI 用法：
  LLM(model="deepseek/deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

## 5. 现有文件与进度
- test.py：环境测试，已通过。
- check_brain.py：直接调用 DeepSeek 验证连通，已通过。
- agent_v1.py：第一个 CrewAI 智能体（"语义提取智能体"雏形），读美联储声明 → 输出鹰鸽分+依据+结论，已跑通（示例给 +0.1，质量不错）。
- agent_v2.py：在 v1 上加 Pydantic 结构化输出（PolicyAnalysis: hawkish_score / key_evidence / conclusion），使分数成为可计算数字。【需确认是否已成功运行】
- 三个 .md：岗位描述与笔试题原文，可作背景参考。

## 6. 已确立的 CrewAI 架构模式
LLM(大脑=DeepSeek) → Agent(角色/目标/背景) → Task(带 output_pydantic 结构化输出) → Crew.kickoff()

## 7. 路线图（按序推进）
1. 确认 agent_v2.py 结构化输出跑通。
2. 造【数据感知智能体】：自动抓美联储官网(federalreserve.gov)声明/纪要，接 FRED 宏观数据(fred.stlouisfed.org，有免费API)。
3. 数据感知 → 语义提取 串成流水线，真实数据喂给分析师。
4. 造【决策推演智能体】：情绪特征 + statsmodels 的 VAR/VECM 融合，必要时加时序注意力(PyTorch)，输出下次 FOMC 预测。
5. 闭环：APScheduler 定时调度 + SQLite 存储，实现"数据触发→滚动预测→风险归因"。
6. Streamlit 做演示仪表盘。
7. 写理论论证：内生性、因果识别，论证 Agent 方案在精度/时效/证据链可解释性上优于传统计量。

## 8. 已确认的技术细节

### DeepSeek 结构化输出方案（重要！）
- DeepSeek 不支持 OpenAI 的 `json_schema` response_format，直接用 CrewAI 的 `output_pydantic` 会报 400 错误。
- 正确做法：**不用** `output_pydantic`；在 Task description 里明确要求输出裸 JSON，然后用 `re.search(r'\{.*\}', raw, re.DOTALL)` 提取，再用 `PolicyAnalysis.model_validate(data)` 校验。
- Python 解释器路径：`E:\Anaconda\envs\fed-agent\python.exe`（直接用此路径运行脚本）

### 已跑通文件
- agent_v1.py：文本输出，hawkish_score +0.6，分析质量优秀
- agent_v2.py：结构化输出，hawkish_score +0.40（float 类型已校验），自动交易信号"做空国债/做多美元"

## 9. 当前这一步
开始路线图第 2 项：数据感知智能体（自动抓取美联储声明 + FRED 宏观数据）。