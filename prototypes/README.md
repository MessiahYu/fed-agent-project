# 原型文件（Prototypes）

此目录存放早期实验脚本，已被生产代码替代，仅作学习参考。

| 文件 | 说明 |
|------|------|
| agent_v1.py | 第一个 CrewAI 智能体雏形，文本输出，hawkish_score +0.6 |
| agent_v2.py | 在 v1 上加 Pydantic 结构化输出，hawkish_score +0.40（float 类型） |

**不要在 pipeline.py 中 import 这里的文件。**
