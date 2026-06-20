"""生成项目全景流程图，保存为 project_flowchart.png"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 中文字体 + emoji 字体
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "Segoe UI Emoji", "SimHei", "Arial Unicode MS", "DejaVu Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 画布 ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 26))
fig.patch.set_facecolor("#EEF2F7")
ax.set_facecolor("#EEF2F7")
ax.set_xlim(0, 14)
ax.set_ylim(0, 26)
ax.axis("off")


# ── 工具函数 ──────────────────────────────────────────────────────────
def box(x, y, w, h, fc, ec="white", lw=2, alpha=1.0, radius=0.3):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=2,
    )
    ax.add_patch(p)

def txt(x, y, s, size=10, weight="normal", color="white", ha="center", va="center", style="normal"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, fontweight=weight,
            color=color, style=style, zorder=4)

def arrow(x1, y1, x2, y2, color="#445566", lw=2.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=18, connectionstyle="arc3,rad=0"),
        zorder=3,
    )

def tag(x, y, label, fc, tc="white"):
    box(x, y, 2.2, 0.55, fc=fc, radius=0.15)
    txt(x + 1.1, y + 0.28, label, size=8, weight="bold", color=tc)


# ═══════════════════════════════════════════════════════
# 大标题
# ═══════════════════════════════════════════════════════
txt(7, 25.3, "FedWatch AI  —  项目全景流程图", size=17, weight="bold", color="#1A237E")
txt(7, 24.75, "美联储议息会议政策走向预测  ·  多智能体编排架构  ·  同花顺宏观研究员笔试项目",
    size=10, color="#3949AB")

# ═══════════════════════════════════════════════════════
# [0] 笔试目标
# ═══════════════════════════════════════════════════════
box(1, 23.0, 12, 1.3, fc="#1A237E", radius=0.25)
txt(7, 23.85, "[ 笔试目标 ]", size=12, weight="bold")
txt(7, 23.35,
    "预测美联储 FOMC 利率决策方向  +  推演新主席沃什政策脉络  +  给出下次会议（2026-07-28/29）预测",
    size=9, color="#C5CAE9")

arrow(7, 23.0, 7, 22.45)

# ═══════════════════════════════════════════════════════
# Agent 1 — 数据感知
# ═══════════════════════════════════════════════════════
box(0.5, 19.5, 13, 2.65, fc="#1565C0", radius=0.3)
tag(10.7, 21.8, "[已完成]", fc="#0D47A1")

txt(7, 21.75, "Agent 1  —  数据感知智能体", size=13, weight="bold")
txt(7, 21.25,
    "目的：自动抓取、清洗原始数据，为后续 Agent 提供统一格式的原材料输入",
    size=9, color="#BBDEFB")

# 两个数据源子框
box(1.0, 19.8, 5.3, 1.15, fc="#1976D2", ec="#90CAF9", lw=1.5, radius=0.2)
txt(3.65, 20.62, "federalreserve.gov", size=9, weight="bold", color="#E3F2FD")
txt(3.65, 20.25, "FOMC 声明全文爬虫  (BeautifulSoup)", size=8, color="#BBDEFB")

box(7.2, 19.8, 5.8, 1.15, fc="#1976D2", ec="#90CAF9", lw=1.5, radius=0.2)
txt(10.1, 20.62, "FRED API（圣路易斯联储，免费）", size=9, weight="bold", color="#E3F2FD")
txt(10.1, 20.25, "CPI  /  失业率  /  联邦基金利率  /  10Y国债", size=8, color="#BBDEFB")

# 输出条
box(2.5, 19.52, 9.0, 0.3, fc="#0D47A1", ec="none", lw=0, radius=0.05)
txt(7, 19.67,
    "输出: MeetingPackage（声明正文 + 宏观指标快照 + 数据截断日期，写入 SQLite）",
    size=8, color="#E3F2FD", weight="bold")

arrow(7, 19.5, 7, 18.9)
txt(7, 19.2, "MeetingPackage", size=8.5, color="#445566", style="italic")

# ═══════════════════════════════════════════════════════
# Agent 2 — 语义提取
# ═══════════════════════════════════════════════════════
box(0.5, 16.3, 13, 2.6, fc="#6A1B9A", radius=0.3)
tag(10.7, 18.55, "[已完成]", fc="#4A148C")

txt(7, 18.55, "Agent 2  —  语义提取智能体", size=13, weight="bold")
txt(7, 18.05,
    "目的：把声明「自然语言文字」转换成程序可计算的「量化特征向量」",
    size=9, color="#E1BEE7")

box(1.0, 16.6, 5.5, 1.35, fc="#7B1FA2", ec="#CE93D8", lw=1.5, radius=0.2)
txt(3.75, 17.47, "DeepSeek LLM  +  CrewAI 框架", size=9, weight="bold", color="#F3E5F5")
txt(3.75, 17.1, "读取声明  →  识别鹰鸽细节措辞", size=8, color="#E1BEE7")
txt(3.75, 16.78, "\"somewhat elevated\" 比 \"elevated\" 更鸽", size=7.5, color="#CE93D8", style="italic")

box(7.0, 16.6, 6.0, 1.35, fc="#7B1FA2", ec="#CE93D8", lw=1.5, radius=0.2)
txt(10.0, 17.47, "Pydantic 结构化输出校验", size=9, weight="bold", color="#F3E5F5")
txt(10.0, 17.1, "hawkish_score: +0.40  (真正的 float)", size=8.5, color="#E1BEE7")
txt(10.0, 16.78, "inflation_concern: 0.80  |  warsh_signal: hawkish", size=7.5, color="#CE93D8")

box(2.5, 16.32, 9.0, 0.3, fc="#4A148C", ec="none", lw=0, radius=0.05)
txt(7, 16.47,
    "输出: SemanticFeature（6个量化字段 + 关键证据列表，写入 SQLite）",
    size=8, color="#E1BEE7", weight="bold")

arrow(7, 16.3, 7, 15.7)
txt(7, 16.0, "SemanticFeature", size=8.5, color="#445566", style="italic")

# ═══════════════════════════════════════════════════════
# Agent 3 — 决策推演
# ═══════════════════════════════════════════════════════
box(0.5, 12.0, 13, 3.7, fc="#2E7D32", radius=0.3)
tag(10.7, 15.35, "[已完成]", fc="#1B5E20")

txt(7, 15.35, "Agent 3  —  决策推演智能体", size=13, weight="bold")
txt(7, 14.88,
    "目的：融合「计量统计」和「LLM 推理」双轨方法，给出下次会议决策预测",
    size=9, color="#C8E6C9")

# 子模块 A
box(1.0, 12.3, 5.2, 2.2, fc="#388E3C", ec="#A5D6A7", lw=1.5, radius=0.25)
txt(3.6, 14.18, "子模块 A", size=9, weight="bold")
txt(3.6, 13.85, "VAR 计量模型", size=10, weight="bold", color="#E8F5E9")
txt(3.6, 13.5, "statsmodels  ·  5年月度历史数据", size=8, color="#C8E6C9")
txt(3.6, 13.17, "变量：FEDFUNDS  /  CPI  /  UNRATE", size=8, color="#C8E6C9")
txt(3.6, 12.83, "拟合 VAR(4)，预测下期利率：3.587%", size=8, color="#A5D6A7")
txt(3.6, 12.5, "VAR 信号：HOLD", size=9, weight="bold", color="#FFEB3B")

# 子模块 B
box(7.3, 12.3, 5.8, 2.2, fc="#33691E", ec="#AED581", lw=1.5, radius=0.25)
txt(10.2, 14.18, "子模块 B", size=9, weight="bold")
txt(10.2, 13.85, "LLM 链式草稿（CoD）", size=10, weight="bold", color="#F1F8E9")
txt(10.2, 13.5, "DeepSeek  +  ICL 历史案例注入", size=8, color="#DCEDC8")
txt(10.2, 13.17, "Step1通胀  Step2就业  Step3金融条件", size=8, color="#DCEDC8")
txt(10.2, 12.83, "Step4沃什立场  Step5历史会议对标", size=8, color="#DCEDC8")
txt(10.2, 12.5, "LLM 信号：HOLD  (置信度 85%)", size=9, weight="bold", color="#FFEB3B")

# 融合
arrow(3.6, 12.3, 5.8, 12.12, color="#A5D6A7", lw=1.5)
arrow(10.2, 12.3, 8.2, 12.12, color="#AED581", lw=1.5)
txt(7, 12.02, "(+)  加权融合  —  方向一致则置信度加成", size=8.5, weight="bold", color="#FFEB3B")

box(2.5, 12.02, 9.0, 0.3, fc="#1B5E20", ec="none", lw=0, radius=0.05)
txt(7, 12.17,
    "最终输出: FOMCPrediction  —  HOLD  综合置信度 77%（写入 SQLite）",
    size=8, color="#C8E6C9", weight="bold")

arrow(7, 12.0, 7, 11.4)
txt(7, 11.7, "FOMCPrediction", size=8.5, color="#445566", style="italic")

# ═══════════════════════════════════════════════════════
# SQLite
# ═══════════════════════════════════════════════════════
box(0.5, 9.8, 13, 1.5, fc="#BF360C", radius=0.3)
tag(10.7, 10.95, "[已完成]", fc="#870000")

txt(7, 11.0, "SQLite 数据库  —  fed_watch.db", size=12, weight="bold")
txt(7, 10.55, "目的：持久化存储每次分析结果，形成可审计的「数据血统链」", size=9, color="#FFCCBC")
txt(7, 10.1,
    "meeting_packages  ->  semantic_results  ->  fomc_predictions  （每层含 raw_json 字段，完整审计）",
    size=8.5, color="#FFAB91")

arrow(7, 9.8, 7, 9.2)

# ═══════════════════════════════════════════════════════
# pipeline.py
# ═══════════════════════════════════════════════════════
box(0.5, 7.8, 13, 1.3, fc="#00695C", radius=0.3)
tag(10.7, 8.75, "[已完成]", fc="#004D40")

txt(7, 8.75, "pipeline.py  —  三智能体串联调度器", size=12, weight="bold")
txt(7, 8.3,
    "目的：一键执行完整流水线 Agent1 -> Agent2 -> Agent3，实现端到端自动化闭环",
    size=9, color="#B2DFDB")

arrow(7, 7.8, 7, 7.2)

# ═══════════════════════════════════════════════════════
# Streamlit
# ═══════════════════════════════════════════════════════
box(0.5, 5.8, 13, 1.8, fc="#00838F", radius=0.3)
tag(10.7, 7.25, "[已完成]", fc="#006064")

txt(7, 7.3, "Streamlit 仪表盘  —  app.py        http://localhost:8501", size=12, weight="bold")
txt(7, 6.88, "目的：可视化演示系统输出，提供「一键触发」按钮，展示完整分析证据链", size=9, color="#B2EBF2")
txt(7, 6.48, "Tab1: 综合仪表盘（4个宏观指标卡 + 预测结论横幅 + 鹰鸽仪表盘 + 雷达图）", size=8, color="#80DEEA")
txt(7, 6.1,  "Tab2: 分析详情（声明原文 + CoD五步推理 + 风险因素 + 证据链）    Tab3: 系统信息", size=8, color="#80DEEA")

arrow(7, 5.8, 7, 5.2)

# ═══════════════════════════════════════════════════════
# 理论论证（待完成）
# ═══════════════════════════════════════════════════════
box(0.5, 3.8, 13, 1.8, fc="#EEEEEE", ec="#BDBDBD", lw=2.5, radius=0.3)
tag(10.7, 5.25, "[下一步]", fc="#757575", tc="white")

txt(7, 5.3, "理论论证  —  待完成", size=12, weight="bold", color="#333")
txt(7, 4.88,
    "目的：从经济学「内生性」和「因果识别」角度，论证 Agent 方案优于传统 VAR/VECM 计量模型",
    size=9, color="#555")
txt(7, 4.48, "参考：Taylor(1993) 泰勒规则  ·  Romer & Romer(2004) 内生性识别", size=8, color="#777")
txt(7, 4.1,  "参考：Hansen & McMahon(2016) 文本量化  ·  对比维度：精度 / 时效 / 可解释性", size=8, color="#777")

# ═══════════════════════════════════════════════════════
# 底部图例
# ═══════════════════════════════════════════════════════
box(0.5, 1.6, 3.8, 0.9, fc="#2E7D32", ec="white", radius=0.2)
txt(2.4, 2.05, "[已完成]  模块已上线", size=9, weight="bold")

box(4.8, 1.6, 4.4, 0.9, fc="#EEEEEE", ec="#BDBDBD", lw=2, radius=0.2)
txt(7.0, 2.05, "[下一步]  待完成", size=9, weight="bold", color="#555")

box(9.7, 1.6, 3.8, 0.9, fc="#1A237E", ec="white", radius=0.2)
txt(11.6, 2.05, "->  数据流方向", size=9, weight="bold")

txt(7, 1.1,
    "技术栈：CrewAI  ·  DeepSeek  ·  statsmodels  ·  pandas  ·  sqlite3  ·  Streamlit  ·  plotly  ·  BeautifulSoup",
    size=8, color="#555")
txt(7, 0.65,
    "学术借鉴：FedSight AI (NeurIPS 2025)  —  Chain-of-Draft 推理  ·  ICL 历史案例  ·  防前视偏差数据截断",
    size=8, color="#777")

plt.tight_layout(pad=0.5)
out_path = r"c:\Users\余青锋\OneDrive\fed-agent-project\project_flowchart.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor="#EEF2F7", edgecolor="none")
print(f"✓ 流程图已保存：{out_path}")
