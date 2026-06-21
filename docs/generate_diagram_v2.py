"""
生成 FedWatch AI 五智能体架构流程图 v2
运行: E:\Anaconda\envs\fed-agent\python.exe generate_diagram_v2.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 颜色 ─────────────────────────────────────────────────────────────────
CA1   = "#1565C0"   # 蓝   Agent 1
CA2   = "#6A1B9A"   # 紫   Agent 2
CA3   = "#2E7D32"   # 绿   Agent 3
CA4   = "#E65100"   # 橙   Agent 4
CA5   = "#1B5E20"   # 深绿 Agent 5
CHIST = "#F57F17"   # 琥珀 历史模块
CFUSE = "#B71C1C"   # 深红 融合层
CDB   = "#37474F"   # 钢灰 数据库
CPIPE = "#263238"   # 近黑 流水线
CDASH = "#0D47A1"   # 深蓝 仪表盘
CEXT  = "#E3F2FD"   # 浅蓝 外部源


def rect(ax, cx, cy, w, h, txt, fc, tc="white", fs=9.5, bold=True,
         lw=1.5, ec="#444"):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.13", linewidth=lw,
        edgecolor=ec, facecolor=fc, zorder=3
    )
    ax.add_patch(p)
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            zorder=4, multialignment="center", linespacing=1.45)


def arr(ax, x1, y1, x2, y2, c="#666", lw=1.8, dashed=False):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=c, lw=lw,
            linestyle="--" if dashed else "-",
            connectionstyle="arc3,rad=0.0"
        ),
        zorder=2
    )


def side_label(ax, x, y, txt, color, fs=7.8):
    ax.text(x, y, txt, ha="left", va="center",
            fontsize=fs, color=color, style="italic", zorder=5)


def bg_zone(ax, x, y, w, h, ec, fc, label_txt, label_y):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        linewidth=1.2, edgecolor=ec, facecolor=fc, alpha=0.3, zorder=1
    )
    ax.add_patch(p)
    ax.text(x + w / 2, label_y, label_txt, ha="center", va="center",
            fontsize=9, color=ec, fontweight="bold", zorder=2,
            bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.75))


# ── 画布 ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 28))
ax.set_xlim(0, 20)
ax.set_ylim(0, 28)
ax.axis("off")
fig.patch.set_facecolor("#F7F9FC")
ax.set_facecolor("#F7F9FC")

# ── 背景区 ───────────────────────────────────────────────────────────────
bg_zone(ax, 0.3, 16.2, 5.8, 8.8,  CHIST, "#FFF8E1", "历史回测轨道",  24.8)
bg_zone(ax, 13.9, 16.2, 5.8, 8.8, CA4,   "#FFF3E0", "并行增强智能体", 24.8)

# ── 标题 ─────────────────────────────────────────────────────────────────
ax.text(10, 27.2, "FedWatch AI — 五智能体美联储政策预测系统",
        ha="center", va="center", fontsize=17, fontweight="bold",
        color="#1A237E", zorder=5)
ax.text(10, 26.5,
        "架构流程图 v2.0  |  训练集: 2000-2020  "
        "测试集: 2021-2024  实时预测: 2025 至今",
        ha="center", va="center", fontsize=10.5, color="#555", zorder=5)

# ── 外部数据源 (y=25.3) ───────────────────────────────────────────────────
rect(ax, 7,  25.3, 5.5, 0.95,
     "美联储官网\nfederalreserve.gov",
     fc=CEXT, tc="#0D47A1", ec="#1565C0", bold=False, fs=9.5)
rect(ax, 14, 25.3, 5.5, 0.95,
     "FRED API\nfed.stlouisfed.org",
     fc=CEXT, tc="#0D47A1", ec="#1565C0", bold=False, fs=9.5)

arr(ax, 7,  24.82, 8.8,  24.0)   # 官网 → Agent1
arr(ax, 14, 24.82, 11.2, 24.0)   # FRED → Agent1

# ── Agent 1 (y=23.55) ────────────────────────────────────────────────────
rect(ax, 10, 23.55, 7.5, 1.1,
     "Agent 1  数据感知智能体\n"
     "自动抓取 FOMC 声明文本 + FRED 宏观快照 → MeetingPackage",
     fc=CA1)
side_label(ax, 13.9, 23.55, " ⟶ meeting_packages (SQLite)", CA1)
arr(ax, 10, 23.0, 10, 22.05)

# fomc_history（左轨，y=22.9）
rect(ax, 2.9, 22.9, 5.2, 1.0,
     "fomc_history  历史决策库\n206 次 FOMC 会议 (2000-2024)",
     fc=CHIST, fs=8.5)
side_label(ax, 0.35, 22.35, "SQLite: fomc_history", CHIST, fs=7.5)
arr(ax, 2.9, 22.4, 2.9, 21.35)

# Agent 4（右轨，y=22.5）
rect(ax, 17.0, 22.5, 5.2, 1.4,
     "Agent 4  主席行为蒸馏\n"
     "五位主席 ICL 投票\n"
     "格林斯潘→鲍威尔→沃什传承",
     fc=CA4, fs=8.5)
side_label(ax, 14.0, 21.85, "⟶ chair_distill_results", CA4, fs=7.5)
# fomc_history → Agent4
arr(ax, 5.5, 22.9, 14.4, 22.5)

# ── Agent 2 (y=21.55) ────────────────────────────────────────────────────
rect(ax, 10, 21.55, 7.5, 1.1,
     "Agent 2  语义提取智能体\n"
     "DeepSeek LLM → 鹰鸽评分 / 通胀担忧 / 沃什信号 / 关键句证据",
     fc=CA2)
side_label(ax, 13.9, 21.55, " ⟶ semantic_results (SQLite)", CA2)
arr(ax, 10, 21.0, 10, 20.05)

# backtest（左轨，y=21.0）
rect(ax, 2.9, 21.0, 5.2, 0.95,
     "backtest.py  回测框架\nWalk-forward VAR(4) → 准确率 87.5%",
     fc=CHIST, fs=8.5)
side_label(ax, 0.35, 20.45, "防前视偏差 | 混淆矩阵", CHIST, fs=7.5)
ax.text(2.9, 20.55,
        "训练 2000-2020 / 测试 2021-2024",
        ha="center", va="center", fontsize=7.2, color="white",
        bbox=dict(fc=CHIST, ec="none", pad=2.5, alpha=0.9), zorder=5)

# Agent 5（右轨，y=20.6）
rect(ax, 17.0, 20.6, 5.2, 1.6,
     "Agent 5  影响者联合\n"
     "市场: 收益率曲线 10Y-2Y\n"
     "政治: 白宫压力 / 独立性风险\n"
     "国际: ECB / 美元趋势",
     fc=CA5, fs=8.5)
side_label(ax, 14.0, 19.7, "⟶ influencer_signals", CA5, fs=7.5)
# FRED → Agent5（虚线，复用同一数据源）
arr(ax, 16.75, 24.82, 17.0, 21.4, c=CA5, dashed=True, lw=1.4)

# ── Agent 3 (y=19.55) ────────────────────────────────────────────────────
rect(ax, 10, 19.55, 7.5, 1.3,
     "Agent 3  决策推演智能体\n"
     "VAR(4) 计量基准 + Chain-of-Draft 五步推理\n"
     "→ FOMCPrediction",
     fc=CA3)
side_label(ax, 13.9, 19.55, " ⟶ fomc_predictions (SQLite)", CA3)
arr(ax, 10, 18.9, 10, 17.9)

# ── 五路信号融合 (y=17.3) ─────────────────────────────────────────────────
rect(ax, 10, 17.3, 17.5, 1.5,
     "五路信号加权融合  ·  最终共识预测\n"
     "Agent3-VAR 35%  +  Agent3-LLM 35%  +  Agent4-蒸馏 15%  +  "
     "Agent5-市场 8%  +  Agent5-综合 7%\n"
     "► 输出: HOLD / CUT / HIKE  +  加权置信度",
     fc=CFUSE)

# 并行 Agent → 融合层
arr(ax, 17.0, 21.8, 18.5, 17.75)   # Agent4 → fusion right
arr(ax, 17.0, 19.8, 18.5, 16.85)   # Agent5 → fusion right
# backtest → fusion（虚线，表示精度验证参考）
arr(ax, 2.9, 20.52, 1.5, 17.75, c=CHIST, dashed=True, lw=1.3)

arr(ax, 10, 16.55, 10, 15.6)

# ── SQLite 数据血统链 (y=15.1) ────────────────────────────────────────────
rect(ax, 10, 15.1, 15, 1.3,
     "SQLite 数据库 (fed_watch.db)  ·  完整数据血统链（六张表）\n"
     "meeting_packages  |  semantic_results  |  fomc_predictions  "
     "|  chair_distill_results  |  influencer_signals  |  fomc_history",
     fc=CDB)
arr(ax, 10, 14.45, 10, 13.5)

# ── pipeline.py (y=13.05) ─────────────────────────────────────────────────
rect(ax, 10, 13.05, 9, 1.0,
     "pipeline.py  五阶段流水线编排\n"
     "顺序执行 A1 → A2 → A3 → A4 → A5，输出综合分析报告",
     fc=CPIPE)
arr(ax, 10, 12.55, 10, 11.6)

# ── Streamlit 仪表盘 (y=11.15) ────────────────────────────────────────────
rect(ax, 10, 11.15, 9, 1.0,
     "Streamlit 仪表盘  (app.py  |  http://localhost:8501)\n"
     "鹰鸽仪表盘 / 五路信号 / 主席投票 / 风险因素 / 历史趋势",
     fc=CDASH)

# ── 理论注记 ─────────────────────────────────────────────────────────────
ax.text(10, 9.8,
        "理论基础: VAR(4) + LLM Chain-of-Draft + In-Context Learning + "
        "收益率曲线 + 政治经济学 + 国际货币传导",
        ha="center", va="center", fontsize=9.5, color="#333",
        style="italic", zorder=4)
ax.text(10, 9.25,
        "核心论证: 多智能体 LLM 在精度 / 时效 / 证据链可解释性上全面优于传统 VAR/VECM",
        ha="center", va="center", fontsize=9.5, color="#333",
        style="italic", zorder=4)

# ── 图例 ─────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=CA1,   label="Agent 1  数据感知"),
    mpatches.Patch(color=CA2,   label="Agent 2  语义提取"),
    mpatches.Patch(color=CA3,   label="Agent 3  决策推演"),
    mpatches.Patch(color=CA4,   label="Agent 4  主席行为蒸馏"),
    mpatches.Patch(color=CA5,   label="Agent 5  影响者联合"),
    mpatches.Patch(color=CHIST, label="历史回测模块"),
    mpatches.Patch(color=CFUSE, label="五路信号融合层"),
    mpatches.Patch(color=CDB,   label="SQLite 持久化存储"),
    mpatches.Patch(color=CPIPE, label="流水线编排"),
    mpatches.Patch(color=CDASH, label="Streamlit 仪表盘"),
]
ax.legend(
    handles=handles, loc="lower center", fontsize=9.5,
    framealpha=0.95, ncol=5, bbox_to_anchor=(0.5, 0.005),
    title="模块图例", title_fontsize=10,
    edgecolor="#ccc"
)

# ── 数据流方向标注 ────────────────────────────────────────────────────────
ax.text(10.3, 22.5, "声明文本 + 宏观数据",
        ha="left", va="center", fontsize=7.5, color="#888", zorder=5)
ax.text(10.3, 20.5, "MeetingPackage",
        ha="left", va="center", fontsize=7.5, color="#888", zorder=5)
ax.text(10.3, 18.5, "SemanticFeature",
        ha="left", va="center", fontsize=7.5, color="#888", zorder=5)

plt.tight_layout(pad=1.5)
plt.savefig(
    "architecture_flowchart.png",
    dpi=150, bbox_inches="tight",
    facecolor="#F7F9FC"
)
plt.close()
print("已生成: architecture_flowchart.png")
