import json
import os
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.db import DB_PATH

# ── 页面基础配置 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FedWatch AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 全局样式
st.markdown("""
<style>
.big-prediction {
    text-align: center;
    padding: 24px 16px;
    border-radius: 12px;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 2px;
    margin: 12px 0;
}
.hold-box  { background:#FFF8E1; border:3px solid #FFC107; color:#E65100; }
.cut-box   { background:#E3F2FD; border:3px solid #2196F3; color:#0D47A1; }
.hike-box  { background:#FFEBEE; border:3px solid #F44336; color:#B71C1C; }
.cod-step  {
    background:#F5F5F5; border-left:4px solid #1976D2;
    padding:10px 16px; margin:6px 0; border-radius:0 8px 8px 0;
    font-size:0.95rem;
}
.risk-item {
    background:#FFF3E0; border-left:4px solid #FF9800;
    padding:8px 14px; margin:4px 0; border-radius:0 6px 6px 0;
}
.evidence-item {
    background:#E8F5E9; border-left:4px solid #4CAF50;
    padding:8px 14px; margin:4px 0; border-radius:0 6px 6px 0;
}
.agent-tag {
    display:inline-block; padding:2px 10px; border-radius:12px;
    font-size:0.8rem; font-weight:600; margin-right:6px;
}
.a1-tag { background:#E3F2FD; color:#1565C0; }
.a2-tag { background:#F3E5F5; color:#6A1B9A; }
.a3-tag { background:#E8F5E9; color:#2E7D32; }
</style>
""", unsafe_allow_html=True)


# ── 数据加载 ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def load_latest():
    """从 SQLite 加载最新的完整分析结果（三表联查）"""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("""
            SELECT
                mp.meeting_date, mp.statement_text, mp.statement_url,
                mp.cpi_yoy, mp.unemployment_rate, mp.fed_funds_rate, mp.treasury_10y,
                mp.data_cutoff_date, mp.packaged_at, mp.source,
                sr.hawkish_score, sr.inflation_concern, sr.growth_concern,
                sr.key_evidence, sr.warsh_signal, sr.conclusion as sem_conclusion,
                fp.next_meeting_date, fp.predicted_action, fp.confidence,
                fp.var_signal, fp.var_forecast_rate, fp.llm_signal,
                fp.cod_steps, fp.risk_factors, fp.evidence_chain,
                fp.final_conclusion, fp.predicted_at
            FROM fomc_predictions fp
            JOIN semantic_results sr ON fp.semantic_id = sr.id
            JOIN meeting_packages mp ON sr.package_id = mp.id
            ORDER BY fp.id DESC LIMIT 1
        """).fetchone()
        conn.close()
        if not row:
            return None
        cols = [
            "meeting_date","statement_text","statement_url",
            "cpi_yoy","unemployment_rate","fed_funds_rate","treasury_10y",
            "data_cutoff_date","packaged_at","source",
            "hawkish_score","inflation_concern","growth_concern",
            "key_evidence","warsh_signal","sem_conclusion",
            "next_meeting_date","predicted_action","confidence",
            "var_signal","var_forecast_rate","llm_signal",
            "cod_steps","risk_factors","evidence_chain",
            "final_conclusion","predicted_at",
        ]
        d = dict(zip(cols, row))
        for f in ["key_evidence","cod_steps","risk_factors","evidence_chain"]:
            d[f] = json.loads(d[f]) if d[f] else []
        return d
    except Exception as e:
        return None


@st.cache_data(ttl=10)
def load_distill():
    """加载最新主席蒸馏结果"""
    try:
        conn = sqlite3.connect(DB_PATH)
        row  = conn.execute(
            "SELECT * FROM chair_distill_results ORDER BY id DESC LIMIT 1"
        ).fetchone()
        cols = [d[0] for d in conn.execute("PRAGMA table_info(chair_distill_results)").fetchall()]
        conn.close()
        return dict(zip(cols, row)) if row else None
    except Exception:
        return None


@st.cache_data(ttl=10)
def load_influencer():
    """加载最新影响者信号"""
    try:
        conn = sqlite3.connect(DB_PATH)
        row  = conn.execute(
            "SELECT raw_json FROM influencer_signals ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


@st.cache_data(ttl=10)
def load_history():
    """加载历史鹰鸽评分，用于趋势图"""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT mp.meeting_date, sr.hawkish_score, sr.warsh_signal, fp.predicted_action
            FROM fomc_predictions fp
            JOIN semantic_results sr ON fp.semantic_id = sr.id
            JOIN meeting_packages mp ON sr.package_id = mp.id
            ORDER BY fp.id ASC
        """).fetchall()
        conn.close()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["meeting_date","hawkish_score","warsh_signal","predicted_action"])
        df["run_no"] = range(1, len(df)+1)
        return df
    except Exception:
        return pd.DataFrame()


# ── 侧边栏 ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🏦 FedWatch AI")
    st.markdown("**美联储政策预测系统**")
    st.markdown("---")

    st.markdown("### 🚀 更新分析")
    st.markdown("点击按钮重新运行完整流水线（约 2–3 分钟）")

    if st.button("▶ 运行五智能体流水线", type="primary", use_container_width=True):
        with st.spinner("Agent 1 数据感知中..."):
            try:
                from agent1_data_perception import main as run_agent1, save_to_db
                package   = run_agent1()
                pkg_db_id = save_to_db(package)
            except Exception as e:
                st.error(f"Agent 1 失败：{e}")
                st.stop()

        with st.spinner("Agent 2 语义提取中..."):
            try:
                from agent2_semantic import run_semantic_analysis, save_semantic_to_db
                macro = {
                    "cpi_yoy":           package.macro.cpi_yoy,
                    "unemployment_rate": package.macro.unemployment_rate,
                    "fed_funds_rate":    package.macro.fed_funds_rate,
                    "treasury_10y":      package.macro.treasury_10y,
                }
                feature   = run_semantic_analysis(package.statement_text, package.meeting_date, macro)
                sem_db_id = save_semantic_to_db(pkg_db_id, feature)
            except Exception as e:
                st.error(f"Agent 2 失败：{e}")
                st.stop()

        with st.spinner("Agent 3 决策推演中..."):
            try:
                from agent3_decision import run_decision_agent
                sem_dict = {
                    "hawkish_score":     feature.hawkish_score,
                    "inflation_concern": feature.inflation_concern,
                    "growth_concern":    feature.growth_concern,
                    "warsh_signal":      feature.warsh_signal,
                }
                run_decision_agent(sem_db_id, sem_dict, macro)
            except Exception as e:
                st.error(f"Agent 3 失败：{e}")
                st.stop()

        with st.spinner("Agent 4 主席行为蒸馏中..."):
            try:
                from agent4_chair_distill import run_chair_distill
                run_chair_distill(macro, package.meeting_date)
            except Exception as e:
                st.error(f"Agent 4 失败：{e}")
                st.stop()

        with st.spinner("Agent 5 影响者信号分析中..."):
            try:
                from agent5_influencers import run_influencer_agents
                run_influencer_agents(macro, package.meeting_date)
            except Exception as e:
                st.error(f"Agent 5 失败：{e}")
                st.stop()

        st.success("✅ 五智能体流水线完成！页面自动刷新。")
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📐 系统架构")
    st.markdown("""
- <span class='agent-tag a1-tag'>Agent 1</span> 数据感知<br>
  美联储官网 + FRED API
- <span class='agent-tag a2-tag'>Agent 2</span> 语义提取<br>
  DeepSeek + Pydantic
- <span class='agent-tag a3-tag'>Agent 3</span> 决策推演<br>
  VAR 计量 + LLM 链式草稿
- <span class='agent-tag' style='background:#FFF3E0;color:#E65100;'>Agent 4</span> 主席行为蒸馏<br>
  五位主席 ICL + 沃什预测
- <span class='agent-tag' style='background:#E8F5E9;color:#1B5E20;'>Agent 5</span> 影响者联合<br>
  市场预期 + 政治压力 + 国际联动
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 参考文献")
    st.markdown("""
- Taylor (1993) 泰勒规则
- Romer & Romer (2004) 因果识别
- Hansen & McMahon (2016) 文本量化
- FedSight AI (NeurIPS 2025)
""")


# ── 主内容区 ──────────────────────────────────────────────────────────────────

d = load_latest()

if d is None:
    st.title("🏦 FedWatch AI")
    st.warning("数据库中暂无分析结果，请先点击左侧 **运行五智能体流水线** 按钮。")
    st.stop()

# 标题行
col_title, col_time = st.columns([3, 1])
with col_title:
    st.title("🏦 FedWatch AI — 美联储政策预测系统")
with col_time:
    st.markdown(f"<br>最新分析：**{d['predicted_at'][:16]}**", unsafe_allow_html=True)
    st.caption(f"数据来源：{d['source'].upper()}")

tab1, tab2, tab3 = st.tabs(["📊 综合仪表盘", "📝 分析详情", "🔬 系统信息"])


# ════════════════════════════════════════════════════════════════
# Tab 1：综合仪表盘
# ════════════════════════════════════════════════════════════════
with tab1:

    # ── 宏观指标快照 ────────────────────────────────────────────
    st.markdown(f"<span class='agent-tag a1-tag'>Agent 1</span> **宏观数据快照** — 数据截断日期：{d['data_cutoff_date']}（防前视偏差）", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌡️ CPI 同比",        f"{d['cpi_yoy']}%",           delta=f"{d['cpi_yoy']-2:.2f}% vs 目标", delta_color="inverse")
    c2.metric("👥 失业率",           f"{d['unemployment_rate']}%", delta="自然率约 4.0%",                  delta_color="off")
    c3.metric("🏦 联邦基金利率",      f"{d['fed_funds_rate']}%",    delta="区间 3.5–3.75%",                 delta_color="off")
    c4.metric("📈 10Y 国债收益率",    f"{d['treasury_10y']}%",      delta=f"利差 {d['treasury_10y']-d['fed_funds_rate']:.2f}%", delta_color="off")

    st.divider()

    # ── 预测结论横幅 ─────────────────────────────────────────────
    st.markdown(f"<span class='agent-tag a3-tag'>Agent 3</span> **下次会议预测** — {d['next_meeting_date']}", unsafe_allow_html=True)

    action = d["predicted_action"].upper()
    css_cls = "hold-box" if "HOLD" in action else ("cut-box" if "CUT" in action else "hike-box")
    action_zh = {"HOLD":"维持不变 HOLD", "CUT_25BP":"降息 25bp CUT", "CUT_50BP":"降息 50bp CUT",
                 "HIKE_25BP":"加息 25bp HIKE"}.get(action, action)

    st.markdown(f"<div class='big-prediction {css_cls}'>{action_zh}</div>", unsafe_allow_html=True)

    col_conf, col_signals = st.columns([1, 2])
    with col_conf:
        conf_pct = int(d["confidence"] * 100)
        st.markdown(f"**综合置信度：{conf_pct}%**")
        st.progress(d["confidence"])
        st.caption(f"VAR 信号：`{d['var_signal']}`  |  LLM 信号：`{d['llm_signal']}`")
        if d["var_signal"] == d["llm_signal"]:
            st.success("✓ 两个子模块方向一致，置信度加成")
        else:
            st.warning("⚠ 两个子模块方向分歧，已降低置信度")

    with col_signals:
        fig_bar = go.Figure(go.Bar(
            x=["VAR 计量基线", "LLM 链式草稿"],
            y=[
                0.8 if d["var_signal"] == "hold" else (0.6 if d["var_signal"] == "cut" else 0.9),
                d["confidence"],
            ],
            marker_color=["#1976D2", "#7B1FA2"],
            text=[d["var_signal"], d["llm_signal"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="子模块信号对比",
            yaxis=dict(range=[0, 1.1], title="置信度"),
            height=220, margin=dict(t=40, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── 鹰鸽仪表盘 + 分维度雷达 ──────────────────────────────────
    st.markdown(f"<span class='agent-tag a2-tag'>Agent 2</span> **政策情绪特征向量**", unsafe_allow_html=True)

    col_gauge, col_radar = st.columns(2)

    with col_gauge:
        score = d["hawkish_score"]
        needle_color = "#F44336" if score > 0.3 else ("#2196F3" if score < -0.3 else "#FF9800")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": 0, "valueformat": "+.2f"},
            number={"valueformat": "+.2f", "font": {"size": 36}},
            title={"text": "鹰鸽评分<br><sub>-1.0（极鸽）→ +1.0（极鹰）</sub>"},
            gauge={
                "axis": {"range": [-1, 1], "tickwidth": 1},
                "bar":  {"color": needle_color, "thickness": 0.3},
                "steps": [
                    {"range": [-1.0, -0.3], "color": "#BBDEFB"},
                    {"range": [-0.3,  0.3], "color": "#FFF9C4"},
                    {"range": [ 0.3,  1.0], "color": "#FFCDD2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=60, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        warsh_color = {"hawkish":"🔴", "neutral":"🟡", "dovish":"🔵"}.get(d["warsh_signal"], "⚪")
        st.markdown(f"**沃什信号：{warsh_color} {d['warsh_signal'].upper()}**")

    with col_radar:
        categories = ["通胀担忧", "增长担忧", "就业担忧（估算）", "鹰派程度", "沃什鹰派度"]
        values = [
            d["inflation_concern"],
            d["growth_concern"],
            0.35,                                             # 就业担忧，使用默认估算值
            max(0, d["hawkish_score"]),
            0.75 if d["warsh_signal"] == "hawkish" else 0.4,
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(244,67,54,0.15)",
            line=dict(color="#F44336", width=2),
            name="当前态势",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="政策情绪雷达图",
            height=280,
            margin=dict(t=60, b=10),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ── 历史趋势 ──────────────────────────────────────────────────
    hist_df = load_history()
    if len(hist_df) > 1:
        st.markdown("**历史鹰鸽评分趋势（每次运行）**")
        fig_hist = go.Figure(go.Scatter(
            x=hist_df["run_no"], y=hist_df["hawkish_score"],
            mode="lines+markers+text",
            text=hist_df["hawkish_score"].apply(lambda x: f"{x:+.2f}"),
            textposition="top center",
            line=dict(color="#F44336", width=2),
            marker=dict(size=10, color="#F44336"),
        ))
        fig_hist.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="中性线")
        fig_hist.update_layout(
            xaxis_title="运行次数", yaxis_title="鹰鸽评分",
            yaxis=dict(range=[-1.1, 1.1]),
            height=220, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("历史趋势图将在积累多次分析后显示。")

    # ── Agent 4：主席行为蒸馏 ─────────────────────────────────────
    distill = load_distill()
    if distill:
        st.divider()
        st.markdown(
            "<span class='agent-tag' style='background:#FFF3E0;color:#E65100;'>Agent 4</span>"
            " **五位主席行为蒸馏 — 沃什决策预测**",
            unsafe_allow_html=True,
        )

        col_votes, col_warsh = st.columns([3, 2])

        with col_votes:
            chairs  = ["Greenspan", "Bernanke", "Yellen", "Powell"]
            keys    = ["greenspan_vote", "bernanke_vote", "yellen_vote", "powell_vote"]
            reasons = ["greenspan_reason", "bernanke_reason", "yellen_reason", "powell_reason"]
            vote_colors = {"hold": "#FFC107", "cut": "#2196F3", "hike": "#F44336"}

            for chair, k, rk in zip(chairs, keys, reasons):
                v = distill.get(k, "?")
                color = vote_colors.get(v.lower(), "#9E9E9E")
                reason_text = distill.get(rk, "")
                st.markdown(
                    f"<div style='display:flex;align-items:flex-start;margin:6px 0;'>"
                    f"<span style='min-width:110px;font-weight:600;'>{chair}</span>"
                    f"<span style='background:{color};color:white;padding:2px 10px;"
                    f"border-radius:10px;font-size:0.85rem;margin-right:8px;min-width:50px;text-align:center;'>"
                    f"{v.upper()}</span>"
                    f"<span style='color:#555;font-size:0.88rem;'>{reason_text[:60]}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.caption(f"最相似时期：{distill.get('most_similar_period', '?')}")

        with col_warsh:
            w_pred = distill.get("warsh_prediction", "hold")
            w_conf = distill.get("warsh_confidence", 0.0)
            w_css  = "hold-box" if w_pred == "hold" else ("cut-box" if w_pred == "cut" else "hike-box")
            st.markdown(
                f"<div class='big-prediction {w_css}' style='font-size:1.6rem;padding:16px;'>"
                f"沃什预测<br>{w_pred.upper()}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**蒸馏置信度：{w_conf:.0%}**")
            st.progress(float(w_conf))
            st.caption(distill.get("warsh_reasoning", "")[:120])

    # ── Agent 5：影响者信号 ───────────────────────────────────────
    inf = load_influencer()
    if inf:
        st.divider()
        st.markdown(
            "<span class='agent-tag' style='background:#E8F5E9;color:#1B5E20;'>Agent 5</span>"
            " **影响者联合信号（市场预期 + 政治压力 + 国际联动）**",
            unsafe_allow_html=True,
        )

        c_mkt, c_pol, c_intl = st.columns(3)

        with c_mkt:
            st.markdown("**市场预期（收益率曲线）**")
            spread = inf.get("yield_spread")
            curve  = inf.get("curve_signal", "?")
            curve_color = {"hold": "#FFC107", "cut": "#2196F3", "hike": "#F44336"}.get(curve, "#9E9E9E")
            if spread is not None:
                st.metric("10Y-2Y 利差", f"{spread:.2f}%")
            st.markdown(
                f"<span style='background:{curve_color};color:white;padding:4px 14px;"
                f"border-radius:10px;font-weight:600;'>曲线信号 {curve.upper()}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"DGS2={inf.get('dgs2', '?')}%  DGS10={inf.get('dgs10', '?')}%")

        with c_pol:
            st.markdown("**政治压力**")
            pol_dir = inf.get("political_direction", "?")
            pol_int = inf.get("political_intensity", 0.0)
            ind_risk = inf.get("fed_independence_risk", "?")
            st.markdown(f"方向：**{pol_dir}**")
            st.markdown(f"强度：**{pol_int:.0%}**")
            st.progress(float(pol_int))
            risk_color = "#F44336" if ind_risk == "high" else ("#FF9800" if ind_risk == "medium" else "#4CAF50")
            st.markdown(
                f"<span style='color:{risk_color};font-weight:600;'>独立性风险：{ind_risk.upper()}</span>",
                unsafe_allow_html=True,
            )

        with c_intl:
            st.markdown("**国际联动**")
            ecb = inf.get("ecb_direction", "?")
            usd = inf.get("dollar_trend", "?")
            intl_sig = inf.get("international_signal", "?")
            intl_color = {"hold": "#FFC107", "cut": "#2196F3", "hike": "#F44336"}.get(intl_sig, "#9E9E9E")
            st.markdown(f"ECB 方向：**{ecb}**")
            st.markdown(f"美元趋势：**{usd}**")
            st.markdown(
                f"<span style='background:{intl_color};color:white;padding:4px 14px;"
                f"border-radius:10px;font-weight:600;'>国际信号 {intl_sig.upper()}</span>",
                unsafe_allow_html=True,
            )

        # 综合融合信号
        combined     = inf.get("combined_signal", "hold")
        combined_conf = inf.get("combined_confidence", 0.0)
        combined_css  = "hold-box" if combined == "hold" else ("cut-box" if combined == "cut" else "hike-box")
        st.markdown(
            f"<div class='big-prediction {combined_css}' style='font-size:1.4rem;padding:12px;margin-top:12px;'>"
            f"Agent 5 综合信号：{combined.upper()}  ({combined_conf:.0%})</div>",
            unsafe_allow_html=True,
        )
        if inf.get("synthesis"):
            st.caption(inf["synthesis"][:200])


# ════════════════════════════════════════════════════════════════
# Tab 2：分析详情
# ════════════════════════════════════════════════════════════════
with tab2:

    # 声明原文
    with st.expander(f"📄 FOMC 声明原文（{d['meeting_date']}）", expanded=False):
        st.markdown(f"[查看原始链接]({d['statement_url']})")
        st.text(d["statement_text"])

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        # Agent 2 关键证据
        st.markdown(f"<span class='agent-tag a2-tag'>Agent 2</span> **语义分析关键证据**", unsafe_allow_html=True)
        for i, ev in enumerate(d["key_evidence"], 1):
            st.markdown(f"<div class='evidence-item'>**{i}.** {ev}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Agent 2 结论：** {d['sem_conclusion']}")

        st.divider()

        # Agent 3 证据链
        st.markdown(f"<span class='agent-tag a3-tag'>Agent 3</span> **决策支撑证据链**", unsafe_allow_html=True)
        for ev in d["evidence_chain"]:
            st.markdown(f"<div class='evidence-item'>· {ev}</div>", unsafe_allow_html=True)

    with col_right:
        # Chain-of-Draft 推理链
        st.markdown(f"<span class='agent-tag a3-tag'>Agent 3</span> **Chain-of-Draft 五步推理**", unsafe_allow_html=True)
        step_labels = ["① 通胀形势", "② 就业与增长", "③ 金融条件", "④ 沃什立场", "⑤ 历史对标"]
        for label, step in zip(step_labels, d["cod_steps"]):
            st.markdown(f"<div class='cod-step'><b>{label}</b><br>{step}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 风险因素
        st.markdown("⚠️ **关键风险因素**（可能改变预测的条件）")
        for r in d["risk_factors"]:
            st.markdown(f"<div class='risk-item'>· {r}</div>", unsafe_allow_html=True)

    st.divider()

    # 综合结论
    st.markdown(f"<span class='agent-tag a3-tag'>Agent 3</span> **综合结论**", unsafe_allow_html=True)
    st.info(d["final_conclusion"])

    # 降息/加息触发条件
    st.markdown("**政策转向触发条件**")
    c_cut, c_hike = st.columns(2)
    with c_cut:
        st.markdown("🔵 **降息触发条件（任一满足）**")
        st.markdown("- CPI 同比降至 **3.0% 以下**\n- 失业率升至 **4.5% 以上**\n- 经济衰退信号明显")
    with c_hike:
        st.markdown("🔴 **加息触发条件（任一满足）**")
        st.markdown("- CPI 同比升至 **5.0% 以上**\n- 通胀预期明显脱锚\n- 工资-价格螺旋形成")


# ════════════════════════════════════════════════════════════════
# Tab 3：系统信息
# ════════════════════════════════════════════════════════════════
with tab3:

    st.markdown("### 📊 数据血统链")
    st.markdown("""
    每次运行流水线，结果会完整保存在本地 SQLite 数据库的五张表中，形成可追溯的数据血统链：
    """)
    st.markdown("""
    ```
    meeting_packages → semantic_results → fomc_predictions
                                        → chair_distill_results
                                        → influencer_signals

    meeting_packages              fomc_history（历史回测库）
    ─────────────────             ──────────────────────────
    meeting_date                  meeting_date  (2000–2024)
    statement_text                chair
    cpi_yoy / unemployment_rate   decision  (hold/cut/hike)
    fed_funds_rate / treasury_10y change_bps / target_rate
    data_cutoff_date              cpi_yoy / unemployment_rate
    source (live/fallback)        fed_funds_rate / split
    ```
    """)

    st.divider()

    st.markdown("### 🏗️ 系统架构（FedSight AI 启发）")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**五智能体分工：**

| 智能体 | 输入 | 输出 |
|---|---|---|
| Agent 1 数据感知 | 网络（美联储/FRED） | MeetingPackage |
| Agent 2 语义提取 | MeetingPackage | SemanticFeature |
| Agent 3 决策推演 | SemanticFeature + 宏观 | FOMCPrediction |
| Agent 4 主席蒸馏 | 历史DB + 宏观 | DistillResult |
| Agent 5 影响者 | FRED + LLM | InfluencerResult |

**五路信号加权融合：**
- Agent3 VAR 35% + Agent3 LLM 35%
- Agent4 蒸馏 15% + Agent5 市场 8% + Agent5 综合 7%
        """)
    with col_b:
        st.markdown("""
**核心技术借鉴（FedSight AI, NeurIPS 2025）：**
- **Chain-of-Draft**：每步 ≤30 字的强制多步推理，减少幻觉
- **ICL（情境学习）**：注入历史会议案例增强决策依据
- **防前视偏差**：数据截断日 = 会议日 − 2 天
- **完整血统追踪**：raw_json 字段记录每层中间结果

**为何优于传统计量（VAR/VECM）：**
- **精度**：LLM 能理解"carefully"比"patiently"更鹰的细微语义
- **时效**：数据一发布立即自动触发，无需人工操作
- **可解释性**：CoD 步骤即证据链，可直接用于汇报
        """)

    st.divider()
    st.markdown("### 📁 当前运行状态")
    try:
        conn = sqlite3.connect(DB_PATH)
        mp_count  = conn.execute("SELECT COUNT(*) FROM meeting_packages").fetchone()[0]
        sr_count  = conn.execute("SELECT COUNT(*) FROM semantic_results").fetchone()[0]
        fp_count  = conn.execute("SELECT COUNT(*) FROM fomc_predictions").fetchone()[0]
        try:
            cd_count = conn.execute("SELECT COUNT(*) FROM chair_distill_results").fetchone()[0]
        except Exception:
            cd_count = 0
        try:
            inf_count = conn.execute("SELECT COUNT(*) FROM influencer_signals").fetchone()[0]
        except Exception:
            inf_count = 0
        try:
            hist_count = conn.execute("SELECT COUNT(*) FROM fomc_history").fetchone()[0]
        except Exception:
            hist_count = 0
        conn.close()
        c1, c2, c3 = st.columns(3)
        c1.metric("📦 MeetingPackage 记录数", mp_count)
        c2.metric("🔬 SemanticFeature 记录数", sr_count)
        c3.metric("🎯 FOMCPrediction 记录数", fp_count)
        c4, c5, c6 = st.columns(3)
        c4.metric("🧠 ChairDistill 记录数", cd_count)
        c5.metric("🌐 InfluencerSignal 记录数", inf_count)
        c6.metric("📚 历史FOMC记录（回测库）", hist_count)
    except Exception:
        st.warning("数据库读取失败")
