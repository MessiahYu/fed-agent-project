"""
第二步 ── 主席行为蒸馏智能体 (Chair Distillation Agent)
=========================================================
核心思路：
  1. 从 fomc_history 数据库自动统计 4 位历史主席的量化决策画像
     （加息时平均 CPI 多高？降息时失业率多少？）
  2. 结合硬编码的"定性画像"（哲学风格、核心理念）
  3. 让 DeepSeek LLM 以四位主席的视角分别投票
  4. 分析沃什最接近哪位主席的哪个历史时期
  5. 输出"蒸馏后的沃什预测"

运行方式：
  E:\\Anaconda\\envs\\fed-agent\\python.exe agent4_chair_distill.py
"""

import os, re, json, sqlite3
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from utils.db import DB_PATH
from utils.config import NEXT_FOMC, LLM_MODEL, LLM_BASE_URL

load_dotenv()
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
NEXT_MEETING = NEXT_FOMC


# ══════════════════════════════════════════════════════════════════════
# Pydantic 结构化输出
# ══════════════════════════════════════════════════════════════════════

class DistillResult(BaseModel):
    greenspan_vote:   str    # hold / cut / hike
    greenspan_reason: str
    bernanke_vote:    str
    bernanke_reason:  str
    yellen_vote:      str
    yellen_reason:    str
    powell_vote:      str
    powell_reason:    str
    most_similar_period: str  # 沃什最接近哪位主席的哪个历史时期
    warsh_prediction: str
    warsh_confidence: float
    warsh_reasoning:  str
    final_signal:     str


# ══════════════════════════════════════════════════════════════════════
# 五位主席的"定性画像"（质化知识，硬编码）
# ══════════════════════════════════════════════════════════════════════

CHAIR_PERSONAS = {
    "Greenspan": {
        "era": "2000-2006（互联网泡沫→9/11→复苏→2004年开始17连加）",
        "style": "建设性模糊（Constructive Ambiguity）",
        "core": [
            "风险管理流派：衰退威胁→大幅降息，通胀威胁→缓慢加息",
            "刻意不透明，让市场自行猜测，拒绝明确前瞻性指引",
            "CPI>5% 才开始真正警觉；就业有松动迹象就会先发制人降息",
            "不对称偏好：避免衰退的优先级 > 避免高通胀",
        ],
    },
    "Bernanke": {
        "era": "2006-2014（次贷危机→金融海啸→QE1/2/3→零利率7年）",
        "style": "学术派，透明化先驱，QE创始人",
        "core": [
            "通胀目标制（2%）和前瞻性指引（Forward Guidance）的制度化推手",
            "金融稳定 > 短期通胀，优先使用数量工具（QE/资产购买）而非价格工具",
            "'直升机本'：绝不允许通缩和大萧条重演，必要时直升机撒钱",
            "在零利率下限（ZLB）依赖 QE 而非进一步降息",
        ],
    },
    "Yellen": {
        "era": "2014-2018（零利率退出→极度渐进加息，4年只加4次）",
        "style": "劳工市场优先，极度渐进主义",
        "core": [
            "双重使命中，就业优先于通胀——宁可过热也不要失业率走高",
            "加息决策异常谨慎：必须在极度确定时才动；通胀略高也愿意等",
            "2015-2018 只加息 4 次，每次都被市场认为太慢了还是太快",
            "数据驱动但偏鸽：寻找推迟加息的理由，而非推迟降息",
        ],
    },
    "Powell": {
        "era": "2018-2026（贸易战→COVID→40年最高通胀→2022年最快加息→降息）",
        "style": "数据依赖，沟通透明，反应滞后后快速修正",
        "core": [
            "标志语：'我们将是 data-dependent'——跟着数据走，不预判",
            "2021年'通胀暂时论'失误后，2022年快速转向为激进鹰派",
            "两种模式：宽松期（数据恶化就降）；紧缩期（CPI降至目标才停）",
            "触发降息：CPI持续降至~3%+就业出现松动+实际利率明显偏高",
        ],
    },
    "Warsh": {
        "era": "2026-今（接替鲍威尔，首次 FOMC 2026-06-17 维持 3.5-3.75% 不变）",
        "style": "规则优先的鹰派传统主义者",
        "core": [
            "批评 QE 等非常规工具，强调规则制度而非相机抉择（Discretion）",
            "通胀纪律是美联储公信力的基石；宁可提前展示决心",
            "主张精简沟通——减少前瞻性指引，让市场依赖数据而非美联储承诺",
            "矛盾点：公开表示有降息空间，但首次 FOMC 措辞偏硬，实际维持不变",
            "待解谜题：当前 CPI 4.17%（中东能源冲击所致），他是否容忍 or 反应？",
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════
# 从数据库自动生成量化画像
# ══════════════════════════════════════════════════════════════════════

def load_chair_stats() -> dict:
    """
    从 fomc_history 表统计每位主席的量化决策画像。
    返回格式：{ "Greenspan": { "total": 60, "hold_pct": 0.4, ... }, ... }
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT chair, actual_decision,
               COUNT(*) as cnt,
               ROUND(AVG(cpi_yoy), 2)     as avg_cpi,
               ROUND(AVG(unemployment), 2) as avg_ur,
               ROUND(AVG(fed_funds_rate), 2) as avg_ff
        FROM fomc_history
        GROUP BY chair, actual_decision
    """, conn)

    totals = pd.read_sql("""
        SELECT chair, COUNT(*) as total FROM fomc_history GROUP BY chair
    """, conn)
    conn.close()

    stats = {}
    for _, tot in totals.iterrows():
        ch = tot["chair"]
        total = int(tot["total"])
        subset = df[df["chair"] == ch]

        def _get(decision):
            row = subset[subset["actual_decision"] == decision]
            if row.empty:
                return {"cnt": 0, "avg_cpi": None, "avg_ur": None}
            r = row.iloc[0]
            return {
                "cnt": int(r["cnt"]),
                "avg_cpi": r["avg_cpi"],
                "avg_ur":  r["avg_ur"],
                "avg_ff":  r["avg_ff"],
            }

        h = _get("hold")
        c = _get("cut")
        k = _get("hike")

        stats[ch] = {
            "total":      total,
            "hold_pct":   round(h["cnt"] / total, 2),
            "cut_pct":    round(c["cnt"] / total, 2),
            "hike_pct":   round(k["cnt"] / total, 2),
            "hike_avg_cpi":  k["avg_cpi"],
            "cut_avg_cpi":   c["avg_cpi"],
            "hold_avg_cpi":  h["avg_cpi"],
            "hike_avg_ur":   k["avg_ur"],
            "cut_avg_ur":    c["avg_ur"],
        }

    return stats


# ══════════════════════════════════════════════════════════════════════
# 规则信号（用于快速回测对比，无需 LLM 调用）
# ══════════════════════════════════════════════════════════════════════

def rule_based_chair_signal(chair: str, cpi: float, unrate: float,
                             fedfunds: float) -> tuple[str, float]:
    """
    基于各主席量化画像的简单规则信号，不调用 LLM。
    返回 (signal, confidence)。

    设计逻辑：
      - 当 VAR 看到的是"过去趋势"，规则信号看的是"当前宏观与主席阈值的距离"
      - 两者结合可以捕捉到 VAR 容易错过的"政策转向节点"
    """
    if cpi is None or fedfunds is None:
        return "hold", 0.40

    if chair == "Powell":
        # ZLB 时期（COVID 零利率）：等待 tapering 完成，不会轻易动
        if fedfunds < 0.5:
            return "hold", 0.65

        # 主动紧缩区间：CPI 明显偏高 + 实际利率尚负
        if cpi > 4.5 and fedfunds < (cpi - 1.5):
            return "hike", 0.72

        # 降息触发：CPI 接近目标 + 就业松动 + 利率明显限制性
        if cpi < 2.8 and unrate is not None and unrate > 4.2 and fedfunds > 4.0:
            return "cut", 0.68

        return "hold", 0.55

    elif chair == "Yellen":
        # 耶伦极度谨慎，只有明确充分就业+通胀稳定才加息
        if cpi > 2.5 and unrate is not None and unrate < 4.5 and fedfunds < 1.5:
            return "hike", 0.55
        if cpi < 1.5 or (unrate is not None and unrate > 5.5):
            return "cut", 0.50
        return "hold", 0.60

    elif chair == "Bernanke":
        # 伯南克：金融危机时大幅降息，零利率时坚守
        if fedfunds < 1.0 and cpi < 3.0:
            return "hold", 0.70  # ZLB + 温和通胀 → QE 而非加息
        if cpi > 5.0 and fedfunds < 3.0:
            return "hike", 0.55
        return "hold", 0.55

    elif chair == "Greenspan":
        # 格林斯潘：就业任何松动就降息；CPI>5% 才真正加息
        if unrate is not None and unrate > 5.5 and cpi < 3.5:
            return "cut", 0.55
        if cpi > 5.0 and fedfunds < 3.5:
            return "hike", 0.55
        return "hold", 0.50

    else:
        # Warsh（偏鹰）：CPI > 3% 就不敢降，低于 2.5% + 就业转弱才降
        if cpi > 3.5:
            return "hold", 0.70
        if cpi < 2.5 and unrate is not None and unrate > 4.5:
            return "cut", 0.60
        return "hold", 0.65


# ══════════════════════════════════════════════════════════════════════
# 构建 LLM 提示词
# ══════════════════════════════════════════════════════════════════════

def build_distill_prompt(macro_context: dict, stats: dict) -> str:
    """
    将量化统计 + 定性画像 + 当前宏观融合成一个完整提示词。
    关键设计：每位主席画像 = 量化统计（来自真实历史数据）+ 定性哲学（硬编码知识）
    """

    # 当前宏观摘要
    macro_text = f"""
当前宏观环境（截至 {NEXT_MEETING} 前）：
  · CPI 同比：{macro_context.get('cpi_yoy', '?')}%（目标 2%，中东能源冲击推高）
  · 失业率：{macro_context.get('unemployment_rate', '?')}%
  · 联邦基金利率：{macro_context.get('fed_funds_rate', '?')}%（区间 3.5-3.75%）
  · 10Y 国债收益率：{macro_context.get('treasury_10y', '?')}%
"""

    # 为每位历史主席生成画像文字块
    chair_blocks = []
    for name, persona in CHAIR_PERSONAS.items():
        st = stats.get(name, {})
        q_lines = []
        if st:
            q_lines += [
                f"  量化统计（基于 {st.get('total', '?')} 次历史会议）：",
                f"    HOLD {st.get('hold_pct', 0):.0%} | CUT {st.get('cut_pct', 0):.0%} | HIKE {st.get('hike_pct', 0):.0%}",
            ]
            if st.get("hike_avg_cpi"):
                q_lines.append(f"    加息时平均 CPI = {st['hike_avg_cpi']}%，"
                                f"降息时平均 CPI = {st.get('cut_avg_cpi', '?')}%")
            if st.get("hike_avg_ur"):
                q_lines.append(f"    加息时平均失业率 = {st['hike_avg_ur']}%，"
                                f"降息时平均失业率 = {st.get('cut_avg_ur', '?')}%")

        core_lines = "\n".join(f"    · {c}" for c in persona["core"])
        block = f"""
【{name}（{persona['era']}）】
  风格：{persona['style']}
{chr(10).join(q_lines)}
  决策理念：
{core_lines}"""
        chair_blocks.append(block)

    chairs_text = "\n".join(chair_blocks)

    prompt = f"""
你是顶级美联储货币政策研究员，专长于多维度比较分析。

{macro_text}

下面是五位美联储主席的完整画像（量化统计 + 定性哲学）：
{chairs_text}

请完成三项分析：

**任务1：历史主席投票**
面对上述当前宏观环境，每位历史主席（Greenspan / Bernanke / Yellen / Powell）会怎么决定？
给出 hold/cut/hike + 一句关键理由（≤30字）。

**任务2：沃什画像匹配**
沃什的已知立场 + 2026-06-17 首次 FOMC 行为，最接近哪位历史主席的哪个具体历史时期？
（例如"格林斯潘 2005 年：通胀略高+渐进加息+就业稳定期"）

**任务3：沃什预测**
综合"沃什自身立场 + 最相似历史案例 + 当前宏观"，预测下次会议（{NEXT_MEETING}）。
注意：CPI 4.17% 远超目标，但主要由外部冲击（中东能源）驱动，核心通胀可能较低。

严格按如下 JSON 格式输出，不要有任何其他文字：
{{
  "greenspan_vote": "hold",
  "greenspan_reason": "...",
  "bernanke_vote": "hold",
  "bernanke_reason": "...",
  "yellen_vote": "hold",
  "yellen_reason": "...",
  "powell_vote": "hold",
  "powell_reason": "...",
  "most_similar_period": "...",
  "warsh_prediction": "hold",
  "warsh_confidence": 0.80,
  "warsh_reasoning": "...",
  "final_signal": "hold"
}}
"""
    return prompt


# ══════════════════════════════════════════════════════════════════════
# LLM 调用（CrewAI）
# ══════════════════════════════════════════════════════════════════════

def run_llm_distillation(macro_context: dict, stats: dict) -> DistillResult:
    """用 DeepSeek 做主席蒸馏推理，返回结构化结果"""
    llm = LLM(
        model="deepseek/deepseek-chat",
        api_key=DEEPSEEK_KEY,
        base_url="https://api.deepseek.com",
    )

    agent = Agent(
        role="美联储主席行为蒸馏分析师",
        goal="从历史主席决策模式中蒸馏沃什的政策偏好，给出当前会议预测",
        backstory=(
            "你精通美联储 25 年货币政策史，掌握每位主席的决策风格与历史数据。"
            "你的特长是跨主席比较——判断当前主席的行为更像哪个历史时期，"
            "再用这种类比做出更准确的政策预测。"
        ),
        llm=llm,
        verbose=False,
    )

    prompt = build_distill_prompt(macro_context, stats)

    task = Task(
        description=prompt,
        expected_output="严格的 JSON 对象，包含 12 个字段",
        agent=agent,
    )

    crew   = Crew(agents=[agent], tasks=[task], verbose=False)
    result = crew.kickoff()
    raw    = str(result)

    # JSON 提取（与其他 Agent 一致的方式）
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"LLM 未返回有效 JSON：{raw[:300]}")

    data = json.loads(match.group())
    return DistillResult.model_validate(data)


# ══════════════════════════════════════════════════════════════════════
# 数据库保存
# ══════════════════════════════════════════════════════════════════════

def save_distill_to_db(result: DistillResult, meeting_date: str) -> int:
    """将蒸馏结果存入 chair_distill_results 表"""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chair_distill_results (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_date   TEXT,
            greenspan_vote TEXT,
            bernanke_vote  TEXT,
            yellen_vote    TEXT,
            powell_vote    TEXT,
            most_similar   TEXT,
            warsh_predict  TEXT,
            warsh_conf     REAL,
            warsh_reason   TEXT,
            final_signal   TEXT,
            raw_json       TEXT,
            created_at     TEXT DEFAULT (datetime('now'))
        )
    """)
    raw_json = result.model_dump_json(ensure_ascii=False)
    cur.execute("""
        INSERT INTO chair_distill_results
          (meeting_date, greenspan_vote, bernanke_vote, yellen_vote, powell_vote,
           most_similar, warsh_predict, warsh_conf, warsh_reason, final_signal, raw_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (meeting_date, result.greenspan_vote, result.bernanke_vote,
          result.yellen_vote, result.powell_vote, result.most_similar_period,
          result.warsh_prediction, result.warsh_confidence,
          result.warsh_reasoning, result.final_signal, raw_json))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


# ══════════════════════════════════════════════════════════════════════
# 主流程：蒸馏当前预测
# ══════════════════════════════════════════════════════════════════════

def run_chair_distill(macro_context: dict, meeting_date: str = NEXT_MEETING) -> DistillResult:
    """对外接口：接收宏观数据，返回蒸馏结果"""
    stats  = load_chair_stats()
    result = run_llm_distillation(macro_context, stats)
    save_distill_to_db(result, meeting_date)
    return result


# ══════════════════════════════════════════════════════════════════════
# 报告打印
# ══════════════════════════════════════════════════════════════════════

def print_distill_report(result: DistillResult, stats: dict):
    W = 62
    print("\n" + "★" * W)
    print("  FedWatch AI — 主席行为蒸馏报告")
    print("  预测会议：" + NEXT_MEETING)
    print("★" * W)

    # 量化统计摘要
    print("\n  各主席历史决策统计（来自真实 FOMC 数据库）：")
    print(f"  {'主席':12}  {'总会议':>5}  {'HOLD%':>6}  {'HIKE%':>6}  {'CUT%':>6}  加息时CPI")
    print("  " + "-" * 55)
    for name in ["Greenspan", "Bernanke", "Yellen", "Powell"]:
        st = stats.get(name, {})
        print(f"  {name:12}  {st.get('total', '?'):>5}  "
              f"{st.get('hold_pct', 0):>5.0%}  "
              f"{st.get('hike_pct', 0):>5.0%}  "
              f"{st.get('cut_pct', 0):>5.0%}  "
              f"{st.get('hike_avg_cpi', '--') or '--'}%")

    # 四位主席投票
    print("\n  面对当前宏观（CPI 4.17%，失业 4.3%，利率 3.63%），各主席投票：")
    votes = [
        ("Greenspan", result.greenspan_vote, result.greenspan_reason),
        ("Bernanke",  result.bernanke_vote,  result.bernanke_reason),
        ("Yellen",    result.yellen_vote,    result.yellen_reason),
        ("Powell",    result.powell_vote,    result.powell_reason),
    ]
    tally = {"hold": 0, "cut": 0, "hike": 0}
    for name, vote, reason in votes:
        symbol = {"hold": "—", "cut": "▼", "hike": "▲"}.get(vote, "?")
        tally[vote] = tally.get(vote, 0) + 1
        print(f"\n  [{symbol}] {name}: {vote.upper()}")
        print(f"      理由: {reason}")

    print(f"\n  票数汇总 — HOLD:{tally['hold']}  HIKE:{tally['hike']}  CUT:{tally['cut']}")

    # 沃什蒸馏结论
    print(f"""
  ─────────────────────────────────────────────────
  最相似历史时期：{result.most_similar_period}

  沃什预测（{NEXT_MEETING}）：
    决策方向   ► {result.warsh_prediction.upper()} ◄
    蒸馏置信度  {result.warsh_confidence:.0%}
    推理依据：
      {result.warsh_reasoning}

  最终信号：{result.final_signal.upper()}
  ─────────────────────────────────────────────────""")
    print("★" * W)


# ══════════════════════════════════════════════════════════════════════
# 独立运行（从数据库读最新宏观数据）
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n主席行为蒸馏智能体 启动...")

    # 从数据库读最新 meeting_packages 的宏观数据
    try:
        conn = sqlite3.connect(DB_PATH)
        row  = pd.read_sql(
            "SELECT raw_json FROM meeting_packages ORDER BY id DESC LIMIT 1",
            conn
        ).iloc[0]["raw_json"]
        conn.close()
        pkg  = json.loads(row)
        macro = {
            "cpi_yoy":           pkg.get("macro", {}).get("cpi_yoy"),
            "unemployment_rate": pkg.get("macro", {}).get("unemployment_rate"),
            "fed_funds_rate":    pkg.get("macro", {}).get("fed_funds_rate"),
            "treasury_10y":      pkg.get("macro", {}).get("treasury_10y"),
        }
        print(f"  已读取宏观数据：CPI={macro['cpi_yoy']}%  "
              f"失业率={macro['unemployment_rate']}%  "
              f"利率={macro['fed_funds_rate']}%")
    except Exception:
        # 降级：用上次已知真实数据
        macro = {
            "cpi_yoy": 4.17, "unemployment_rate": 4.3,
            "fed_funds_rate": 3.63, "treasury_10y": 4.49,
        }
        print(f"  [降级] 使用缺省宏观数据：CPI={macro['cpi_yoy']}%")

    print("\n[1/2] 加载历史主席量化统计...")
    stats = load_chair_stats()
    print("  统计完成：", {k: v["total"] for k, v in stats.items()})

    print("\n[2/2] 调用 DeepSeek LLM 做主席蒸馏推理（约需 30 秒）...")
    result = run_chair_distill(macro)

    print_distill_report(result, stats)
