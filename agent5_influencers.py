"""
第三步 ── 影响者联合 Agent 群 (Influencer Agents)
=========================================================
背景：美联储名义上独立，但现实中受三类外部力量影响：

  A. 市场预期 Agent  ── 国债收益率曲线斜率（纯 FRED 定量，不调 LLM）
  B. 政治压力 Agent  ── 白宫/国会施压方向（LLM 定性分析）
  C. 国际联动 Agent  ── ECB + 美元 + 全球增长（FRED 数据 + LLM 分析）

设计原则：
  · 市场信号用纯数据：收益率曲线是市场对未来利率的集体预测，客观可信
  · 政治+国际用 LLM：这两类因素高度依赖措辞和背景理解
  · 只发起一次 LLM 调用（B+C合并），降低 API 成本

运行方式：
  E:\\Anaconda\\envs\\fed-agent\\python.exe agent5_influencers.py
"""

import os, re, json, sqlite3
from typing import Optional

import pandas as pd
import requests
from pydantic import BaseModel
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
FRED_KEY     = os.getenv("FRED_API_KEY")
DB_PATH      = r"c:\Users\余青锋\OneDrive\fed-agent-project\fed_watch.db"
NEXT_MEETING = "2026-07-28/29"


# ══════════════════════════════════════════════════════════════════════
# Pydantic 输出模型
# ══════════════════════════════════════════════════════════════════════

class InfluencerResult(BaseModel):
    # ── A. 市场预期（定量）──
    dgs2:             Optional[float]  # 2Y 国债收益率
    dgs10:            Optional[float]  # 10Y 国债收益率
    yield_spread:     Optional[float]  # 10Y - 2Y（正=正常，负=倒挂）
    ecb_rate:         Optional[float]  # ECB 利率
    dollar_index:     Optional[float]  # 美元指数
    curve_signal:     str              # cut / hold / hike
    curve_confidence: float
    curve_reason:     str

    # ── B. 政治压力（LLM）──
    political_direction:  str    # toward_cut / neutral / toward_hike
    political_intensity:  float  # 0–1，0=无压力，1=极端压力
    political_context:    str
    fed_independence_risk: str   # low / medium / high

    # ── C. 国际联动（FRED + LLM）──
    ecb_direction:       str    # cutting / holding / hiking
    dollar_trend:        str    # strengthening / stable / weakening
    global_growth_risk:  str    # low / medium / high
    international_signal: str   # cut / hold / hike
    international_context: str

    # ── 三路融合 ──
    combined_signal:     str
    combined_confidence: float
    synthesis:           str


# ══════════════════════════════════════════════════════════════════════
# A. 市场预期 Agent（纯 FRED 数据，无 LLM）
# ══════════════════════════════════════════════════════════════════════

def _fred_latest(series_id: str) -> Optional[float]:
    """拉取 FRED 某指标的最新值"""
    if not FRED_KEY:
        return None
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        r = requests.get(url, params={
            "series_id": series_id, "api_key": FRED_KEY,
            "file_type": "json", "sort_order": "desc",
            "limit": 5, "observation_start": "2026-01-01",
        }, timeout=15)
        obs = [o for o in r.json().get("observations", [])
               if o["value"] not in (".", "")]
        return round(float(obs[0]["value"]), 4) if obs else None
    except Exception:
        return None


def run_market_agent() -> dict:
    """
    市场预期 Agent：
    读取收益率曲线斜率（10Y - 2Y）解读市场对未来利率的隐含预测。

    经济学逻辑：
      · 曲线倒挂（10Y < 2Y）→ 市场预期短期高利率不可持续 → 信号 CUT
      · 曲线持平（差 < 0.5%）→ 无方向信号 → HOLD
      · 曲线正斜率（> 0.5%）→ 经济强劲，市场预期利率稳定 → HOLD
      · 曲线陡峭（> 1.5%）→ 通胀预期升温 → 偏向 HIKE
    """
    print("  [A] 拉取市场数据（FRED: DGS2 / DGS10 / ECBDFR / DTWEXBGS）...")
    dgs2     = _fred_latest("DGS2")
    dgs10    = _fred_latest("DGS10")
    ecb_rate = _fred_latest("ECBDFR")
    dollar   = _fred_latest("DTWEXBGS")

    spread = round(dgs10 - dgs2, 4) if (dgs10 and dgs2) else None

    # 收益率曲线信号
    if spread is None:
        curve_signal = "hold"; curve_conf = 0.40
        curve_reason = "无法获取曲线数据，默认 HOLD"
    elif spread < -0.20:
        curve_signal = "cut"; curve_conf = 0.70
        curve_reason = f"曲线倒挂 {spread:+.2f}%，市场强烈预期降息"
    elif spread < 0.30:
        curve_signal = "hold"; curve_conf = 0.60
        curve_reason = f"曲线接近持平 {spread:+.2f}%，市场无明确方向"
    elif spread < 1.00:
        curve_signal = "hold"; curve_conf = 0.65
        curve_reason = f"曲线轻微正斜率 {spread:+.2f}%，经济正常，维持"
    else:
        curve_signal = "hike"; curve_conf = 0.55
        curve_reason = f"曲线陡峭 {spread:+.2f}%，通胀预期上行"

    print(f"     DGS2={dgs2}%  DGS10={dgs10}%  利差={spread}%  "
          f"ECB={ecb_rate}%  美元指数={dollar}")
    print(f"     市场信号: {curve_signal.upper()}  置信度: {curve_conf:.0%}")

    return {
        "dgs2": dgs2, "dgs10": dgs10, "yield_spread": spread,
        "ecb_rate": ecb_rate, "dollar_index": dollar,
        "curve_signal": curve_signal, "curve_confidence": curve_conf,
        "curve_reason": curve_reason,
    }


# ══════════════════════════════════════════════════════════════════════
# B + C. 政治压力 + 国际联动 Agent（合并为一次 LLM 调用）
# ══════════════════════════════════════════════════════════════════════

# 已知背景知识（截至 2026-06-20，硬编码避免 API 实时抓取）
POLITICAL_CONTEXT_2026 = """
2026年美国政治背景（截至分析日）：
  · 特朗普政府于2025-01上任，持续公开呼吁美联储降息，认为高利率不利于经济增长
  · 特朗普曾多次在社交媒体批评鲍威尔"太慢"，但拜登时期任命的鲍威尔顶住了压力
  · 2026-05-22 沃什接任后，特朗普表示满意（沃什是他提名的），但期待宽松政策
  · 参议院银行委员会：共和党主席倾向去监管+支持增长；民主党关注就业保障
  · 国会预算赤字扩大（减税+支出），存在通过财政政策代替货币宽松的可能
  · 美联储独立性挑战：白宫法律团队2025年曾研究是否可以解雇联储主席的可能性
"""

INTERNATIONAL_CONTEXT_2026 = """
2026年国际经济背景（截至分析日）：
  · ECB（欧洲央行）：2025年已多次降息，欧洲通胀接近目标，目前处于宽松周期中
  · 中国：经济增速放缓，通缩压力持续，人民币相对稳定但出口压力加大
  · 中东局势：2025-2026能源供应紧张（冲突影响），是推高美国CPI至4.17%的主因
  · 美元：偏强（受高利率和避险需求支撑），美元走强对新兴市场形成外部压力
  · 日本：终于走出通缩，日银正常化利率，日元升值中
  · G7/IMF：警告各国不应因政治压力偏离价格稳定目标
  · 贸易：特朗普关税引发全球贸易摩擦，增加了通胀的不确定性
"""


def build_political_international_prompt(macro: dict, market_data: dict) -> str:
    return f"""
你是美联储外部环境分析专家，专长政治经济学和国际货币政策协调。

【当前宏观指标】
  CPI 同比: {macro.get('cpi_yoy', '?')}%
  失业率: {macro.get('unemployment_rate', '?')}%
  联邦基金利率: {macro.get('fed_funds_rate', '?')}%
  10Y 国债: {market_data.get('dgs10', '?')}%
  2Y 国债: {market_data.get('dgs2', '?')}%
  ECB 利率: {market_data.get('ecb_rate', '?')}%
  美元指数: {market_data.get('dollar_index', '?')}

【美国政治背景】
{POLITICAL_CONTEXT_2026}

【国际经济背景】
{INTERNATIONAL_CONTEXT_2026}

请分析两个维度，为下次 FOMC 会议（{NEXT_MEETING}）提供判断：

**B. 政治压力分析**
  1. 白宫对沃什降息的压力有多大？（0-1打分，1=极度施压）
  2. 沃什会在多大程度上屈服？（基于他的"独立鹰派"公开立场）
  3. 政治压力的最终方向：偏向降息(toward_cut) / 中性(neutral) / 偏向加息(toward_hike)
  4. 美联储独立性风险评级：low / medium / high

**C. 国际联动分析**
  1. ECB 已降息 vs 美联储维持高位：汇率和资本流动有何影响？
  2. 能源冲击导致的 CPI 上升：美联储是否应跟随还是穿透式(look-through)处理？
  3. 全球增长风险：low / medium / high
  4. 美元走势：strengthening / stable / weakening
  5. ECB方向：cutting / holding / hiking
  6. 综合国际信号：cut / hold / hike（沃什应该怎么做来协调国际压力）

**综合融合**
综合市场信号({market_data.get('curve_signal', 'hold')})、政治压力、国际联动，给出最终预测：
  - combined_signal: cut / hold / hike
  - combined_confidence: 0-1（置信度）
  - synthesis: ≤80字的综合推理

严格按如下 JSON 格式输出，不要有其他文字：
{{
  "political_direction": "neutral",
  "political_intensity": 0.5,
  "political_context": "...",
  "fed_independence_risk": "medium",
  "ecb_direction": "cutting",
  "dollar_trend": "stable",
  "global_growth_risk": "medium",
  "international_signal": "hold",
  "international_context": "...",
  "combined_signal": "hold",
  "combined_confidence": 0.80,
  "synthesis": "..."
}}
"""


def run_political_international_agent(macro: dict, market_data: dict) -> dict:
    """B+C 合并 LLM 调用"""
    llm = LLM(
        model="deepseek/deepseek-chat",
        api_key=DEEPSEEK_KEY,
        base_url="https://api.deepseek.com",
    )
    agent = Agent(
        role="美联储外部环境分析师",
        goal="从政治压力和国际联动两个维度评估对下次 FOMC 决策的影响",
        backstory=(
            "你是专攻政治经济学和国际货币政策协调的研究员。"
            "你深知美联储名义独立，但现实中受到白宫、国会、全球央行政策的影响。"
            "你能精确评估这些非经济因素对货币政策决策的实际影响力度。"
        ),
        llm=llm, verbose=False,
    )
    task = Task(
        description=build_political_international_prompt(macro, market_data),
        expected_output="严格的 JSON 对象，包含 11 个字段",
        agent=agent,
    )
    crew   = Crew(agents=[agent], tasks=[task], verbose=False)
    raw    = str(crew.kickoff())
    match  = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"LLM 未返回有效 JSON：{raw[:300]}")
    return json.loads(match.group())


# ══════════════════════════════════════════════════════════════════════
# 三路信号融合逻辑
# ══════════════════════════════════════════════════════════════════════

def fuse_three_signals(curve: str, curve_conf: float,
                        political: str, pol_intensity: float,
                        intl: str,
                        combined_llm: str, combined_conf: float) -> tuple[str, float]:
    """
    三路加权投票：
      市场信号 权重 = 0.35（最客观，基于真实资金流动）
      政治信号 权重 = 0.20（有影响但美联储倾向抵制）
      国际信号 权重 = 0.25（外部约束，不可忽视）
      LLM综合  权重 = 0.20（LLM已综合考虑了部分因素）
    """
    votes = {
        "cut":  0.0,
        "hold": 0.0,
        "hike": 0.0,
    }
    votes[curve]       += 0.35 * curve_conf
    votes[political]   += 0.20 * pol_intensity if political != "neutral" else 0
    votes["hold"]      += 0.20 * (1 - pol_intensity)   # 中性政治压力投HOLD
    votes[intl]        += 0.25
    votes[combined_llm]+= 0.20 * combined_conf

    winner = max(votes, key=votes.get)
    total  = sum(votes.values())
    conf   = round(votes[winner] / total, 2) if total > 0 else 0.50
    conf   = max(0.40, min(conf, 0.92))
    return winner, conf


# ══════════════════════════════════════════════════════════════════════
# 数据库保存
# ══════════════════════════════════════════════════════════════════════

def save_influencer_to_db(result: InfluencerResult, meeting_date: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS influencer_signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_date    TEXT,
            curve_signal    TEXT,
            political_dir   TEXT,
            international   TEXT,
            combined_signal TEXT,
            combined_conf   REAL,
            raw_json        TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)
    cur.execute("""
        INSERT INTO influencer_signals
          (meeting_date, curve_signal, political_dir, international,
           combined_signal, combined_conf, raw_json)
        VALUES (?,?,?,?,?,?,?)
    """, (meeting_date, result.curve_signal, result.political_direction,
          result.international_signal, result.combined_signal,
          result.combined_confidence, result.model_dump_json(ensure_ascii=False)))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def run_influencer_agents(macro_context: dict,
                          meeting_date: str = NEXT_MEETING) -> InfluencerResult:
    """
    对外接口：接收宏观数据 → 三路影响者分析 → InfluencerResult
    """
    # ── A. 市场预期（定量，FRED）──────────────────────────────────
    print("\n[A] 市场预期 Agent...")
    market = run_market_agent()

    # ── B+C. 政治+国际（LLM）────────────────────────────────────
    print("\n[B+C] 政治压力 + 国际联动 Agent（LLM，约需 30 秒）...")
    llm_result = run_political_international_agent(macro_context, market)
    print(f"  政治压力: {llm_result.get('political_direction')}  "
          f"强度: {llm_result.get('political_intensity', 0):.0%}")
    print(f"  国际信号: {llm_result.get('international_signal')}  "
          f"ECB方向: {llm_result.get('ecb_direction')}")

    # ── 三路融合 ──────────────────────────────────────────────────
    pol_dir = llm_result.get("political_direction", "neutral")
    pol_int = float(llm_result.get("political_intensity", 0.3))
    # 把 toward_cut / toward_hike 映射为 cut / hike 供投票使用
    pol_vote = {"toward_cut": "cut", "toward_hike": "hike",
                "neutral": "hold"}.get(pol_dir, "hold")

    final_signal, final_conf = fuse_three_signals(
        curve        = market["curve_signal"],
        curve_conf   = market["curve_confidence"],
        political    = pol_vote,
        pol_intensity= pol_int,
        intl         = llm_result.get("international_signal", "hold"),
        combined_llm = llm_result.get("combined_signal", "hold"),
        combined_conf= float(llm_result.get("combined_confidence", 0.6)),
    )

    result = InfluencerResult(
        # 市场数据
        dgs2             = market["dgs2"],
        dgs10            = market["dgs10"],
        yield_spread     = market["yield_spread"],
        ecb_rate         = market.get("ecb_rate"),
        dollar_index     = market.get("dollar_index"),
        curve_signal     = market["curve_signal"],
        curve_confidence = market["curve_confidence"],
        curve_reason     = market["curve_reason"],
        # 政治
        political_direction   = pol_dir,
        political_intensity   = pol_int,
        political_context     = llm_result.get("political_context", ""),
        fed_independence_risk = llm_result.get("fed_independence_risk", "medium"),
        # 国际
        ecb_direction        = llm_result.get("ecb_direction", "cutting"),
        dollar_trend         = llm_result.get("dollar_trend", "stable"),
        global_growth_risk   = llm_result.get("global_growth_risk", "medium"),
        international_signal = llm_result.get("international_signal", "hold"),
        international_context= llm_result.get("international_context", ""),
        # 融合
        combined_signal     = final_signal,
        combined_confidence = final_conf,
        synthesis           = llm_result.get("synthesis", ""),
    )

    save_influencer_to_db(result, meeting_date)
    return result


# ══════════════════════════════════════════════════════════════════════
# 报告打印
# ══════════════════════════════════════════════════════════════════════

def print_influencer_report(result: InfluencerResult):
    W = 62
    SIGNAL_ICON = {"hold": "—", "cut": "▼", "hike": "▲"}

    print("\n" + "★" * W)
    print("  FedWatch AI — 影响者信号报告")
    print(f"  预测会议：{NEXT_MEETING}")
    print("★" * W)

    # A. 市场预期
    print(f"""
  ┌─ A. 市场预期（国债收益率曲线）
  │  2Y 国债:    {result.dgs2 or '?'}%
  │  10Y 国债:   {result.dgs10 or '?'}%
  │  利差(10Y-2Y): {result.yield_spread or '?'}%
  │  ECB 利率:   {result.ecb_rate or '?'}%
  │  美元指数:    {result.dollar_index or '?'}
  │
  │  信号: [{SIGNAL_ICON.get(result.curve_signal,'?')}] {result.curve_signal.upper()}
  │  置信: {result.curve_confidence:.0%}
  │  理由: {result.curve_reason}""")

    # B. 政治压力
    pol_bar = "█" * int(result.political_intensity * 10)
    print(f"""
  ├─ B. 政治压力
  │  方向:    {result.political_direction}
  │  强度:    {pol_bar:<10} {result.political_intensity:.0%}
  │  独立性风险: {result.fed_independence_risk.upper()}
  │  分析: {result.political_context}""")

    # C. 国际联动
    print(f"""
  └─ C. 国际联动
     ECB 方向:    {result.ecb_direction}
     美元趋势:    {result.dollar_trend}
     全球增长风险: {result.global_growth_risk.upper()}
     国际信号:  [{SIGNAL_ICON.get(result.international_signal,'?')}] {result.international_signal.upper()}
     分析: {result.international_context}""")

    # 融合结论
    icon = SIGNAL_ICON.get(result.combined_signal, "?")
    print(f"""
  ══════════════════════════════════════════════════
  三路信号加权融合（市场35% + 政治20% + 国际25% + LLM20%）：

    市场信号:  {result.curve_signal.upper():6}  权重 35%
    政治信号:  {result.political_direction:16}  权重 20%
    国际信号:  {result.international_signal.upper():6}  权重 25%

    ► 综合预测：{result.combined_signal.upper()}  [{icon}]
    ► 综合置信：{result.combined_confidence:.0%}

  推理：{result.synthesis}
  ══════════════════════════════════════════════════""")
    print("★" * W)


# ══════════════════════════════════════════════════════════════════════
# 独立运行
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n影响者联合 Agent 群 启动...")

    # 从数据库读最新宏观数据
    try:
        conn = sqlite3.connect(DB_PATH)
        row  = pd.read_sql(
            "SELECT raw_json FROM meeting_packages ORDER BY id DESC LIMIT 1", conn
        ).iloc[0]["raw_json"]
        conn.close()
        pkg  = json.loads(row)
        macro = {
            "cpi_yoy":           pkg.get("macro", {}).get("cpi_yoy"),
            "unemployment_rate": pkg.get("macro", {}).get("unemployment_rate"),
            "fed_funds_rate":    pkg.get("macro", {}).get("fed_funds_rate"),
            "treasury_10y":      pkg.get("macro", {}).get("treasury_10y"),
        }
    except Exception:
        macro = {
            "cpi_yoy": 4.17, "unemployment_rate": 4.3,
            "fed_funds_rate": 3.63, "treasury_10y": 4.49,
        }
        print("  [降级] 使用缺省宏观数据")

    print(f"  宏观数据: CPI={macro['cpi_yoy']}%  失业={macro['unemployment_rate']}%  "
          f"利率={macro['fed_funds_rate']}%")

    result = run_influencer_agents(macro)
    print_influencer_report(result)
