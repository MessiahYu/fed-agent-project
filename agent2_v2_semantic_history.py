"""
agent2_v2_semantic_history.py
================================================
V2 版本 Agent 2 —— 历史语义特征库构建器
================================================

对 fomc_history 表中 2000-2024 年所有 ~170 次 FOMC 会议进行
批量语义分析，结果存入 semantic_history 表，供 V2 权重优化使用。

设计原则：
  · V1 兼容：agent2_semantic.py 原文不动，V1 pipeline 不受影响
  · 断点续跑：每次会议处理完立即写库，重复运行自动跳过已处理条目
  · 批量静默：LLM 调用使用 verbose=False，避免 170 次 CrewAI 日志刷屏
  · 防前视偏差：宏观背景取 meeting_date - 2天 之前的数据（与 fomc_history 一致）

接口函数（供 V2 pipeline 调用）：
  get_semantic_for_meeting(meeting_date)  → dict
  get_semantic_history_df()              → pd.DataFrame（用于训练集特征矩阵）

运行方式：
  E:\\Anaconda\\envs\\fed-agent\\python.exe agent2_v2_semantic_history.py
"""

import os, re, json, sqlite3, time
from datetime import datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()

import yaml
_cfg  = yaml.safe_load(open("config.yml", encoding="utf-8"))
_sh   = _cfg["v2_semantic_history"]

from utils.db import DB_PATH
from utils.config import LLM_MODEL, LLM_BASE_URL, HTTP_HEADERS
from agent2_semantic import SemanticFeature   # 复用 V1 的 Pydantic 模型，无需重定义

# ── 从 config.yml 读取参数 ────────────────────────────────────────────
HIST_TABLE   = _sh["table_name"]
SLEEP_SEC    = _sh["batch_sleep_seconds"]
MIN_CHARS    = _sh["min_statement_chars"]
URL_NEW      = _sh["url_new"]   # 2011 年后：/newsevents/pressreleases/monetary{date}a.htm
URL_OLD      = _sh["url_old"]   # 2011 年前：/boarddocs/press/general/{year}/{date}/


# ══════════════════════════════════════════════════════════════════════
# 1. 建表（历史语义特征表）
# ══════════════════════════════════════════════════════════════════════

def init_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {HIST_TABLE} (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_date      TEXT NOT NULL UNIQUE,   -- UNIQUE 保证断点续跑不重复
            chair             TEXT,
            actual_decision   TEXT,                   -- 来自 fomc_history，方便验证
            hawkish_score     REAL,
            inflation_concern REAL,
            growth_concern    REAL,
            key_evidence      TEXT,                   -- JSON 数组
            warsh_signal      TEXT,
            conclusion        TEXT,
            statement_url     TEXT,
            statement_chars   INTEGER,
            macro_context     TEXT,                   -- JSON
            raw_json          TEXT,
            analyzed_at       TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════
# 2. 抓取历史 FOMC 声明文本
# ══════════════════════════════════════════════════════════════════════

def _candidate_urls(meeting_date: str) -> list[str]:
    """
    生成两个候选 URL（新版优先，旧版备用）：
      新版（2011+）：/newsevents/pressreleases/monetary{YYYYMMDD}a.htm
      旧版（2000-2010）：/boarddocs/press/general/{YYYY}/{YYYYMMDD}/
    """
    date_compact = meeting_date.replace("-", "")
    year = meeting_date[:4]
    return [
        URL_NEW.format(date=date_compact),
        URL_OLD.format(year=year, date=date_compact),
    ]


def _extract_text(html: str) -> str:
    """从 HTML 提取纯文本，兼容新旧两种页面结构"""
    soup = BeautifulSoup(html, "html.parser")

    # 方法1：新版页面（有 col-xs-12 col-sm-8 div）
    candidates = soup.find_all(
        "div", class_=lambda c: c and "col-xs-12" in c and "col-sm-8" in c
    )
    best_text = ""
    for div in candidates:
        text = "\n\n".join(
            p.get_text(separator=" ", strip=True)
            for p in div.find_all("p") if len(p.get_text(strip=True)) > 30
        )
        if len(text) > len(best_text):
            best_text = text

    # 方法2：旧版页面退路（全页 <p> 标签）
    if len(best_text) < MIN_CHARS:
        best_text = "\n\n".join(
            p.get_text(separator=" ", strip=True)
            for p in soup.find_all("p") if len(p.get_text(strip=True)) > 50
        )

    return re.sub(r"\s{3,}", "\n\n", best_text).strip()


def fetch_statement_text(meeting_date: str) -> tuple[str, str]:
    """
    按优先级尝试两个 URL，返回 (文本, 成功的URL)。
    新版 URL 失败（文本过短）时自动回退到旧版 URL。
    """
    for url in _candidate_urls(meeting_date):
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=15)
            if resp.status_code != 200:
                continue
            resp.encoding = "utf-8"
            text = _extract_text(resp.text)
            if len(text) >= MIN_CHARS:
                return text, url
        except Exception:
            continue
    return "", _candidate_urls(meeting_date)[0]


# ══════════════════════════════════════════════════════════════════════
# 3. 批量静默版语义分析（verbose=False，避免 170 次日志刷屏）
# ══════════════════════════════════════════════════════════════════════

def _run_semantic_silent(
    statement_text: str,
    meeting_date: str,
    macro_context: dict,
) -> SemanticFeature:
    """
    与 agent2_semantic.run_semantic_analysis 逻辑相同，
    但 Agent/Crew 均设为 verbose=False，适合批量运行。
    """
    llm = LLM(
        model=LLM_MODEL,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=LLM_BASE_URL,
    )

    macro_text = ""
    if macro_context:
        macro_text = (
            f"\n宏观数据（会议前2天快照）：\n"
            f"- CPI 同比：{macro_context.get('cpi_yoy', 'N/A')}%\n"
            f"- 失业率：{macro_context.get('unemployment_rate', 'N/A')}%\n"
            f"- 联邦基金利率：{macro_context.get('fed_funds_rate', 'N/A')}%\n"
        )

    analyst = Agent(
        role="美联储政策语义分析师",
        goal="从 FOMC 声明中提取可量化的政策情绪特征向量",
        backstory=(
            "你是顶尖宏观对冲基金的美联储文本分析师，有15年解读FOMC文件的经验。"
            "你擅长识别声明措辞的细微变化，结合宏观数据背景校准评分。"
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=(
            f"请分析以下 {meeting_date} FOMC 声明，提取政策情绪特征。\n"
            f"{macro_text}\n"
            f"声明原文：\n{statement_text}\n\n"
            "你必须只输出一个合法的 JSON 对象（不要加 ``` 标记），格式如下：\n"
            "{\n"
            '  "hawkish_score": <-1.0到1.0，正=鹰派>,\n'
            '  "inflation_concern": <0.0到1.0>,\n'
            '  "growth_concern": <0.0到1.0>,\n'
            '  "key_evidence": ["<证据1>", "<证据2>", "<证据3>"],\n'
            '  "warsh_signal": "<hawkish 或 neutral 或 dovish>",\n'
            '  "conclusion": "<100字以内的政策走向结论>"\n'
            "}"
        ),
        expected_output="严格 JSON，6 个字段，无任何 markdown 标记",
        agent=analyst,
    )

    crew   = Crew(agents=[analyst], tasks=[task], verbose=False)
    result = crew.kickoff()
    raw    = result.raw

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"无法提取 JSON：{raw[:200]}")

    data = json.loads(match.group())
    return SemanticFeature.model_validate(data)


# ══════════════════════════════════════════════════════════════════════
# 4. 存库
# ══════════════════════════════════════════════════════════════════════

def save_to_history(
    meeting_date: str,
    chair: str,
    actual_decision: str,
    feature: SemanticFeature,
    url: str,
    statement_chars: int,
    macro_context: dict,
):
    """INSERT OR IGNORE：meeting_date 已存在则跳过，保证幂等"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"""
        INSERT OR IGNORE INTO {HIST_TABLE}
          (meeting_date, chair, actual_decision,
           hawkish_score, inflation_concern, growth_concern,
           key_evidence, warsh_signal, conclusion,
           statement_url, statement_chars, macro_context, raw_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        meeting_date, chair, actual_decision,
        feature.hawkish_score, feature.inflation_concern, feature.growth_concern,
        json.dumps(feature.key_evidence, ensure_ascii=False),
        feature.warsh_signal, feature.conclusion,
        url, statement_chars,
        json.dumps(macro_context, ensure_ascii=False),
        feature.model_dump_json(),
    ))
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════
# 5. 主批处理流程
# ══════════════════════════════════════════════════════════════════════

def run_batch():
    """
    从 fomc_history 加载所有历史会议，逐一处理。
    已在 semantic_history 中的会议自动跳过（断点续跑）。
    """
    print("\n" + "=" * 64)
    print("  FedWatch AI V2 — 历史语义特征库构建器 启动")
    print("=" * 64)

    init_table()

    # 加载待处理会议列表
    conn = sqlite3.connect(DB_PATH)
    all_meetings = pd.read_sql(
        "SELECT meeting_date, chair, actual_decision, "
        "cpi_yoy, unemployment, fed_funds_rate FROM fomc_history ORDER BY meeting_date",
        conn
    )
    # 已处理的会议
    try:
        done = pd.read_sql(
            f"SELECT meeting_date FROM {HIST_TABLE}", conn
        )["meeting_date"].tolist()
    except Exception:
        done = []
    conn.close()

    pending = all_meetings[~all_meetings["meeting_date"].isin(done)]
    total   = len(all_meetings)
    skipped = len(done)
    todo    = len(pending)

    print(f"\n  历史会议总数：{total}")
    print(f"  已处理（跳过）：{skipped}")
    print(f"  待处理：{todo}")
    if todo == 0:
        print("\n  所有会议已处理完毕！")
        return
    print(f"\n  预计时间：{todo * (SLEEP_SEC + 15) // 60} ~ {todo * (SLEEP_SEC + 30) // 60} 分钟")
    print("  开始批量处理...\n")

    success = 0
    failed  = []

    for i, row in enumerate(pending.itertuples(), 1):
        meeting_date    = row.meeting_date
        chair           = row.chair
        actual_decision = row.actual_decision
        macro = {
            "cpi_yoy":           row.cpi_yoy,
            "unemployment_rate": row.unemployment,
            "fed_funds_rate":    row.fed_funds_rate,
        }

        print(f"  [{skipped + i:>3}/{total}] {meeting_date}  {chair:<12}  "
              f"实际决策={actual_decision.upper()}", end="  ", flush=True)

        # 抓取声明文本（自动尝试新版/旧版两种 URL）
        text, url = fetch_statement_text(meeting_date)
        if len(text) < MIN_CHARS:
            print(f"✗ 声明文本过短（{len(text)}字符），跳过")
            failed.append((meeting_date, "文本过短"))
            continue

        # LLM 语义分析
        try:
            feature = _run_semantic_silent(text, meeting_date, macro)
            save_to_history(meeting_date, chair, actual_decision,
                            feature, url, len(text), macro)
            print(f"✓ hawkish={feature.hawkish_score:+.2f}  "
                  f"通胀={feature.inflation_concern:.2f}  "
                  f"增长={feature.growth_concern:.2f}")
            success += 1
        except Exception as e:
            print(f"✗ LLM 错误：{str(e)[:60]}")
            failed.append((meeting_date, str(e)[:60]))

        # 限速：两次调用之间休眠，避免触发 API 频率限制
        if i < todo:
            time.sleep(SLEEP_SEC)

    # 最终报告
    print("\n" + "=" * 64)
    print(f"  批处理完成！成功 {success}/{todo}，失败 {len(failed)}")
    if failed:
        print("\n  失败列表：")
        for d, reason in failed:
            print(f"    {d}: {reason}")
    print("=" * 64)


# ══════════════════════════════════════════════════════════════════════
# 6. 对外接口（供 V2 pipeline / 权重优化模块调用）
# ══════════════════════════════════════════════════════════════════════

def get_semantic_for_meeting(meeting_date: str) -> dict:
    """
    返回指定会议日期的语义特征（从 semantic_history 表查询）。
    供 V2 回测循环逐条读取。
    """
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute(
        f"SELECT * FROM {HIST_TABLE} WHERE meeting_date = ?",
        (meeting_date,)
    ).fetchone()
    if row:
        cols = [d[0] for d in conn.execute(f"PRAGMA table_info({HIST_TABLE})").fetchall()]
    conn.close()
    return dict(zip(cols, row)) if row else {}


def get_semantic_history_df() -> pd.DataFrame:
    """
    返回完整历史语义特征矩阵（DataFrame），索引 = meeting_date。
    供 V2 权重优化模块直接使用。
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            f"SELECT * FROM {HIST_TABLE} ORDER BY meeting_date",
            conn
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def print_summary():
    """打印数据库中已有的历史语义特征统计"""
    df = get_semantic_history_df()
    if df.empty:
        print("  semantic_history 表为空，请先运行 run_batch()")
        return

    print("\n" + "★" * 64)
    print("  FedWatch AI V2 — 历史语义特征库 摘要")
    print("★" * 64)
    print(f"\n  已处理会议数：{len(df)}")
    print(f"  时间范围：{df['meeting_date'].min()} → {df['meeting_date'].max()}")

    print(f"\n  鹰鸽评分分布：")
    print(f"    均值 = {df['hawkish_score'].mean():+.3f}")
    print(f"    最鹰 = {df['hawkish_score'].max():+.3f}  ({df.loc[df['hawkish_score'].idxmax(), 'meeting_date']})")
    print(f"    最鸽 = {df['hawkish_score'].min():+.3f}  ({df.loc[df['hawkish_score'].idxmin(), 'meeting_date']})")

    print(f"\n  按主席分组均值：")
    for chair, grp in df.groupby("chair"):
        print(f"    {chair:<12}  hawkish={grp['hawkish_score'].mean():+.3f}  "
              f"通胀担忧={grp['inflation_concern'].mean():.3f}  n={len(grp)}")

    print(f"\n  与实际决策的相关性（初步验证）：")
    mapping = {"hold": 0, "cut": -1, "hike": 1}
    df["decision_num"] = df["actual_decision"].map(mapping)
    corr = df[["hawkish_score", "inflation_concern",
               "growth_concern", "decision_num"]].corr()["decision_num"].drop("decision_num")
    for feat, val in corr.items():
        print(f"    {feat:<20} r = {val:+.3f}")

    print("\n★" * 64)


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        # 只打印摘要，不运行批处理
        print_summary()
    else:
        run_batch()
        print_summary()
