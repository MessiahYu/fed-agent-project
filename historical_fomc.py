"""
第一步 (1/2) ── 历史 FOMC 数据库构建器
================================================
功能：
  1. 硬编码 2000-2024 年所有 FOMC 会议日期 + 实际决策
  2. 从 FRED API 批量拉取宏观时间序列（只需 3 次 API 调用）
  3. 为每次会议生成"当时能看到的宏观快照"（防前视偏差）
  4. 保存到 SQLite fomc_history 表

运行方式：
  E:\Anaconda\envs\fed-agent\python.exe historical_fomc.py
"""

import os, json, sqlite3
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
DB_PATH  = r"c:\Users\余青锋\OneDrive\fed-agent-project\fed_watch.db"

# ══════════════════════════════════════════════════════════════════════
# 历史 FOMC 决策表（2000-2024，共 ~170 次会议）
#
# 格式：(会议日期, 主持主席, 决策方向, 变动幅度bp, 会议后目标利率%)
# 数据来源：美联储官网历史记录（本人根据公开记录整理）
# ══════════════════════════════════════════════════════════════════════
FOMC_DECISIONS = [
    # ─────────── Greenspan（格林斯潘）2000-2006 ───────────
    # 2000：先加息到顶，然后维持
    ("2000-02-02", "Greenspan", "hike",  25,  5.75),
    ("2000-03-21", "Greenspan", "hike",  25,  6.00),
    ("2000-05-16", "Greenspan", "hike",  50,  6.50),
    ("2000-06-28", "Greenspan", "hold",   0,  6.50),
    ("2000-08-22", "Greenspan", "hold",   0,  6.50),
    ("2000-10-03", "Greenspan", "hold",   0,  6.50),
    ("2000-11-15", "Greenspan", "hold",   0,  6.50),
    ("2000-12-19", "Greenspan", "hold",   0,  6.50),
    # 2001：互联网泡沫+9/11，紧急连续降息 11 次
    ("2001-01-03", "Greenspan", "cut",  -50,  6.00),  # 紧急会议
    ("2001-01-31", "Greenspan", "cut",  -50,  5.50),
    ("2001-03-20", "Greenspan", "cut",  -50,  5.00),
    ("2001-04-18", "Greenspan", "cut",  -50,  4.50),  # 紧急会议
    ("2001-05-15", "Greenspan", "cut",  -50,  4.00),
    ("2001-06-27", "Greenspan", "cut",  -25,  3.75),
    ("2001-08-21", "Greenspan", "cut",  -25,  3.50),
    ("2001-09-17", "Greenspan", "cut",  -50,  3.00),  # 9/11后紧急
    ("2001-10-02", "Greenspan", "cut",  -50,  2.50),
    ("2001-11-06", "Greenspan", "cut",  -50,  2.00),
    ("2001-12-11", "Greenspan", "cut",  -25,  1.75),
    # 2002：观望，11月再降一次
    ("2002-01-30", "Greenspan", "hold",   0,  1.75),
    ("2002-03-19", "Greenspan", "hold",   0,  1.75),
    ("2002-05-07", "Greenspan", "hold",   0,  1.75),
    ("2002-06-26", "Greenspan", "hold",   0,  1.75),
    ("2002-08-13", "Greenspan", "hold",   0,  1.75),
    ("2002-09-24", "Greenspan", "hold",   0,  1.75),
    ("2002-11-06", "Greenspan", "cut",  -50,  1.25),
    ("2002-12-10", "Greenspan", "hold",   0,  1.25),
    # 2003：再降一次到1%历史低点，然后长期维持
    ("2003-01-29", "Greenspan", "hold",   0,  1.25),
    ("2003-03-18", "Greenspan", "hold",   0,  1.25),
    ("2003-05-06", "Greenspan", "cut",  -25,  1.00),
    ("2003-06-25", "Greenspan", "hold",   0,  1.00),
    ("2003-08-12", "Greenspan", "hold",   0,  1.00),
    ("2003-09-16", "Greenspan", "hold",   0,  1.00),
    ("2003-10-28", "Greenspan", "hold",   0,  1.00),
    ("2003-12-09", "Greenspan", "hold",   0,  1.00),
    # 2004：6月起开始"17连加"，每次25bp
    ("2004-01-28", "Greenspan", "hold",   0,  1.00),
    ("2004-03-16", "Greenspan", "hold",   0,  1.00),
    ("2004-05-04", "Greenspan", "hold",   0,  1.00),
    ("2004-06-30", "Greenspan", "hike",  25,  1.25),
    ("2004-08-10", "Greenspan", "hike",  25,  1.50),
    ("2004-09-21", "Greenspan", "hike",  25,  1.75),
    ("2004-11-10", "Greenspan", "hike",  25,  2.00),
    ("2004-12-14", "Greenspan", "hike",  25,  2.25),
    # 2005：继续加息，共8次
    ("2005-02-02", "Greenspan", "hike",  25,  2.50),
    ("2005-03-22", "Greenspan", "hike",  25,  2.75),
    ("2005-05-03", "Greenspan", "hike",  25,  3.00),
    ("2005-06-30", "Greenspan", "hike",  25,  3.25),
    ("2005-08-09", "Greenspan", "hike",  25,  3.50),
    ("2005-09-20", "Greenspan", "hike",  25,  3.75),
    ("2005-11-01", "Greenspan", "hike",  25,  4.00),
    ("2005-12-13", "Greenspan", "hike",  25,  4.25),
    # 2006：格林斯潘最后一次+伯南克接任，加息至5.25%顶部
    ("2006-01-31", "Greenspan", "hike",  25,  4.50),  # 格林斯潘最后一次
    ("2006-03-28", "Bernanke",  "hike",  25,  4.75),  # 伯南克首次主持
    ("2006-05-10", "Bernanke",  "hike",  25,  5.00),
    ("2006-06-29", "Bernanke",  "hike",  25,  5.25),  # 顶部
    ("2006-08-08", "Bernanke",  "hold",   0,  5.25),
    ("2006-09-20", "Bernanke",  "hold",   0,  5.25),
    ("2006-10-25", "Bernanke",  "hold",   0,  5.25),
    ("2006-12-12", "Bernanke",  "hold",   0,  5.25),

    # ─────────── Bernanke（伯南克）2006-2014 ───────────
    # 2007：次贷危机，9月开始降息
    ("2007-01-31", "Bernanke",  "hold",   0,  5.25),
    ("2007-03-21", "Bernanke",  "hold",   0,  5.25),
    ("2007-05-09", "Bernanke",  "hold",   0,  5.25),
    ("2007-06-28", "Bernanke",  "hold",   0,  5.25),
    ("2007-08-07", "Bernanke",  "hold",   0,  5.25),
    ("2007-09-18", "Bernanke",  "cut",  -50,  4.75),  # 次贷危机信号
    ("2007-10-31", "Bernanke",  "cut",  -25,  4.50),
    ("2007-12-11", "Bernanke",  "cut",  -25,  4.25),
    # 2008：金融危机，激进降息至零
    ("2008-01-22", "Bernanke",  "cut",  -75,  3.50),  # 紧急会议
    ("2008-01-30", "Bernanke",  "cut",  -50,  3.00),
    ("2008-03-18", "Bernanke",  "cut",  -75,  2.25),
    ("2008-04-30", "Bernanke",  "cut",  -25,  2.00),
    ("2008-06-25", "Bernanke",  "hold",   0,  2.00),
    ("2008-08-05", "Bernanke",  "hold",   0,  2.00),
    ("2008-09-16", "Bernanke",  "hold",   0,  2.00),
    ("2008-10-08", "Bernanke",  "cut",  -50,  1.50),  # 全球协调降息
    ("2008-10-29", "Bernanke",  "cut",  -50,  1.00),
    ("2008-12-16", "Bernanke",  "cut",  -75,  0.25),  # 零利率下限（ZLB）
    # 2009-2013：零利率时代，全部 HOLD
    ("2009-01-28", "Bernanke",  "hold",   0,  0.25),
    ("2009-03-18", "Bernanke",  "hold",   0,  0.25),
    ("2009-04-29", "Bernanke",  "hold",   0,  0.25),
    ("2009-06-24", "Bernanke",  "hold",   0,  0.25),
    ("2009-08-12", "Bernanke",  "hold",   0,  0.25),
    ("2009-09-23", "Bernanke",  "hold",   0,  0.25),
    ("2009-11-04", "Bernanke",  "hold",   0,  0.25),
    ("2009-12-16", "Bernanke",  "hold",   0,  0.25),
    ("2010-01-27", "Bernanke",  "hold",   0,  0.25),
    ("2010-03-16", "Bernanke",  "hold",   0,  0.25),
    ("2010-04-28", "Bernanke",  "hold",   0,  0.25),
    ("2010-06-23", "Bernanke",  "hold",   0,  0.25),
    ("2010-08-10", "Bernanke",  "hold",   0,  0.25),
    ("2010-09-21", "Bernanke",  "hold",   0,  0.25),
    ("2010-11-03", "Bernanke",  "hold",   0,  0.25),
    ("2010-12-14", "Bernanke",  "hold",   0,  0.25),
    ("2011-01-26", "Bernanke",  "hold",   0,  0.25),
    ("2011-03-15", "Bernanke",  "hold",   0,  0.25),
    ("2011-04-27", "Bernanke",  "hold",   0,  0.25),
    ("2011-06-22", "Bernanke",  "hold",   0,  0.25),
    ("2011-08-09", "Bernanke",  "hold",   0,  0.25),
    ("2011-09-21", "Bernanke",  "hold",   0,  0.25),
    ("2011-11-02", "Bernanke",  "hold",   0,  0.25),
    ("2011-12-13", "Bernanke",  "hold",   0,  0.25),
    ("2012-01-25", "Bernanke",  "hold",   0,  0.25),
    ("2012-03-13", "Bernanke",  "hold",   0,  0.25),
    ("2012-04-25", "Bernanke",  "hold",   0,  0.25),
    ("2012-06-20", "Bernanke",  "hold",   0,  0.25),
    ("2012-08-01", "Bernanke",  "hold",   0,  0.25),
    ("2012-09-13", "Bernanke",  "hold",   0,  0.25),
    ("2012-10-24", "Bernanke",  "hold",   0,  0.25),
    ("2012-12-12", "Bernanke",  "hold",   0,  0.25),
    ("2013-01-30", "Bernanke",  "hold",   0,  0.25),
    ("2013-03-20", "Bernanke",  "hold",   0,  0.25),
    ("2013-05-01", "Bernanke",  "hold",   0,  0.25),
    ("2013-06-19", "Bernanke",  "hold",   0,  0.25),
    ("2013-07-31", "Bernanke",  "hold",   0,  0.25),
    ("2013-09-18", "Bernanke",  "hold",   0,  0.25),
    ("2013-10-30", "Bernanke",  "hold",   0,  0.25),
    ("2013-12-18", "Bernanke",  "hold",   0,  0.25),
    ("2014-01-29", "Bernanke",  "hold",   0,  0.25),  # 伯南克最后一次

    # ─────────── Yellen（耶伦）2014-2018 ───────────
    ("2014-03-19", "Yellen",    "hold",   0,  0.25),  # 耶伦首次
    ("2014-04-30", "Yellen",    "hold",   0,  0.25),
    ("2014-06-18", "Yellen",    "hold",   0,  0.25),
    ("2014-07-30", "Yellen",    "hold",   0,  0.25),
    ("2014-09-17", "Yellen",    "hold",   0,  0.25),
    ("2014-10-29", "Yellen",    "hold",   0,  0.25),
    ("2014-12-17", "Yellen",    "hold",   0,  0.25),
    # 2015：12月首次加息（时隔9年！）
    ("2015-01-28", "Yellen",    "hold",   0,  0.25),
    ("2015-03-18", "Yellen",    "hold",   0,  0.25),
    ("2015-04-29", "Yellen",    "hold",   0,  0.25),
    ("2015-06-17", "Yellen",    "hold",   0,  0.25),
    ("2015-07-29", "Yellen",    "hold",   0,  0.25),
    ("2015-09-17", "Yellen",    "hold",   0,  0.25),
    ("2015-10-28", "Yellen",    "hold",   0,  0.25),
    ("2015-12-16", "Yellen",    "hike",  25,  0.50),  # 时隔9年首加息
    # 2016：全年只加了一次（12月），中间多次"险些加息"
    ("2016-01-27", "Yellen",    "hold",   0,  0.50),
    ("2016-03-16", "Yellen",    "hold",   0,  0.50),
    ("2016-04-27", "Yellen",    "hold",   0,  0.50),
    ("2016-06-15", "Yellen",    "hold",   0,  0.50),
    ("2016-07-27", "Yellen",    "hold",   0,  0.50),
    ("2016-09-21", "Yellen",    "hold",   0,  0.50),
    ("2016-11-02", "Yellen",    "hold",   0,  0.50),
    ("2016-12-14", "Yellen",    "hike",  25,  0.75),
    # 2017：加了三次
    ("2017-02-01", "Yellen",    "hold",   0,  0.75),
    ("2017-03-15", "Yellen",    "hike",  25,  1.00),
    ("2017-05-03", "Yellen",    "hold",   0,  1.00),
    ("2017-06-14", "Yellen",    "hike",  25,  1.25),
    ("2017-07-26", "Yellen",    "hold",   0,  1.25),
    ("2017-09-20", "Yellen",    "hold",   0,  1.25),
    ("2017-11-01", "Yellen",    "hold",   0,  1.25),
    ("2017-12-13", "Yellen",    "hike",  25,  1.50),
    ("2018-01-31", "Yellen",    "hold",   0,  1.50),  # 耶伦最后一次

    # ─────────── Powell（鲍威尔）2018-2026 ───────────
    ("2018-03-21", "Powell",    "hike",  25,  1.75),  # 鲍威尔首次
    ("2018-05-02", "Powell",    "hold",   0,  1.75),
    ("2018-06-13", "Powell",    "hike",  25,  2.00),
    ("2018-08-01", "Powell",    "hold",   0,  2.00),
    ("2018-09-26", "Powell",    "hike",  25,  2.25),
    ("2018-11-08", "Powell",    "hold",   0,  2.25),
    ("2018-12-19", "Powell",    "hike",  25,  2.50),
    # 2019：贸易战压力，3次"保险性降息"
    ("2019-01-30", "Powell",    "hold",   0,  2.50),
    ("2019-03-20", "Powell",    "hold",   0,  2.50),
    ("2019-05-01", "Powell",    "hold",   0,  2.50),
    ("2019-06-19", "Powell",    "hold",   0,  2.50),
    ("2019-07-31", "Powell",    "cut",  -25,  2.25),
    ("2019-09-18", "Powell",    "cut",  -25,  2.00),
    ("2019-10-30", "Powell",    "cut",  -25,  1.75),
    ("2019-12-11", "Powell",    "hold",   0,  1.75),
    # 2020：新冠疫情，紧急两次降至零
    ("2020-01-29", "Powell",    "hold",   0,  1.75),
    ("2020-03-03", "Powell",    "cut",  -50,  1.25),  # 紧急
    ("2020-03-15", "Powell",    "cut", -100,  0.25),  # 紧急，回零
    ("2020-04-29", "Powell",    "hold",   0,  0.25),
    ("2020-06-10", "Powell",    "hold",   0,  0.25),
    ("2020-07-29", "Powell",    "hold",   0,  0.25),
    ("2020-09-16", "Powell",    "hold",   0,  0.25),
    ("2020-11-05", "Powell",    "hold",   0,  0.25),
    ("2020-12-16", "Powell",    "hold",   0,  0.25),
    # ── 以下为测试集（2021-2024）──
    # 2021：通胀暂时论，全年按兵不动
    ("2021-01-27", "Powell",    "hold",   0,  0.25),
    ("2021-03-17", "Powell",    "hold",   0,  0.25),
    ("2021-04-28", "Powell",    "hold",   0,  0.25),
    ("2021-06-16", "Powell",    "hold",   0,  0.25),
    ("2021-07-28", "Powell",    "hold",   0,  0.25),
    ("2021-09-22", "Powell",    "hold",   0,  0.25),
    ("2021-11-03", "Powell",    "hold",   0,  0.25),
    ("2021-12-15", "Powell",    "hold",   0,  0.25),
    # 2022：史上最快加息周期（CPI最高9.1%）
    ("2022-01-26", "Powell",    "hold",   0,  0.25),
    ("2022-03-16", "Powell",    "hike",  25,  0.50),
    ("2022-05-04", "Powell",    "hike",  50,  1.00),
    ("2022-06-15", "Powell",    "hike",  75,  1.75),  # 首次75bp
    ("2022-07-27", "Powell",    "hike",  75,  2.50),
    ("2022-09-21", "Powell",    "hike",  75,  3.25),
    ("2022-11-02", "Powell",    "hike",  75,  4.00),
    ("2022-12-14", "Powell",    "hike",  50,  4.50),
    # 2023：减速收尾，最后加至5.25-5.5%顶部
    ("2023-02-01", "Powell",    "hike",  25,  4.75),
    ("2023-03-22", "Powell",    "hike",  25,  5.00),
    ("2023-05-03", "Powell",    "hike",  25,  5.25),
    ("2023-06-14", "Powell",    "hold",   0,  5.25),  # "跳过"（skip）
    ("2023-07-26", "Powell",    "hike",  25,  5.50),  # 最后一次加息
    ("2023-09-20", "Powell",    "hold",   0,  5.50),
    ("2023-11-01", "Powell",    "hold",   0,  5.50),
    ("2023-12-13", "Powell",    "hold",   0,  5.50),
    # 2024：三次降息启动宽松周期
    ("2024-01-31", "Powell",    "hold",   0,  5.50),
    ("2024-03-20", "Powell",    "hold",   0,  5.50),
    ("2024-05-01", "Powell",    "hold",   0,  5.50),
    ("2024-06-12", "Powell",    "hold",   0,  5.50),
    ("2024-07-31", "Powell",    "hold",   0,  5.50),
    ("2024-09-18", "Powell",    "cut",  -50,  5.00),  # 首降50bp
    ("2024-11-07", "Powell",    "cut",  -25,  4.75),
    ("2024-12-18", "Powell",    "cut",  -25,  4.50),
]


# ══════════════════════════════════════════════════════════════════════
# FRED 数据拉取（只调用 3 次，拿完整时间序列）
# ══════════════════════════════════════════════════════════════════════

def fetch_fred_series(series_id: str, start: str = "1998-01-01") -> pd.Series:
    """从 FRED 拉取一条时间序列，返回 pd.Series（索引=日期，值=数据）"""
    if not FRED_KEY:
        print(f"  [警告] 没有 FRED_API_KEY，{series_id} 将用 None 填充")
        return pd.Series(dtype=float)

    url = "https://api.stlouisfed.org/fred/series/observations"
    resp = requests.get(url, params={
        "series_id":        series_id,
        "api_key":          FRED_KEY,
        "file_type":        "json",
        "observation_start": start,
        "frequency":        "m",   # 月频
        "aggregation_method": "avg",
    }, timeout=30)

    if resp.status_code != 200:
        print(f"  [警告] FRED {series_id} 请求失败 {resp.status_code}")
        return pd.Series(dtype=float)

    obs = resp.json().get("observations", [])
    data = {}
    for o in obs:
        try:
            data[pd.Timestamp(o["date"])] = float(o["value"])
        except (ValueError, KeyError):
            pass
    return pd.Series(data, name=series_id)


def get_macro_snapshot(s_cpi, s_unrate, s_fedfunds, meeting_date: str):
    """
    以 meeting_date 前两天为截断点，取当时能看到的最新宏观数据。
    这模拟了"FOMC委员在开会前能看到什么数据"，防止前视偏差。
    """
    cutoff = pd.Timestamp(meeting_date) - timedelta(days=2)

    def latest_before(s: pd.Series):
        subset = s[s.index <= cutoff].dropna()
        return round(float(subset.iloc[-1]), 4) if len(subset) > 0 else None

    return {
        "cpi_yoy":           latest_before(s_cpi),
        "unemployment_rate": latest_before(s_unrate),
        "fed_funds_rate":    latest_before(s_fedfunds),
    }


# ══════════════════════════════════════════════════════════════════════
# 数据库写入
# ══════════════════════════════════════════════════════════════════════

def build_historical_db():
    """主函数：构建 fomc_history 表"""
    print("\n" + "="*60)
    print("  历史 FOMC 数据库构建器 启动")
    print("="*60)

    # ── Step 1: 拉取完整时间序列（3次API调用）──────────────────────
    print("\n[1/3] 从 FRED 拉取宏观时间序列...")
    s_cpi      = fetch_fred_series("CPIAUCSL")   # CPI（需要自己算同比）
    s_cpi_yoy  = s_cpi.pct_change(12) * 100      # 转换为同比%
    s_unrate   = fetch_fred_series("UNRATE")      # 失业率
    s_fedfunds = fetch_fred_series("FEDFUNDS")    # 联邦基金利率（月均）
    print(f"  CPI同比数据点数: {len(s_cpi_yoy)}")
    print(f"  失业率数据点数:  {len(s_unrate)}")
    print(f"  联邦基金利率点数:{len(s_fedfunds)}")

    # ── Step 2: 建表 ──────────────────────────────────────────────
    print("\n[2/3] 创建/重建 fomc_history 表...")
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS fomc_history")
    cur.execute("""
        CREATE TABLE fomc_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_date    TEXT NOT NULL,
            chair           TEXT NOT NULL,
            actual_decision TEXT NOT NULL,   -- hold / cut / hike
            change_bps      INTEGER,         -- 变动幅度（负=降息）
            target_rate     REAL,            -- 会议后目标利率%
            cpi_yoy         REAL,            -- 当时CPI同比%
            unemployment    REAL,            -- 当时失业率%
            fed_funds_rate  REAL,            -- 当时联邦基金利率%
            split           TEXT NOT NULL,   -- train / test
            raw_json        TEXT             -- 完整备份
        )
    """)
    conn.commit()

    # ── Step 3: 逐条插入 ──────────────────────────────────────────
    print(f"\n[3/3] 处理 {len(FOMC_DECISIONS)} 次会议记录...")
    inserted = 0
    for row in FOMC_DECISIONS:
        date, chair, decision, change_bps, target_rate = row

        # 拉取当时宏观快照
        macro = get_macro_snapshot(s_cpi_yoy, s_unrate, s_fedfunds, date)

        # 判断属于训练集还是测试集
        year = int(date[:4])
        split = "test" if year >= 2021 else "train"

        raw = json.dumps({
            "meeting_date": date, "chair": chair,
            "actual_decision": decision, "change_bps": change_bps,
            "target_rate": target_rate, "split": split, **macro
        }, ensure_ascii=False)

        cur.execute("""
            INSERT INTO fomc_history
              (meeting_date, chair, actual_decision, change_bps, target_rate,
               cpi_yoy, unemployment, fed_funds_rate, split, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (date, chair, decision, change_bps, target_rate,
              macro["cpi_yoy"], macro["unemployment_rate"],
              macro["fed_funds_rate"], split, raw))
        inserted += 1

        if inserted % 20 == 0:
            print(f"  已处理 {inserted}/{len(FOMC_DECISIONS)} 条...")

    conn.commit()
    conn.close()

    # ── 汇总统计 ──────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT split, actual_decision, COUNT(*) as cnt "
                     "FROM fomc_history GROUP BY split, actual_decision", conn)
    conn.close()

    print("\n" + "="*60)
    print("  数据库构建完成！汇总统计：")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\n  总计 {inserted} 条记录写入 fomc_history 表")
    print(f"  数据库路径: {DB_PATH}")
    print("\n  [提示] 运行 backtest.py 开始回测！")


if __name__ == "__main__":
    build_historical_db()
