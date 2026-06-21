import os
import re
import json
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup

load_dotenv()

FRED_KEY = os.getenv("FRED_API_KEY")

# 确保项目根目录在 sys.path，使 utils/ 包可被寻址
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from utils.db import DB_PATH
from utils.config import (FED_BASE_URL  as FED_BASE,
                           FRED_BASE_URL as FRED_BASE,
                           HTTP_HEADERS  as HEADERS,
                           FALLBACK_STMT_URL  as FALLBACK_STATEMENT_URL,
                           FOMC_FALLBACK_TEXT as FALLBACK_STATEMENT_TEXT)


# ── 数据模型 ──────────────────────────────────────────────────────────────────

class MacroIndicators(BaseModel):
    cpi_yoy: Optional[float] = Field(default=None, description="CPI同比增速（%）")
    unemployment_rate: Optional[float] = Field(default=None, description="失业率（%）")
    fed_funds_rate: Optional[float] = Field(default=None, description="联邦基金利率（%）")
    treasury_10y: Optional[float] = Field(default=None, description="10年期国债收益率（%）")
    data_date: str = Field(description="数据获取日期")


class MeetingPackage(BaseModel):
    meeting_date: str = Field(description="FOMC会议日期（YYYY-MM-DD）")
    statement_url: str = Field(description="声明原文URL")
    statement_text: str = Field(description="声明正文（已清洗）")
    macro: MacroIndicators = Field(description="宏观指标快照")
    data_cutoff_date: str = Field(description="信息截断日期（会议日前2天，防前视偏差）")
    packaged_at: str = Field(description="打包时间戳")
    source: str = Field(default="live", description="live=实时抓取 / fallback=本地备用")


# ── 第一步：从美联储官网抓最新声明 ────────────────────────────────────────────

def fetch_fomc_calendar() -> list[dict]:
    """
    爬美联储 FOMC 日历页，提取所有声明链接。
    返回格式：[{"date": "2026-06-17", "url": "https://..."}, ...]，按日期倒序。
    """
    url = f"{FED_BASE}/monetarypolicy/fomccalendars.htm"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [警告] 日历页加载失败：{e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    pattern = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm")
    results = []

    for a_tag in soup.find_all("a", href=pattern):
        href = a_tag["href"]
        match = pattern.search(href)
        if match:
            raw_date = match.group(1)
            formatted = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            results.append({
                "date": formatted,
                "url": FED_BASE + href if href.startswith("/") else href,
            })

    results.sort(key=lambda x: x["date"], reverse=True)
    return results


def fetch_statement_text(url: str) -> str:
    """
    抓取单篇 FOMC 声明页面，提取纯文本（去掉 HTML 标签和导航栏）。
    美联储网页结构：有两个 col-xs-12 col-sm-8 col-md-8 的 div，
    第一个是标题区（字数少），第二个才是正文（字数多）。
    策略：找到所有候选 div，取文字最长的那个。
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = "utf-8"
    except Exception as e:
        print(f"  [警告] 声明页加载失败：{e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # 找所有包含 col-xs-12 和 col-sm-8 的 div，取内容最长的
    candidates = soup.find_all("div", class_=lambda c: c and "col-xs-12" in c and "col-sm-8" in c)
    best_text = ""
    for div in candidates:
        paragraphs = div.find_all("p")
        text = "\n\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs if p.get_text(strip=True))
        if len(text) > len(best_text):
            best_text = text

    # 如果上面没找到，退而求其次：取页面所有超过 50 字的段落
    if len(best_text) < 100:
        all_p = soup.find_all("p")
        best_text = "\n\n".join(
            p.get_text(separator=" ", strip=True)
            for p in all_p
            if len(p.get_text(strip=True)) > 50
        )

    return re.sub(r"\s{3,}", "\n\n", best_text).strip()


# ── 第二步：从 FRED 拉宏观数据 ───────────────────────────────────────────────

def fetch_fred_value(series_id: str, units: str = "lin") -> Optional[float]:
    """
    从 FRED API 拉某个经济指标的最新值。
    units="pc1" 时返回同比增速，units="lin" 时返回原始值。
    没有 API Key 或网络失败时返回 None（不崩溃）。
    """
    if not FRED_KEY:
        return None

    params = {
        "series_id": series_id,
        "api_key": FRED_KEY,
        "file_type": "json",
        "limit": 5,
        "sort_order": "desc",
        "units": units,
    }

    try:
        resp = requests.get(FRED_BASE, params=params, timeout=10)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        for obs in observations:
            if obs.get("value") and obs["value"] != ".":
                return round(float(obs["value"]), 2)
    except Exception as e:
        print(f"  [警告] FRED {series_id} 获取失败：{e}")

    return None


def build_macro_indicators() -> MacroIndicators:
    """
    拉取四项关键宏观指标：CPI同比、失业率、联邦基金利率、10年期国债收益率。
    """
    if not FRED_KEY:
        print("  [提示] 未检测到 FRED_API_KEY，宏观数据将显示为 None")
        print("         注册免费 Key：https://fred.stlouisfed.org/docs/api/api_key.html")
        print("         注册后在 .env 文件加一行：FRED_API_KEY=你的key")

    cpi = fetch_fred_value("CPIAUCSL", units="pc1")        # CPI同比 %
    unrate = fetch_fred_value("UNRATE")                     # 失业率 %
    fedfunds = fetch_fred_value("FEDFUNDS")                 # 联邦基金利率 %
    treasury = fetch_fred_value("DGS10")                    # 10Y国债收益率 %

    return MacroIndicators(
        cpi_yoy=cpi,
        unemployment_rate=unrate,
        fed_funds_rate=fedfunds,
        treasury_10y=treasury,
        data_date=datetime.now().strftime("%Y-%m-%d"),
    )


# ── 第三步：打包成 MeetingPackage ─────────────────────────────────────────────

def build_meeting_package() -> MeetingPackage:
    """
    主流程：抓声明 + 拉宏观数据 → 打包成 MeetingPackage。
    网络不可达时自动降级到本地备用文本。
    """
    print("\n[Agent 1] 正在抓取 FOMC 声明...")
    calendar = fetch_fomc_calendar()
    source = "live"

    if calendar:
        latest = calendar[0]
        meeting_date = latest["date"]
        statement_url = latest["url"]
        print(f"  ✓ 找到最新声明：{meeting_date}  {statement_url}")

        statement_text = fetch_statement_text(statement_url)
        if not statement_text:
            print("  [降级] 声明正文提取失败，使用本地备用文本")
            statement_text = FALLBACK_STATEMENT_TEXT.strip()
            source = "fallback"
        else:
            print(f"  ✓ 声明正文提取成功（{len(statement_text)} 字符）")
    else:
        print("  [降级] 日历页不可达，使用本地备用文本")
        meeting_date = "2026-06-17"
        statement_url = FALLBACK_STATEMENT_URL
        statement_text = FALLBACK_STATEMENT_TEXT.strip()
        source = "fallback"

    # 信息截断日期：会议日前 2 天（防前视偏差，借鉴 FedSight AI）
    cutoff = (datetime.strptime(meeting_date, "%Y-%m-%d") - timedelta(days=2)).strftime("%Y-%m-%d")

    print("\n[Agent 1] 正在获取 FRED 宏观数据...")
    macro = build_macro_indicators()
    print(f"  CPI 同比   : {macro.cpi_yoy}%")
    print(f"  失业率     : {macro.unemployment_rate}%")
    print(f"  联邦基金率 : {macro.fed_funds_rate}%")
    print(f"  10Y国债    : {macro.treasury_10y}%")

    return MeetingPackage(
        meeting_date=meeting_date,
        statement_url=statement_url,
        statement_text=statement_text,
        macro=macro,
        data_cutoff_date=cutoff,
        packaged_at=datetime.now().isoformat(),
        source=source,
    )


# ── 第四步：存入 SQLite ───────────────────────────────────────────────────────

def init_db():
    """建表（如果不存在）。"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meeting_packages (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_date     TEXT NOT NULL,
            statement_url    TEXT,
            statement_text   TEXT,
            cpi_yoy          REAL,
            unemployment_rate REAL,
            fed_funds_rate   REAL,
            treasury_10y     REAL,
            data_cutoff_date TEXT,
            packaged_at      TEXT,
            source           TEXT,
            raw_json         TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(package: MeetingPackage) -> int:
    """
    把 MeetingPackage 存入数据库，同时保存完整 JSON 作为血统追踪备份。
    返回新插入行的 id。
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        INSERT INTO meeting_packages
            (meeting_date, statement_url, statement_text,
             cpi_yoy, unemployment_rate, fed_funds_rate, treasury_10y,
             data_cutoff_date, packaged_at, source, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        package.meeting_date,
        package.statement_url,
        package.statement_text,
        package.macro.cpi_yoy,
        package.macro.unemployment_rate,
        package.macro.fed_funds_rate,
        package.macro.treasury_10y,
        package.data_cutoff_date,
        package.packaged_at,
        package.source,
        package.model_dump_json(),
    ))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Agent 1：数据感知智能体")
    print("=" * 60)

    init_db()

    package = build_meeting_package()
    row_id = save_to_db(package)

    print("\n" + "=" * 60)
    print("MeetingPackage 打包完成，已存入数据库")
    print("=" * 60)
    print(f"数据库路径     : {DB_PATH}")
    print(f"数据库行 ID    : {row_id}")
    print(f"会议日期       : {package.meeting_date}")
    print(f"信息截断日期   : {package.data_cutoff_date}  ← 防前视偏差")
    print(f"声明来源       : {package.source}")
    print(f"声明字数       : {len(package.statement_text)} 字符")
    print(f"打包时间       : {package.packaged_at}")
    print()
    print("声明正文（前 300 字符预览）：")
    print("-" * 40)
    print(package.statement_text[:300], "...")
    print()
    print("→ MeetingPackage 已就绪，可传入 Agent 2（语义提取智能体）")
    print("=" * 60)

    return package


if __name__ == "__main__":
    main()
