"""
统一数据库管理。
DB_PATH 由 config.yml 的 database.filename + 项目根目录拼接，任何机器都能正确定位。
所有 agent 文件 `from utils.db import DB_PATH, get_conn, init_all_tables` 即可。
"""
import sqlite3
from pathlib import Path
from utils.config import cfg

_ROOT   = Path(__file__).resolve().parent.parent
DB_PATH = str(_ROOT / cfg["database"]["filename"])


def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_all_tables() -> None:
    """一次性建立所有表（如果不存在）。pipeline.py 启动时调用一次。"""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS meeting_packages (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_date      TEXT NOT NULL,
            statement_url     TEXT,
            statement_text    TEXT,
            cpi_yoy           REAL,
            unemployment_rate REAL,
            fed_funds_rate    REAL,
            treasury_10y      REAL,
            data_cutoff_date  TEXT,
            packaged_at       TEXT,
            source            TEXT,
            raw_json          TEXT
        );

        CREATE TABLE IF NOT EXISTS semantic_results (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            package_id        INTEGER NOT NULL,
            hawkish_score     REAL,
            inflation_concern REAL,
            growth_concern    REAL,
            key_evidence      TEXT,
            warsh_signal      TEXT,
            conclusion        TEXT,
            raw_json          TEXT,
            analyzed_at       TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS fomc_predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            semantic_id       INTEGER NOT NULL,
            next_meeting_date TEXT,
            predicted_action  TEXT,
            confidence        REAL,
            var_signal        TEXT,
            var_forecast_rate REAL,
            llm_signal        TEXT,
            cod_steps         TEXT,
            risk_factors      TEXT,
            evidence_chain    TEXT,
            final_conclusion  TEXT,
            raw_json          TEXT,
            predicted_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS chair_distill_results (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            greenspan_vote    TEXT,
            bernanke_vote     TEXT,
            yellen_vote       TEXT,
            powell_vote       TEXT,
            most_similar_period TEXT,
            warsh_prediction  TEXT,
            warsh_confidence  REAL,
            final_signal      TEXT,
            raw_json          TEXT,
            created_at        TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS influencer_results (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            market_signal     TEXT,
            political_signal  TEXT,
            intl_signal       TEXT,
            composite_signal  TEXT,
            raw_json          TEXT,
            created_at        TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()
