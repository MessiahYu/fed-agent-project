"""
统一配置加载器。
所有模块 `from utils.config import cfg, LLM_MODEL, LLM_BASE_URL, NEXT_FOMC` 即可。
"""
import os
import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "config.yml"

with open(_CONFIG_PATH, encoding="utf-8") as _f:
    cfg: dict = yaml.safe_load(_f)

# ── LLM ──────────────────────────────────────────────────────
LLM_MODEL    = cfg["llm"]["model"]
LLM_BASE_URL = cfg["llm"]["base_url"]

# ── 数据源 ────────────────────────────────────────────────────
FED_BASE_URL       = cfg["data_sources"]["fed_base_url"]
FRED_BASE_URL      = cfg["data_sources"]["fred_base_url"]
FRED_RECENT_START  = cfg.get("fred_recent_start", "2024-01-01")
FALLBACK_STMT_URL  = cfg["data_sources"]["fallback_statement_url"]
HTTP_HEADERS       = {"User-Agent": cfg["data_sources"]["http_user_agent"]}
FOMC_FALLBACK_TEXT = cfg["fomc_statement_fallback"]

# ── FOMC 背景 ─────────────────────────────────────────────────
NEXT_FOMC          = cfg["fomc_context"]["next_meeting_date"]
CURRENT_CHAIR      = cfg["fomc_context"]["current_chair"]

# ── FRED 系列 ID ──────────────────────────────────────────────
FRED_SERIES        = cfg["fred_series"]

# ── 评分 & 交易信号 ───────────────────────────────────────────
HAWKISH_THRESHOLD  = cfg["scoring"]["hawkish_threshold"]
DOVISH_THRESHOLD   = cfg["scoring"]["dovish_threshold"]
TRADING_SIGNALS    = cfg["trading_signals"]

# ── 回测参数 ──────────────────────────────────────────────────
BACKTEST_THRESHOLD = cfg["backtest"]["threshold_bps"]
VAR_LAGS           = cfg["backtest"]["var_lags"]
VAR_WINDOW         = cfg["backtest"]["var_window"]
TRAIN_SPLIT_YEAR   = cfg["backtest"]["train_split_year"]

# ── 融合权重 ──────────────────────────────────────────────────
FUSION_WEIGHTS     = cfg["fusion_weights"]

# ── 置信度 ────────────────────────────────────────────────────
CONFIDENCE         = cfg["confidence"]

# ── 分析背景文本 ──────────────────────────────────────────────
CONTEXT_POLITICAL      = cfg["context"]["political_2026"]
CONTEXT_INTERNATIONAL  = cfg["context"]["international_2026"]
CONTEXT_HISTORICAL     = cfg["context"]["historical_cases"]
