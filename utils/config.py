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

# 常用快捷访问
LLM_MODEL    = cfg["llm"]["model"]
LLM_BASE_URL = cfg["llm"]["base_url"]
FED_BASE_URL = cfg["data_sources"]["fed_base_url"]
FRED_BASE_URL = cfg["data_sources"]["fred_base_url"]
HTTP_HEADERS = {"User-Agent": cfg["data_sources"]["http_user_agent"]}
NEXT_FOMC    = cfg["fomc_context"]["next_meeting_date"]
HAWKISH_THRESHOLD = cfg["scoring"]["hawkish_threshold"]
DOVISH_THRESHOLD  = cfg["scoring"]["dovish_threshold"]
TRADING_SIGNALS   = cfg["trading_signals"]
FRED_SERIES       = cfg["fred_series"]
FOMC_FALLBACK_TEXT = cfg["fomc_statement_fallback"]
