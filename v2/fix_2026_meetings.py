"""
v2/fix_2026_meetings.py
================================================
修复 2026 年 fomc_history 会议日期（全部对准周三声明发布日）

错误日期     → 正确日期      原因
2026-01-29  → 2026-01-28    Jan 27-28 会议，CUT -25bp → 3.75%
2026-03-19  → 2026-03-18    Mar 17-18 会议，HOLD
2026-05-07  → 2026-04-29    Apr 28-29 会议，HOLD（Powell 最后一次主持）

运行方式：
  E:\\Anaconda\\envs\\fed-agent\\python.exe v2/fix_2026_meetings.py
"""

import json, os, sqlite3, sys, time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv()

FRED_KEY = os.getenv("FRED_API_KEY")
_cfg     = yaml.safe_load(open(str(_ROOT / "config.yml"), encoding="utf-8"))
from utils.db import DB_PATH

# ── 修正映射 ──────────────────────────────────────────────────────────
CORRECTIONS = {
    "2026-01-29": ("2026-01-28", "Powell", "cut",  -25, 3.75),
    "2026-03-19": ("2026-03-18", "Powell", "hold",   0, 3.75),
    "2026-05-07": ("2026-04-29", "Powell", "hold",   0, 3.75),
}

# 新增正确记录时需要的宏观数据（从 FRED 拉取）
def _latest_macro(date_str: str) -> dict:
    cutoff = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=2)).strftime("%Y-%m-%d")
    def _get(sid):
        if not FRED_KEY: return None
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": sid, "api_key": FRED_KEY, "file_type": "json",
                    "observation_start": "2025-01-01", "observation_end": cutoff,
                    "frequency": "m", "aggregation_method": "avg",
                    "sort_order": "desc", "limit": 1},
            timeout=20,
        )
        if r.status_code != 200: return None
        obs = r.json().get("observations", [])
        try: return float(obs[0]["value"]) if obs else None
        except (ValueError, KeyError): return None

    fedfunds = _get("FEDFUNDS")
    unrate   = _get("UNRATE")

    # CPI 同比（取最近月 / 去年同期）
    cpi_now = _get("CPIAUCSL")
    cpi_yoy = None
    if cpi_now and FRED_KEY:
        c1 = (datetime.strptime(cutoff, "%Y-%m-%d") - timedelta(days=395)).strftime("%Y-%m-%d")
        c2 = (datetime.strptime(cutoff, "%Y-%m-%d") - timedelta(days=335)).strftime("%Y-%m-%d")
        r2 = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "CPIAUCSL", "api_key": FRED_KEY, "file_type": "json",
                    "observation_start": c1, "observation_end": c2,
                    "frequency": "m", "sort_order": "desc", "limit": 1},
            timeout=20,
        )
        if r2.status_code == 200:
            obs2 = r2.json().get("observations", [])
            try:
                cpi_y1 = float(obs2[0]["value"]) if obs2 else None
                if cpi_y1 and cpi_now and cpi_y1 > 0:
                    cpi_yoy = round((cpi_now / cpi_y1 - 1) * 100, 2)
            except (ValueError, KeyError, TypeError): pass
    return {"cpi_yoy": cpi_yoy, "unemployment": unrate, "fed_funds_rate": fedfunds}


def fix_fomc_history():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    print("\n[1/3] 修复 fomc_history 日期...")
    for wrong_date, (correct_date, chair, decision, chg, rate) in CORRECTIONS.items():
        # 删旧
        deleted = cur.execute("DELETE FROM fomc_history WHERE meeting_date=?",
                              (wrong_date,)).rowcount
        if deleted:
            print(f"  删除旧记录: {wrong_date}")

        # 删 semantic_history 旧记录（如有）
        cur.execute("DELETE FROM semantic_history WHERE meeting_date=?", (wrong_date,))

        # 是否已有正确记录
        if cur.execute("SELECT id FROM fomc_history WHERE meeting_date=?",
                       (correct_date,)).fetchone():
            print(f"  正确记录已存在: {correct_date}，跳过")
            continue

        macro = _latest_macro(correct_date)
        raw   = json.dumps({"meeting_date": correct_date, "chair": chair,
                            "decision": decision, "change_bps": chg,
                            "target_rate": rate, **macro}, ensure_ascii=False)

        cur.execute("""
            INSERT INTO fomc_history
              (meeting_date, chair, actual_decision, change_bps, target_rate,
               cpi_yoy, unemployment, fed_funds_rate, split, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (correct_date, chair, decision, chg, rate,
              macro["cpi_yoy"], macro["unemployment"], macro["fed_funds_rate"],
              "final_test", raw))

        print(f"  插入正确记录: {correct_date}  {chair}  "
              f"{decision.upper()} {chg:+}bp → {rate}%")

    conn.commit()
    conn.close()


def run_semantic_for_new_dates():
    """对三个修正后的日期跑语义分析（复用现有批处理函数）"""
    print("\n[2/3] 对修正后的三次会议跑语义分析...")
    from v2.agent2_v2_semantic_history import run_batch
    run_batch()


def show_final_test_summary():
    print("\n[3/3] 最终测试集数据汇总...")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT f.meeting_date, f.chair, f.actual_decision,
               s.hawkish_score, s.conclusion
        FROM fomc_history f
        LEFT JOIN semantic_history s ON f.meeting_date = s.meeting_date
        WHERE f.split = 'final_test'
        ORDER BY f.meeting_date
    """).fetchall()
    conn.close()

    print(f"\n  {'日期':12}  {'主席':8}  {'决策':6}  {'LLM分':>7}  备注")
    print("  " + "─" * 62)
    sem_count = 0
    for row in rows:
        date, chair, dec, score, conclusion = row
        score_str = f"{score:+.2f}" if score is not None else "缺失"
        if score is not None:
            sem_count += 1
        print(f"  {date:12}  {chair:8}  {dec.upper():6}  {score_str:>7}")

    print(f"\n  共 {len(rows)} 次会议，semantic_history 覆盖 {sem_count}/{len(rows)}")
    return sem_count, len(rows)


if __name__ == "__main__":
    print("=" * 60)
    print("  修复 2026 年 FOMC 会议日期错误")
    print("=" * 60)

    fix_fomc_history()
    run_semantic_for_new_dates()
    sem_ok, total = show_final_test_summary()

    if sem_ok == total:
        print(f"\n  所有 {total} 次会议数据完整，正在重新计算测试集...")
        from v2.final_test_eval import evaluate_final_test
        evaluate_final_test()
    else:
        print(f"\n  仍有 {total - sem_ok} 次会议缺失 semantic_history，建议排查后重新运行。")
