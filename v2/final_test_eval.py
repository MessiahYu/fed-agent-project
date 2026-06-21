"""
v2/final_test_eval.py
================================================
最终测试集：2025 年至今（保留集，从未用于训练/验证）
================================================

流程：
  Phase 1  从 FRED DFEDTARU（每日目标利率上界）精确推断每次会议决策
  Phase 2  把 2025-2026 会议写入 fomc_history（split='final_test'）
  Phase 3  对这些声明跑语义分析（复用 agent2_v2_semantic_history.run_batch）
  Phase 4  用 config.yml 里的最优权重做融合预测，报告准确率

运行方式（从项目根目录）：
  E:\\Anaconda\\envs\\fed-agent\\python.exe v2/final_test_eval.py
"""

import json, os, sqlite3, sys, warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── sys.path ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
_cfg     = yaml.safe_load(open(str(_ROOT / "config.yml"), encoding="utf-8"))

from utils.db import DB_PATH

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2025-2026 FOMC 官方日历（每次会议第 2 天 = 决策公布日）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_ALL_MEETINGS = [
    ("Powell", "2025-01-29"),
    ("Powell", "2025-03-19"),
    ("Powell", "2025-05-07"),   # Apr 30 - May 1
    ("Powell", "2025-06-18"),
    ("Powell", "2025-07-30"),
    ("Powell", "2025-09-17"),
    ("Powell", "2025-10-29"),
    ("Powell", "2025-12-10"),
    ("Powell", "2026-01-29"),
    ("Powell", "2026-03-19"),
    ("Powell", "2026-05-07"),   # Powell 最后一次（Warsh 5月22日接任）
    ("Warsh",  "2026-06-17"),   # Warsh 首次（CLAUDE.md 确认维持不变）
]

TODAY = datetime.now().strftime("%Y-%m-%d")
FOMC_2025_2026 = [(c, d) for c, d in _ALL_MEETINGS if d <= TODAY]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1：用 DFEDTARU 精确推断每次会议的 HOLD/HIKE/CUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_daily_target_rate(start="2024-10-01") -> pd.Series:
    """
    DFEDTARU = 联邦基金目标利率上界（日度）。
    每次 FOMC 改变目标利率时更新，与会议日期精确对应。
    """
    if not FRED_KEY:
        print("  [警告] 无 FRED_API_KEY")
        return pd.Series(dtype=float)

    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": "DFEDTARU", "api_key": FRED_KEY,
                "file_type": "json", "observation_start": start,
                "frequency": "d"},
        timeout=30,
    )
    if r.status_code != 200:
        print(f"  [警告] DFEDTARU 请求失败 {r.status_code}")
        return pd.Series(dtype=float)

    data = {}
    for o in r.json().get("observations", []):
        try:
            data[pd.Timestamp(o["date"])] = float(o["value"])
        except (ValueError, KeyError):
            pass
    s = pd.Series(data, name="DFEDTARU").dropna()
    print(f"  DFEDTARU: {len(s)} 个日度数据，最新 {s.index[-1].date()} = {s.iloc[-1]:.2f}%")
    return s


def infer_decisions(dfedtaru: pd.Series) -> list:
    """
    对每次会议比较 DFEDTARU 在会议日前后的变化 → HIKE/HOLD/CUT。
    返回 list of (chair, date_str, decision, change_bps, target_rate)
    """
    results   = []
    prev_rate = None

    for chair, date_str in FOMC_2025_2026:
        ts = pd.Timestamp(date_str)

        if dfedtaru.empty:
            dec, chg, rate = _infer_from_fedfunds(date_str, prev_rate)
        else:
            after = dfedtaru[dfedtaru.index >= ts]
            if after.empty:
                results.append((chair, date_str, "unknown", 0, None))
                continue

            rate_now = after.iloc[0]
            rate_pre = (prev_rate if prev_rate is not None
                        else dfedtaru[dfedtaru.index < ts].iloc[-1]
                        if len(dfedtaru[dfedtaru.index < ts]) > 0
                        else rate_now)

            change = round(rate_now - rate_pre, 2)
            if   change > 0: dec, chg = "hike", int(change * 100)
            elif change < 0: dec, chg = "cut",  int(change * 100)
            else:            dec, chg = "hold", 0

            rate      = rate_now
            prev_rate = rate_now

        results.append((chair, date_str, dec, chg, rate))

    return results


def _infer_from_fedfunds(date_str: str, prev_rate) -> tuple:
    """FRED API 不可用时，从 political_quant_monthly 月均利率降级推断"""
    conn  = sqlite3.connect(DB_PATH)
    month = date_str[:7] + "-01"
    row   = conn.execute(
        "SELECT fedfunds FROM political_quant_monthly WHERE month=?", (month,)
    ).fetchone()
    conn.close()

    if not row or row[0] is None:
        return "unknown", 0, None

    rate = round(row[0], 2)
    if prev_rate is None:
        return "hold", 0, rate

    chg = round(rate - prev_rate, 2)
    if   chg >  0.12: return "hike", int(chg * 100), rate
    elif chg < -0.12: return "cut",  int(chg * 100), rate
    else:             return "hold", 0, rate


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2：写入 fomc_history
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fetch_macro_snapshot(date_str: str) -> dict:
    """取会议前最新的宏观快照（逐点调用 FRED，共 3 个指标）"""
    cutoff = pd.Timestamp(date_str) - timedelta(days=2)

    def _latest(sid):
        if not FRED_KEY:
            return None
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": sid, "api_key": FRED_KEY,
                    "file_type": "json", "observation_start": "2024-01-01",
                    "observation_end": cutoff.strftime("%Y-%m-%d"),
                    "frequency": "m", "aggregation_method": "avg",
                    "sort_order": "desc", "limit": 1},
            timeout=20,
        )
        if r.status_code != 200:
            return None
        obs = r.json().get("observations", [])
        try:
            return float(obs[0]["value"]) if obs else None
        except (ValueError, KeyError):
            return None

    cpi_raw = _latest("CPIAUCSL")
    cpi_yoy = None
    if cpi_raw and FRED_KEY:
        y1_end   = cutoff - timedelta(days=335)
        r2 = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "CPIAUCSL", "api_key": FRED_KEY,
                    "file_type": "json",
                    "observation_start": (y1_end - timedelta(days=30)).strftime("%Y-%m-%d"),
                    "observation_end": y1_end.strftime("%Y-%m-%d"),
                    "frequency": "m", "aggregation_method": "avg",
                    "sort_order": "desc", "limit": 1},
            timeout=20,
        )
        if r2.status_code == 200:
            obs2 = r2.json().get("observations", [])
            try:
                cpi_y1 = float(obs2[0]["value"]) if obs2 else None
                if cpi_y1 and cpi_y1 > 0:
                    cpi_yoy = round((cpi_raw / cpi_y1 - 1) * 100, 2)
            except (ValueError, KeyError, TypeError):
                pass

    return {
        "cpi_yoy":        cpi_yoy,
        "unemployment":   _latest("UNRATE"),
        "fed_funds_rate": _latest("FEDFUNDS"),
    }


def insert_final_test_meetings(decisions: list) -> tuple:
    """把推断的 2025-2026 会议记录写入 fomc_history（已存在则跳过）"""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    inserted = skipped = 0

    for chair, date_str, decision, change_bps, target_rate in decisions:
        if decision == "unknown":
            print(f"  [跳过] {date_str}：决策未知")
            skipped += 1
            continue

        if cur.execute("SELECT id FROM fomc_history WHERE meeting_date=?",
                       (date_str,)).fetchone():
            skipped += 1
            continue

        macro = _fetch_macro_snapshot(date_str)
        raw   = json.dumps({"meeting_date": date_str, "chair": chair,
                            "decision": decision, "change_bps": change_bps,
                            **macro}, ensure_ascii=False)

        cur.execute("""
            INSERT INTO fomc_history
              (meeting_date, chair, actual_decision, change_bps, target_rate,
               cpi_yoy, unemployment, fed_funds_rate, split, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (date_str, chair, decision, change_bps, target_rate,
              macro["cpi_yoy"], macro["unemployment"], macro["fed_funds_rate"],
              "final_test", raw))

        rate_str = f"{target_rate:.2f}%" if target_rate else "?"
        print(f"  [写入] {date_str}  {chair:8} {decision.upper():5} "
              f"{change_bps:+4}bp → {rate_str}")
        inserted += 1

    conn.commit()
    conn.close()
    return inserted, skipped


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 4：融合评估（Phase 3 由 run_batch 外部完成后调用此函数）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_final_test():
    """加载 final_test 特征矩阵 → VAR 信号 → 融合预测 → 打印报告"""
    from v2.weight_optimizer import (
        load_feature_matrix, compute_var_signals, add_numeric_signals,
        fuse_predict, compute_metrics, CLASSES,
    )
    from v2.backtest import fetch_full_macro_series

    w_cfg    = _cfg.get("v2_optimal_weights", {})
    w_llm    = w_cfg.get("w_llm",    0.45)
    w_var    = w_cfg.get("w_var",    0.20)
    w_market = w_cfg.get("w_market", 0.15)
    w_pol    = w_cfg.get("w_pol",    0.20)
    hike_thr = w_cfg.get("hike_thr", 0.25)
    cut_thr  = w_cfg.get("cut_thr", -0.35)

    df_all = load_feature_matrix()
    df_ft  = df_all[df_all["split"] == "final_test"].copy().reset_index(drop=True)

    if df_ft.empty:
        print("  [警告] final_test 特征矩阵为空（semantic_history 尚无数据）")
        return

    print(f"  计算 {len(df_ft)} 次 VAR 信号...")
    df_macro = fetch_full_macro_series()
    var_sigs = compute_var_signals(df_ft, df_macro)
    df_ft    = add_numeric_signals(df_ft, var_sigs)

    pred = fuse_predict(
        df_ft["hawkish_score"].values.astype(float),
        df_ft["var_num"].values.astype(float),
        df_ft["market_num"].values.astype(float),
        df_ft["pol_hawkish"].values.astype(float),
        w_llm, w_var, w_market, w_pol, hike_thr, cut_thr,
    )
    df_ft["pred"] = pred
    actual   = df_ft["label"].values
    metrics  = compute_metrics(actual, pred)
    baseline = (df_ft["label"] == "hold").mean()

    train_acc = float(w_cfg.get("train_acc", 0))
    val_acc   = float(w_cfg.get("val_acc",   0))

    W = 66
    print("\n" + "★" * W)
    print("  FedWatch AI V2 — 最终测试集报告（2025-至今，保留集）")
    print("★" * W)

    print(f"""
  ── 最优融合权重（网格搜索自 2000-2020 训练集）────────────────────
  w_llm={w_llm}  w_var={w_var}  w_market={w_market}  w_pol={w_pol}
  hike_thr={hike_thr:+.2f}  cut_thr={cut_thr:+.2f}

  ── 三集准确率全景 ────────────────────────────────────────────────
  {"数据集":<22}  {"样本数":>6}  {"基线":>8}  {"V2模型":>8}  {"超基线":>8}
  {"─"*60}
  {"训练集 (2000-2020)":<22}  {"173":>6}  {"67.1%":>8}  {train_acc*100:.1f}%{" ":>3}  {(train_acc-0.671)*100:>+7.1f}%
  {"验证集 (2021-2024)":<22}  {"32":>6}  {"56.2%":>8}  {val_acc*100:.1f}%{" ":>3}  {(val_acc-0.562)*100:>+7.1f}%
  {"最终测试集 (2025-今)":<22}  {len(df_ft):>6}  {baseline:>8.1%}  {metrics["accuracy"]:>8.1%}  {metrics["accuracy"]-baseline:>+8.1%}
""")

    print("  ── 最终测试集 各类别指标 ────────────────────────────────────")
    print(f"  {'类别':6}  {'精确率':>6}  {'召回率':>6}  {'F1':>6}  {'样本数':>5}")
    for cls in CLASSES:
        pc = metrics["per_class"][cls]
        print(f"  {cls.upper():6}  {pc['precision']:>6.1%}  "
              f"{pc['recall']:>6.1%}  {pc['f1']:>6.1%}  {pc['support']:>5}")

    print(f"\n  ── 混淆矩阵（行=实际，列=预测）────────────────────────────")
    print(f"  {'':12}  {'HOLD':>6}  {'CUT':>6}  {'HIKE':>6}")
    cm = metrics["conf_matrix"]
    for a in CLASSES:
        row = cm.get(a, {})
        print(f"  实际-{a.upper():4}  {row.get('hold',0):>6}  "
              f"{row.get('cut',0):>6}  {row.get('hike',0):>6}")

    print(f"\n  ── 逐次预测明细 ───────────────────────────────────────────")
    print(f"  {'日期':12}  {'主席':8}  {'LLM分':>6}  {'VAR':>5}  {'预测':6}  {'实际':6}  结果")
    print("  " + "─" * 62)
    for _, r in df_ft.iterrows():
        ok = "✓" if r["pred"] == r["label"] else "✗"
        print(f"  {str(r['meeting_date'])[:10]:12}  {r['chair']:8}  "
              f"{r['hawkish_score']:>+6.2f}  {r['var_signal']:>5}  "
              f"{r['pred'].upper():6}  {r['label'].upper():6}  {ok}")

    print("\n" + "★" * W)
    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("\n" + "=" * 66)
    print("  FedWatch AI V2 — 最终测试集评估 启动")
    print(f"  今天：{TODAY}  |  覆盖 2025-01-01 至今")
    print("=" * 66)

    # ── Phase 1 ──────────────────────────────────────────────────────
    print("\n[Phase 1] 从 FRED DFEDTARU 推断 2025-2026 会议决策...")
    dfedtaru  = fetch_daily_target_rate(start="2024-10-01")
    decisions = infer_decisions(dfedtaru)

    print("\n  推断结果预览：")
    print(f"  {'日期':12}  {'主席':8}  {'决策':6}  {'变动':>5}  {'目标上界':>8}")
    print("  " + "─" * 48)
    for chair, date, dec, chg, rate in decisions:
        rate_str = f"{rate:.2f}%" if rate else "?"
        print(f"  {date:12}  {chair:8}  {dec.upper():6}  {chg:>+5}bp  {rate_str:>8}")

    # ── Phase 2 ──────────────────────────────────────────────────────
    print("\n[Phase 2] 写入 fomc_history（split='final_test'）...")
    ins, skp = insert_final_test_meetings(decisions)
    print(f"  新插入：{ins} 条  |  已存在/跳过：{skp} 条")

    # ── Phase 3 ──────────────────────────────────────────────────────
    # 检查 semantic_history 覆盖情况
    conn = sqlite3.connect(DB_PATH)
    done_sem = {r[0] for r in conn.execute(
        "SELECT meeting_date FROM semantic_history WHERE meeting_date >= '2025-01-01'"
    ).fetchall()}
    conn.close()

    need_sem = [d for _, d, dec, _, _ in decisions
                if dec != "unknown" and d not in done_sem]

    if need_sem:
        print(f"\n[Phase 3] 以下 {len(need_sem)} 次会议缺少语义分析，"
              f"正在调用 agent2_v2_semantic_history.run_batch()...")
        print("  (将自动处理所有未分析的会议，包括 final_test 新记录)")
        from v2.agent2_v2_semantic_history import run_batch
        run_batch()
    else:
        print(f"\n[Phase 3] 所有最终测试集会议已有语义分析，跳过。")

    # 再次检查覆盖
    conn = sqlite3.connect(DB_PATH)
    done_sem = {r[0] for r in conn.execute(
        "SELECT meeting_date FROM semantic_history WHERE meeting_date >= '2025-01-01'"
    ).fetchall()}
    conn.close()

    covered = [d for _, d, dec, _, _ in decisions
               if dec != "unknown" and d in done_sem]
    missing = [d for _, d, dec, _, _ in decisions
               if dec != "unknown" and d not in done_sem]

    if missing:
        print(f"\n  [注意] {len(missing)} 次会议仍无 semantic_history：{missing}")
        print("  （可能是 LLM 失败或声明文本获取失败，将从评估中排除）")

    # ── Phase 4 ──────────────────────────────────────────────────────
    if not covered:
        print("\n[Phase 4] 无完整数据可评估，退出。")
        print("  提示：手动补全 semantic_history 后重新运行此脚本。")
        return

    print(f"\n[Phase 4] 融合评估（{len(covered)} 次会议）...")
    evaluate_final_test()


if __name__ == "__main__":
    main()
