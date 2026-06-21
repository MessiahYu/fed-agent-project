"""
v5/weight_optimizer_v5.py
V5 = V4（LLM 滞后一期）+ V3（分层随机 80/20 划分）同时修复两个问题。

核心思路：
  先按时间顺序计算 hawkish_score_lag1（保证 lag 的经济含义正确），
  再对历史会议做分层随机 80/20 划分（消除时间固定效应）。
  两步顺序不可颠倒。

运行方式：
  E:\\Anaconda\\envs\\fed-agent\\python.exe v5/weight_optimizer_v5.py
"""

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from v2.weight_optimizer import (
    load_feature_matrix, compute_var_signals,
    add_numeric_signals, fuse_predict
)
from v2.backtest import fetch_full_macro_series

_cfg_path = _ROOT / "config.yml"

# ── 参数搜索空间（与 V2/V3/V4 完全相同）──────────────────────────────────────
W_GRID    = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
HIKE_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
CUT_GRID  = [-0.20, -0.25, -0.30, -0.35, -0.40]

FINAL_TEST_START = "2025-01-01"
TRAIN_RATIO      = 0.80
SEED             = 42


# ── V5 特征矩阵构建 ───────────────────────────────────────────────────────────

def _stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """对历史会议按 HOLD/HIKE/CUT 分层随机 80/20 划分，返回带 split 列的 df。"""
    rng = np.random.default_rng(SEED)
    df = df.copy()
    df["split"] = "train"
    for cls in ["hold", "hike", "cut"]:
        idx = df[df["actual_decision"].str.lower() == cls].index.tolist()
        rng.shuffle(idx)
        n_train = int(len(idx) * TRAIN_RATIO)
        df.loc[idx[n_train:], "split"] = "test"
    return df


def load_feature_matrix_v5() -> pd.DataFrame:
    """
    V5 特征矩阵，两步修复同时生效：
    Step 1: 按时间顺序对 LLM 分做 shift(1) —— 修复同期信息泄露（V4）
    Step 2: 对历史会议做分层随机 80/20 —— 修复时间固定效应（V3）
    """
    df = load_feature_matrix()
    df = df.sort_values("meeting_ts").reset_index(drop=True)

    # Step 1: LLM 信号滞后一期（在时间顺序上计算，与划分无关）
    for col in ["hawkish_score", "inflation_concern", "growth_concern"]:
        if col in df.columns:
            df[col] = df[col].shift(1)
    n_before = len(df)
    df = df.dropna(subset=["hawkish_score"]).reset_index(drop=True)
    print(f"  [V5] LLM 信号已滞后一期，丢弃首行（无前驱），剩 {len(df)} 次（原 {n_before} 次）")

    # Step 2: 分离 final_test（2025 年以后）
    final_mask = df["meeting_date"] >= FINAL_TEST_START
    df_final = df[final_mask].copy()
    df_hist  = df[~final_mask].copy().reset_index(drop=True)

    # Step 3: 对历史会议做分层随机 80/20（只加标签，不写数据库）
    df_hist = _stratified_split(df_hist)

    # Step 4: 合并，final_test 保持不变
    df_final = df_final.copy()
    df_final["split"] = "final_test"
    df_out = pd.concat([df_hist, df_final], ignore_index=True)
    df_out = df_out.sort_values("meeting_ts").reset_index(drop=True)
    return df_out


# ── 网格搜索 ──────────────────────────────────────────────────────────────────

def run_grid_search(df_train: pd.DataFrame, df_macro: pd.DataFrame) -> dict:
    var_sigs = compute_var_signals(df_train, df_macro)
    df_t = add_numeric_signals(df_train.copy(), var_sigs)

    llm_v = df_t["hawkish_score"].values.astype(float)
    var_v = df_t["var_num"].values.astype(float)
    mkt_v = df_t["market_num"].values.astype(float)
    pol_v = df_t["pol_hawkish"].values.astype(float)
    actual = df_t["actual_decision"].str.lower().values

    combos = [
        (wl, wv, wm, wp, ht, ct)
        for wl in W_GRID for wv in W_GRID for wm in W_GRID for wp in W_GRID
        for ht in HIKE_GRID for ct in CUT_GRID
        if abs(wl + wv + wm + wp - 1.0) < 1e-9
    ]
    print(f"  [网格搜索] 共 {len(combos)} 个参数组合...")

    best = dict(acc=-1)
    for wl, wv, wm, wp, ht, ct in combos:
        pred = fuse_predict(llm_v, var_v, mkt_v, pol_v, wl, wv, wm, wp, ht, ct)
        acc = (pred == actual).mean()
        if acc > best["acc"]:
            best = dict(acc=acc, w_llm=wl, w_var=wv, w_market=wm, w_pol=wp,
                        hike_thr=ht, cut_thr=ct)
    return best, df_t


def evaluate_split(df: pd.DataFrame, df_macro: pd.DataFrame, best: dict,
                   label: str) -> tuple:
    var_sigs = compute_var_signals(df, df_macro)
    df_s = add_numeric_signals(df.copy(), var_sigs)
    pred = fuse_predict(
        df_s["hawkish_score"].values.astype(float),
        df_s["var_num"].values.astype(float),
        df_s["market_num"].values.astype(float),
        df_s["pol_hawkish"].values.astype(float),
        best["w_llm"], best["w_var"], best["w_market"], best["w_pol"],
        best["hike_thr"], best["cut_thr"]
    )
    actual = df_s["actual_decision"].str.lower().values
    acc = (pred == actual).mean()
    baseline = (actual == "hold").mean()
    return acc, baseline, pred, actual


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("  FedWatch AI V5 权重优化器")
    print("  V5 = LLM 滞后一期（V4）+ 分层随机 80/20（V3）")
    print("=" * 62)

    # 构建特征矩阵
    print("\n正在构建 V5 特征矩阵...")
    df_all = load_feature_matrix_v5()

    df_train = df_all[df_all["split"] == "train"].copy().reset_index(drop=True)
    df_test  = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)

    print(f"\n  数据划分：训练集 {len(df_train)} 次 / 测试集 {len(df_test)} 次")
    for cls in ["hold", "hike", "cut"]:
        nt = (df_train["actual_decision"].str.lower() == cls).sum()
        nv = (df_test["actual_decision"].str.lower()  == cls).sum()
        print(f"    {cls.upper():5}：训练 {nt:3}  测试 {nv:3}")

    # 拉取宏观数据
    print("\n正在拉取宏观数据...")
    df_macro = fetch_full_macro_series()

    # 网格搜索
    print("\n【训练集网格搜索】")
    best, df_train_sig = run_grid_search(df_train, df_macro)
    train_acc = best["acc"]
    print(f"  训练集最优准确率：{train_acc:.1%}")
    print(f"  最优权重：w_llm={best['w_llm']} w_var={best['w_var']} "
          f"w_market={best['w_market']} w_pol={best['w_pol']}")
    print(f"  最优阈值：hike>{best['hike_thr']:+.2f}  cut<{best['cut_thr']:+.2f}")

    # 测试集评估
    val_acc, val_baseline, pred_test, actual_test = evaluate_split(
        df_test, df_macro, best, "测试集"
    )
    n_test = len(df_test)
    print(f"\n  测试集基线（HOLD）：{val_baseline:.1%}")
    print(f"  测试集准确率：      {val_acc:.1%}  ({int(round(val_acc*n_test))}/{n_test})")
    print(f"  超越基线：          {val_acc - val_baseline:+.1%}")

    # 各类别测试集明细
    print("\n  【测试集各类别表现】")
    for cls in ["hold", "hike", "cut"]:
        mask = actual_test == cls
        if mask.sum() > 0:
            ok = (pred_test[mask] == actual_test[mask]).sum()
            print(f"    {cls.upper():5}：{ok}/{mask.sum()}  "
                  f"({'✓ 全部识别' if ok==mask.sum() else f'漏判 {mask.sum()-ok} 次'})")

    # 与 V3/V4 对比
    cfg_now = yaml.safe_load(open(str(_cfg_path), encoding="utf-8"))
    print("\n  【与其他版本准确率对比（历史测试集）】")
    print(f"    V3（随机划分，同期 LLM）：{cfg_now['v3_optimal_weights']['val_acc']:.1%}（基线 {cfg_now['v3_optimal_weights']['val_baseline']:.1%}）")
    print(f"    V4（时间切割，滞后 LLM）：{cfg_now['v4_optimal_weights']['val_acc']:.1%}（基线 56.2%）")
    print(f"    V5（随机划分，滞后 LLM）：{val_acc:.1%}（基线 {val_baseline:.1%}）  ← 本次")

    # 写入 config.yml
    cfg_now["v5_optimal_weights"] = {
        "w_llm":        float(best["w_llm"]),
        "w_var":        float(best["w_var"]),
        "w_market":     float(best["w_market"]),
        "w_pol":        float(best["w_pol"]),
        "hike_thr":     float(best["hike_thr"]),
        "cut_thr":      float(best["cut_thr"]),
        "train_acc":    float(round(train_acc, 4)),
        "val_acc":      float(round(val_acc, 4)),
        "val_baseline": float(round(val_baseline, 4)),
        "split_method": "stratified_random_8020_llm_lag1",
        "llm_lag":      1,
        "seed":         42,
        "optimized_at": "2026-06-21",
    }
    with open(str(_cfg_path), "w", encoding="utf-8") as f:
        yaml.dump(cfg_now, f, allow_unicode=True, sort_keys=False)
    print(f"\n  最优权重已写入 config.yml → v5_optimal_weights")


if __name__ == "__main__":
    main()
