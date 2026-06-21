"""
分析：6012个参数组合在验证集上的准确率分布
回答问题：65.6%是训练集优化能达到的验证集最大值吗？
"""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from itertools import product

from v2.weight_optimizer import (
    load_feature_matrix, compute_var_signals, add_numeric_signals, fuse_predict
)
from v2.backtest import fetch_full_macro_series

# ── 加载数据（复用优化器的特征矩阵）──────────────────────────────────
df_all = load_feature_matrix()
df_tr  = df_all[df_all["split"] == "train"].copy().reset_index(drop=True)
df_val = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)

print("拉取宏观数据...")
df_macro = fetch_full_macro_series()

print(f"计算训练集 VAR 信号（{len(df_tr)} 次）...")
df_tr = add_numeric_signals(df_tr, compute_var_signals(df_tr, df_macro))

print(f"计算验证集 VAR 信号（{len(df_val)} 次）...")
df_val = add_numeric_signals(df_val, compute_var_signals(df_val, df_macro))

# ── 向量化信号 ────────────────────────────────────────────────────────
tr_llm = df_tr["hawkish_score"].values.astype(float)
tr_var = df_tr["var_num"].values.astype(float)
tr_mkt = df_tr["market_num"].values.astype(float)
tr_pol = df_tr["pol_hawkish"].values.astype(float)
tr_lbl = df_tr["label"].values

vl_llm = df_val["hawkish_score"].values.astype(float)
vl_var = df_val["var_num"].values.astype(float)
vl_mkt = df_val["market_num"].values.astype(float)
vl_pol = df_val["pol_hawkish"].values.astype(float)
vl_lbl = df_val["label"].values

# ── 网格（与优化器完全一致）──────────────────────────────────────────
grid_w_llm    = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
grid_w_var    = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
grid_w_market = [0.05, 0.10, 0.15, 0.20]
grid_hike_thr = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
grid_cut_thr  = [-0.35, -0.30, -0.25, -0.20, -0.15, -0.10]

# ── 遍历全部组合，同时记录训练集和验证集准确率 ──────────────────────
print("网格搜索中（同时计算训练集和验证集）...")
records = []
for w_llm, w_var, w_market in product(grid_w_llm, grid_w_var, grid_w_market):
    w_pol = round(1.0 - w_llm - w_var - w_market, 4)
    if w_pol < 0:
        continue
    tr_fusion = w_llm*tr_llm + w_var*tr_var + w_market*tr_mkt + w_pol*tr_pol
    vl_fusion = w_llm*vl_llm + w_var*vl_var + w_market*vl_mkt + w_pol*vl_pol

    for h_thr, c_thr in product(grid_hike_thr, grid_cut_thr):
        if c_thr >= h_thr:
            continue
        tr_pred = np.where(tr_fusion > h_thr, "hike",
                  np.where(tr_fusion < c_thr, "cut", "hold"))
        vl_pred = np.where(vl_fusion > h_thr, "hike",
                  np.where(vl_fusion < c_thr, "cut", "hold"))

        records.append({
            "w_llm": w_llm, "w_var": w_var, "w_market": w_market, "w_pol": w_pol,
            "hike_thr": h_thr, "cut_thr": c_thr,
            "train_acc": (tr_pred == tr_lbl).mean(),
            "val_acc":   (vl_pred == vl_lbl).mean(),
        })

df = pd.DataFrame(records)
print(f"有效参数组合：{len(df)} 个\n")

# ── 核心问题：验证集准确率的分布 ─────────────────────────────────────
print("=" * 60)
print("  问题：65.6% 是训练集优化能达到的验证集最大值吗？")
print("=" * 60)

# 1. 训练集最优组合对应的验证集准确率（我们目前的做法）
best_train = df.sort_values("train_acc", ascending=False).iloc[0]
print(f"\n① 按训练集准确率排序第1名 → 验证集准确率：{best_train['val_acc']:.1%}")
print(f"   训练集准确率：{best_train['train_acc']:.1%}")

# 2. 如果直接按验证集准确率排序，最高能到多少？
best_val = df.sort_values("val_acc", ascending=False).iloc[0]
print(f"\n② 直接最大化验证集准确率的理论上界：{best_val['val_acc']:.1%}")
print(f"   对应训练集准确率：{best_val['train_acc']:.1%}")
print(f"   最优参数：w_llm={best_val['w_llm']}, w_var={best_val['w_var']}, "
      f"w_market={best_val['w_market']}, w_pol={best_val['w_pol']}, "
      f"hike_thr={best_val['hike_thr']}, cut_thr={best_val['cut_thr']}")

# 3. 验证集准确率分布
print(f"\n③ 全部 {len(df)} 个组合的验证集准确率分布：")
vc = df["val_acc"].value_counts().sort_index(ascending=False)
for acc, cnt in vc.items():
    bar = "█" * int(cnt / len(df) * 100)
    pct = cnt / len(df) * 100
    marker = " ← 我们目前的做法" if abs(acc - best_train["val_acc"]) < 0.001 else ""
    marker2 = " ← 理论上界（直接优化验证集）" if abs(acc - best_val["val_acc"]) < 0.001 else ""
    print(f"   {acc:.1%}  {cnt:4d}组合  {pct:5.1f}%  {bar}{marker}{marker2}")

# 4. 训练集准确率与验证集准确率的相关性
corr = df[["train_acc", "val_acc"]].corr().iloc[0, 1]
print(f"\n④ 训练集准确率 vs 验证集准确率 相关系数：{corr:.3f}")

# 5. 训练集Top20的验证集准确率范围
top20_val = df.nlargest(20, "train_acc")["val_acc"]
print(f"\n⑤ 训练集Top20组合对应的验证集准确率：")
print(f"   最高 {top20_val.max():.1%}  最低 {top20_val.min():.1%}  "
      f"平均 {top20_val.mean():.1%}")
