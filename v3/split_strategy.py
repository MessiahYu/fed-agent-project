"""
v3/split_strategy.py
============================================================
V3 核心改进：将 2000-2024 历史会议从"时间切割"改为"分层随机 80/20 划分"。

V2 做法（时间切割）：
  train = 2000-2020 (173次)   test = 2021-2024 (32次)
  问题：时间固定效应 —— 验证集集中在疫后高通胀+快速加息期，
        与训练集的经济环境系统性不同，验证准确率无法代表真实泛化能力。

V3 做法（分层随机抽样）：
  对 2000-2024 全部 205 次会议，按 HOLD/HIKE/CUT 三类各自独立随机抽取：
    - 80% 的会议 → train
    - 20% 的会议 → test
  这样训练集和测试集都覆盖各种经济周期，消除时代分布的系统性偏差。

  2025-至今 (final_test) 保持不变，不参与重新划分。

运行方式：
  E:\\Anaconda\\envs\\fed-agent\\python.exe v3/split_strategy.py
  （加 --seed 42 可固定随机种子，加 --dry-run 只打印不写库）
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from utils.db import DB_PATH


def load_historical_meetings(conn: sqlite3.Connection) -> pd.DataFrame:
    """读取 2000-2024 全部历史会议（排除 final_test）。"""
    df = pd.read_sql_query(
        """
        SELECT id, meeting_date, actual_decision, split
        FROM fomc_history
        WHERE split IN ('train', 'test')
        ORDER BY meeting_date
        """,
        conn,
    )
    return df


def stratified_split(df: pd.DataFrame, train_ratio: float = 0.8, seed: int = 42) -> pd.DataFrame:
    """
    按 HOLD/HIKE/CUT 三类分层随机抽样，返回带新 split 列的 DataFrame。

    每一类内部随机打乱后，前 80% 分到 train，后 20% 分到 test。
    使用固定 seed 保证结果可复现。
    """
    rng = np.random.default_rng(seed)
    result_parts = []

    for decision in ["hold", "hike", "cut"]:
        subset = df[df["actual_decision"] == decision].copy()
        if subset.empty:
            continue

        # 随机打乱顺序（按 seed 固定）
        idx = rng.permutation(len(subset))
        subset = subset.iloc[idx].reset_index(drop=True)

        # 按比例切分
        n_train = int(round(len(subset) * train_ratio))
        subset["new_split"] = "test"
        subset.loc[:n_train - 1, "new_split"] = "train"

        result_parts.append(subset)

    return pd.concat(result_parts, ignore_index=True).sort_values("meeting_date")


def apply_split(df_new: pd.DataFrame, conn: sqlite3.Connection, dry_run: bool = False) -> None:
    """将新的 split 写回数据库。"""
    cur = conn.cursor()

    train_ids = df_new[df_new["new_split"] == "train"]["id"].tolist()
    test_ids  = df_new[df_new["new_split"] == "test"]["id"].tolist()

    if not dry_run:
        cur.executemany("UPDATE fomc_history SET split=? WHERE id=?",
                        [("train", i) for i in train_ids])
        cur.executemany("UPDATE fomc_history SET split=? WHERE id=?",
                        [("test",  i) for i in test_ids])
        conn.commit()


def print_summary(df_original: pd.DataFrame, df_new: pd.DataFrame) -> None:
    """打印划分前后的对比汇总。"""
    print("\n" + "=" * 65)
    print("  V3 分层随机划分汇总")
    print("=" * 65)

    print("\n【V2 划分（时间切割）】")
    for split in ["train", "test"]:
        sub = df_original[df_original["split"] == split]
        dates = pd.to_datetime(sub["meeting_date"])
        counts = sub["actual_decision"].value_counts()
        print(f"  {split:6s}: {len(sub):3d}次  "
              f"{dates.min().date()} — {dates.max().date()}  "
              f"HOLD={counts.get('hold',0)} HIKE={counts.get('hike',0)} CUT={counts.get('cut',0)}")

    print("\n【V3 划分（分层随机 80/20）】")
    for split in ["train", "test"]:
        sub = df_new[df_new["new_split"] == split]
        dates = pd.to_datetime(sub["meeting_date"])
        counts = sub["actual_decision"].value_counts()
        print(f"  {split:6s}: {len(sub):3d}次  "
              f"{dates.min().date()} — {dates.max().date()}  "
              f"HOLD={counts.get('hold',0)} HIKE={counts.get('hike',0)} CUT={counts.get('cut',0)}")

    print("\n【各类别分层抽样明细】")
    print(f"  {'类别':6}  {'总数':>5}  {'train(80%)':>10}  {'test(20%)':>9}")
    print("  " + "─" * 36)
    for decision in ["hold", "hike", "cut"]:
        total = len(df_new[df_new["actual_decision"] == decision])
        n_tr  = len(df_new[(df_new["actual_decision"] == decision) & (df_new["new_split"] == "train")])
        n_te  = len(df_new[(df_new["actual_decision"] == decision) & (df_new["new_split"] == "test")])
        print(f"  {decision.upper():6}  {total:5d}  {n_tr:10d}  {n_te:9d}")

    print("  " + "─" * 36)
    total_all = len(df_new)
    n_tr_all  = len(df_new[df_new["new_split"] == "train"])
    n_te_all  = len(df_new[df_new["new_split"] == "test"])
    print(f"  {'合计':6}  {total_all:5d}  {n_tr_all:10d}  {n_te_all:9d}")


def main():
    parser = argparse.ArgumentParser(description="V3 分层随机 80/20 数据划分")
    parser.add_argument("--seed",    type=int,  default=42,    help="随机种子（默认 42）")
    parser.add_argument("--ratio",   type=float, default=0.8,  help="训练集比例（默认 0.8）")
    parser.add_argument("--dry-run", action="store_true",      help="只打印，不写数据库")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)

    print(f"随机种子: {args.seed}  训练集比例: {args.ratio:.0%}  写库: {'否(dry-run)' if args.dry_run else '是'}")

    df_original = load_historical_meetings(conn)
    print(f"\n读取历史会议 {len(df_original)} 次（split=train 或 test）")

    df_new = stratified_split(df_original, train_ratio=args.ratio, seed=args.seed)
    print_summary(df_original, df_new)

    if not args.dry_run:
        apply_split(df_new, conn, dry_run=False)
        print(f"\n✓ 数据库已更新。train={len(df_new[df_new['new_split']=='train'])} 次，"
              f"test={len(df_new[df_new['new_split']=='test'])} 次")
    else:
        print("\n[dry-run 模式，数据库未修改]")

    conn.close()

    # 写出划分记录供审计
    out_path = _ROOT / "v3" / "v3_split_assignments.csv"
    df_new[["meeting_date", "actual_decision", "new_split"]].rename(
        columns={"new_split": "v3_split"}
    ).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n划分明细已保存至: {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
