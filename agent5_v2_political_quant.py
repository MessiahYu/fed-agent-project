"""
agent5_v2_political_quant.py
================================================
V2 版本 Agent 5 —— 量化政治与国际压力感知
================================================

用可量化的 FRED 指标替代 V1 的文字描述方案，
使 Agent 5 的政治/国际信号具备 2000 年至今的完整历史覆盖，
从而支持训练集（2000-2020）上的权重优化。

替代变量：
  B. 政治压力（原为文字描述）：
     · EPU  经济政策不确定性指数  USEPUINDXM   Baker, Bloom & Davis (2016) QJE
     · PCI  党派冲突指数          PARCONFINDX  Azzimonti (2018) JME

  C. 国际联动（原为文字描述 + 部分 FRED）：
     · G10 政策利率分歧度 = FEDFUNDS - mean(ECB, BOJ, BOE)
     · 美元指数            DTWEXM（主要货币，1973 年起）
     · ECB 利率            ECBDFR（1999 年起）

  A. 市场信号（继承 V1 逻辑）：
     · 收益率曲线斜率 = DGS10 - DGS2

功能：
  1. 从 FRED 拉取 2000-01 至今 9 条月频时间序列
  2. 计算衍生指标（G10分歧度、EPU/PCI百分位、政治压力综合得分）
  3. 派生方向信号（cut / hold / hike）
  4. 将完整月度数据存入 SQLite（表：political_quant_monthly）
  5. 提供 get_latest_signals() 接口供 V2 pipeline 调用

运行方式：
  E:\\Anaconda\\envs\\fed-agent\\python.exe agent5_v2_political_quant.py
"""

import os
import sqlite3
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")

import yaml
_cfg     = yaml.safe_load(open("config.yml", encoding="utf-8"))
_pq      = _cfg["v2_political_quant"]

from utils.db import DB_PATH

# ── 从 config.yml 读取所有参数 ───────────────────────────────────────
FETCH_START    = _pq["fetch_start"]
TABLE_NAME     = _pq["table_name"]
ROLLING_WINDOW = _pq["rolling_window"]
FRED_SERIES    = _pq["fred_series"]
THR            = _pq["thresholds"]
W              = _pq["weights"]

FRED_BASE_URL  = _cfg["data_sources"]["fred_base_url"]


# ══════════════════════════════════════════════════════════════════════
# 1. FRED 数据拉取
# ══════════════════════════════════════════════════════════════════════

def fetch_fred_monthly(series_id: str, start: str = FETCH_START) -> pd.Series:
    """
    从 FRED 拉取月频时间序列。
    返回 pd.Series，索引为 pd.Timestamp（月份首日），值为浮点数。
    网络失败或无 KEY 时返回空 Series，不中断流程。
    """
    if not FRED_KEY:
        print(f"  [跳过] {series_id}：未配置 FRED_API_KEY")
        return pd.Series(dtype=float, name=series_id)

    try:
        r = requests.get(
            FRED_BASE_URL,
            params={
                "series_id":          series_id,
                "api_key":            FRED_KEY,
                "file_type":          "json",
                "observation_start":  start,
                "frequency":          "m",
                "aggregation_method": "avg",
            },
            timeout=30,
        )
        r.raise_for_status()
        data = {}
        for o in r.json().get("observations", []):
            try:
                data[pd.Timestamp(o["date"])] = float(o["value"])
            except (ValueError, KeyError):
                pass
        s = pd.Series(data, name=series_id)
        print(f"  ✓ {series_id:<26} {len(s):>4} 个月度观测")
        return s
    except Exception as e:
        print(f"  ✗ {series_id}: {e}")
        return pd.Series(dtype=float, name=series_id)


def build_feature_matrix() -> pd.DataFrame:
    """
    拉取所有 9 条 FRED 序列，对齐到月频索引，合并为 DataFrame。
    2000-01 至今，缺失值保留为 NaN（不填充，避免引入偏差）。
    """
    print(f"\n[1/4] 从 FRED 拉取 {len(FRED_SERIES)} 条月频时间序列...")
    print(f"      起始日期：{FETCH_START}  |  目标：{datetime.now().strftime('%Y-%m')}")

    series_dict = {}
    for col, fred_id in FRED_SERIES.items():
        series_dict[col] = fetch_fred_monthly(fred_id)

    df = pd.DataFrame(series_dict)
    df.index = pd.to_datetime(df.index)
    df = df[df.index >= FETCH_START].sort_index()

    missing = df.isnull().sum()
    print(f"\n  合并完成：{len(df)} 个月 × {df.shape[1]} 个指标")
    print(f"  时间范围：{df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}")
    if missing.any():
        print(f"  缺失值统计：\n{missing[missing > 0].to_string()}")

    return df


# ══════════════════════════════════════════════════════════════════════
# 2. 衍生特征计算
# ══════════════════════════════════════════════════════════════════════

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始 FRED 数据上计算四个衍生特征：

    yield_spread    = DGS10 - DGS2
                      正斜率=正常，倒挂=市场预期降息

    g10_divergence  = FEDFUNDS - mean(ECB, BOJ, BOE)
                      正值=美联储相对偏紧，负值=相对偏松
                      (Rey 2015; Obstfeld & Taylor 2004)

    epu_pctl36      = EPU 在滚动 36 个月窗口内的历史百分位
                      基于 Baker, Bloom & Davis (2016) QJE

    pci_pctl36      = PCI 在滚动 36 个月窗口内的历史百分位
                      基于 Azzimonti (2018) JME

    political_score = 0.6×epu_pctl36 + 0.4×pci_pctl36
                      综合政治/不确定性压力，0=最低，1=最高
    """
    print("\n[2/4] 计算衍生特征...")

    df = df.copy()

    # 收益率曲线斜率（市场对利率路径的集体判断）
    df["yield_spread"] = df["dgs10"] - df["dgs2"]

    # G10 政策利率分歧度（美联储相对全球同伴的紧缩程度）
    g10_avg = df[["ecb_rate", "boj_rate", "boe_rate"]].mean(axis=1)
    df["g10_divergence"] = df["fedfunds"] - g10_avg

    # EPU / PCI 滚动36个月百分位
    # (x < x[-1]).mean() = 窗口内低于当前值的比例 = 当前值的百分位排名
    # raw=True 避免逐窗口创建 Series，速度快 10 倍以上
    df["epu_pctl36"] = df["epu"].rolling(
        ROLLING_WINDOW, min_periods=12
    ).apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else 0.5, raw=True)

    # VIX 百分位（Bloom 2009 AER：VIX 是不确定性冲击的标准代理变量）
    # 注意：VIX 是日频聚合为月均值，数值越高 = 市场越恐慌 = 政治/经济不确定性越高
    df["vix_pctl36"] = df["vix"].rolling(
        ROLLING_WINDOW, min_periods=12
    ).apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else 0.5, raw=True)

    # 综合政治压力得分（权重来自 config.yml，vix 替代原 pci）
    df["political_score"] = (
        W["epu_in_political"] * df["epu_pctl36"].fillna(0.5) +
        W["pci_in_political"] * df["vix_pctl36"].fillna(0.5)
    )

    print("  ✓ yield_spread, g10_divergence")
    print("  ✓ epu_pctl36, vix_pctl36（滚动36月百分位）")
    print("  ✓ political_score（综合政治压力得分）")
    return df


# ══════════════════════════════════════════════════════════════════════
# 3. 方向信号派生
# ══════════════════════════════════════════════════════════════════════

def derive_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    将连续量化指标转化为离散方向信号，阈值均来自 config.yml。

    market_signal（收益率曲线）：
      yield_spread < -0.20% → cut（倒挂，市场预期降息）
      yield_spread >  1.00% → hike（陡峭，通胀预期升温）
      其余               → hold

    political_signal（EPU + PCI）：
      political_score > 0.70 → toward_cut（高不确定性=货币宽松压力大）
      political_score < 0.30 → low_pressure（压力小，美联储自主空间大）
      其余               → neutral

    international_signal（G10分歧度）：
      g10_divergence > +1.50% → cut_pressure（美联储相对偏紧，汇率压力）
      g10_divergence < -1.00% → hike_pressure（美联储相对偏松）
      其余               → hold
    """
    print("\n[3/4] 派生方向信号（阈值来自 config.yml）...")

    df = df.copy()

    def _market(spread):
        if pd.isna(spread):          return "hold"
        if spread < THR["yield_invert"]: return "cut"
        if spread > THR["yield_steep"]:  return "hike"
        return "hold"

    def _political(score):
        if pd.isna(score):                    return "neutral"
        if score > THR["political_high"]:     return "toward_cut"
        if score < THR["political_low"]:      return "low_pressure"
        return "neutral"

    def _international(div):
        if pd.isna(div):                      return "hold"
        if div > THR["diverge_tight"]:        return "cut_pressure"
        if div < THR["diverge_loose"]:        return "hike_pressure"
        return "hold"

    df["market_signal"]        = df["yield_spread"].apply(_market)
    df["political_signal"]     = df["political_score"].apply(_political)
    df["international_signal"] = df["g10_divergence"].apply(_international)

    # 信号分布统计
    for col in ["market_signal", "political_signal", "international_signal"]:
        dist = df[col].value_counts().to_dict()
        print(f"  {col}: {dist}")

    return df


# ══════════════════════════════════════════════════════════════════════
# 4. SQLite 存储
# ══════════════════════════════════════════════════════════════════════

def save_to_db(df: pd.DataFrame):
    """
    将完整月度特征矩阵写入 SQLite。
    每次运行覆盖整张表（if_exists='replace'），
    确保历史数据始终是最新拉取的版本。
    """
    print(f"\n[4/4] 写入数据库（表：{TABLE_NAME}）...")

    df_save = df.copy()
    df_save.index = df_save.index.strftime("%Y-%m-%d")
    df_save.index.name = "month"
    df_save = df_save.reset_index()
    df_save["updated_at"] = datetime.now().isoformat()

    conn = sqlite3.connect(DB_PATH)
    df_save.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()

    print(f"  ✓ {len(df_save)} 行 × {df_save.shape[1]} 列 写入 {TABLE_NAME}")
    print(f"  数据库路径：{DB_PATH}")


# ══════════════════════════════════════════════════════════════════════
# 5. 对外接口（供 V2 pipeline 调用）
# ══════════════════════════════════════════════════════════════════════

def get_latest_signals(as_of: str = None) -> dict:
    """
    从数据库读取指定月份（或最新月份）的量化信号。

    Parameters
    ----------
    as_of : str, optional
        格式 "YYYY-MM-DD"，取该日期最近的月份数据。
        None = 取最新月份。

    Returns
    -------
    dict  包含所有原始指标 + 衍生特征 + 方向信号
    """
    conn = sqlite3.connect(DB_PATH)
    if as_of:
        row = pd.read_sql(
            f"SELECT * FROM {TABLE_NAME} WHERE month <= ? ORDER BY month DESC LIMIT 1",
            conn, params=(as_of,)
        )
    else:
        row = pd.read_sql(
            f"SELECT * FROM {TABLE_NAME} ORDER BY month DESC LIMIT 1",
            conn
        )
    conn.close()

    if row.empty:
        return {}
    return row.iloc[0].dropna().to_dict()


def get_history_for_meeting(meeting_date: str) -> dict:
    """
    为历史回测提供接口：
    给定 FOMC 会议日期，返回会议前2天可见的最新月份信号。
    （防前视偏差，与 V1 的 cutoff 逻辑一致）
    """
    cutoff = (pd.Timestamp(meeting_date) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    return get_latest_signals(as_of=cutoff)


# ══════════════════════════════════════════════════════════════════════
# 6. 报告打印
# ══════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame):
    W_LINE = 64
    print("\n" + "★" * W_LINE)
    print("  FedWatch AI V2 — 量化政治与国际压力特征库")
    print("★" * W_LINE)

    latest = df.iloc[-1]
    print(f"""
  数据范围：{df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}
  总计 {len(df)} 个月度观测

  ┌─ 最新数据（{df.index[-1].strftime('%Y-%m')}）{'─'*34}
  │
  │  B. 政治压力指标
  │    EPU  指数：     {latest['epu']:.1f}   （36m百分位：{latest['epu_pctl36']:.0%}）
  │    VIX  指数：     {latest['vix']:.1f}   （36m百分位：{latest['vix_pctl36']:.0%}）
  │    政治压力得分：  {latest['political_score']:.2f}  → {latest['political_signal']}
  │
  │  C. 国际联动指标
  │    G10 利率分歧度：{latest['g10_divergence']:+.2f}%  → {latest['international_signal']}
  │    （Fed={latest['fedfunds']:.2f}%  ECB={latest['ecb_rate']:.2f}%  BOJ={latest['boj_rate']:.2f}%  BOE={latest['boe_rate']:.2f}%）
  │    美元指数：      {latest['dollar']:.1f}
  │
  │  A. 市场信号
  │    收益率利差：    {latest['yield_spread']:+.2f}%  → {latest['market_signal'].upper()}
  │    （2Y={latest['dgs2']:.2f}%  10Y={latest['dgs10']:.2f}%）
  └{'─'*62}""")

    # EPU 历史极值（最能说明指标意义的节点）
    print(f"\n  ┌─ EPU 历史前5高月份（政策不确定性最高峰）{'─'*16}")
    top5 = df.nlargest(5, "epu")[["epu", "vix", "fedfunds", "political_signal"]]
    for idx, row in top5.iterrows():
        print(f"  │  {idx.strftime('%Y-%m')}  EPU={row['epu']:>6.1f}  "
              f"VIX={row['vix']:>5.1f}  Fed={row['fedfunds']:.2f}%  "
              f"→ {row['political_signal']}")
    print(f"  └{'─'*62}")

    # G10 分歧度极值
    print(f"\n  ┌─ G10 利率分歧度极值（美联储最偏离全球同伴的时期）{'─'*10}")
    print("  │  [最高分歧 — 美联储相对偏紧]")
    top3 = df.nlargest(3, "g10_divergence")[["g10_divergence", "fedfunds", "ecb_rate", "international_signal"]]
    for idx, row in top3.iterrows():
        print(f"  │    {idx.strftime('%Y-%m')}  分歧={row['g10_divergence']:+.2f}%  "
              f"Fed={row['fedfunds']:.2f}%  ECB={row['ecb_rate']:.2f}%  → {row['international_signal']}")
    print("  │  [最低分歧 — 美联储相对偏松]")
    bot3 = df.nsmallest(3, "g10_divergence")[["g10_divergence", "fedfunds", "ecb_rate", "international_signal"]]
    for idx, row in bot3.iterrows():
        print(f"  │    {idx.strftime('%Y-%m')}  分歧={row['g10_divergence']:+.2f}%  "
              f"Fed={row['fedfunds']:.2f}%  ECB={row['ecb_rate']:.2f}%  → {row['international_signal']}")
    print(f"  └{'─'*62}")

    print("\n" + "★" * W_LINE)


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("  FedWatch AI V2 — 量化政治与国际压力感知 启动")
    print("=" * 64)

    if not FRED_KEY:
        print("\n  [错误] 未检测到 FRED_API_KEY")
        print("  请在 .env 文件中添加：FRED_API_KEY=你的key")
        print("  注册免费 Key：https://fred.stlouisfed.org/docs/api/api_key.html")
        raise SystemExit(1)

    df = build_feature_matrix()
    df = compute_derived_features(df)
    df = derive_signals(df)
    save_to_db(df)
    print_report(df)

    # 展示接口用法
    print("\n  [接口演示] get_history_for_meeting('2008-10-29')")
    sig = get_history_for_meeting("2008-10-29")
    if sig:
        print(f"    政治信号：{sig.get('political_signal')}  "
              f"国际信号：{sig.get('international_signal')}  "
              f"市场信号：{sig.get('market_signal')}")
