import os
import pathlib
from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from FinMind.data import DataLoader

ETF_SOURCES = {
    "00981A": "https://github.com/4ru1013/united-etf-00981a-portfolio/blob/main/data/out/00981A_latest.csv?raw=1",
    "00992A": "https://github.com/4ru1013/capital-etf-00992a-portfolio/blob/main/data/out/00992A_latest.csv?raw=1",
}
TAIEX_SYMBOL = "TAIEX"
OUTPUT_DIR = pathlib.Path("output")
RAW_DIR = pathlib.Path("data/raw")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def normalize_code(code) -> str:
    return str(code).strip().replace(".0", "")


def read_csv_from_url(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    # utf-8-sig handles BOM from Excel-friendly CSVs.
    from io import StringIO

    return pd.read_csv(StringIO(resp.text), dtype={"code": "string"})


def load_etf_holdings() -> pd.DataFrame:
    frames = []
    for etf_code, url in ETF_SOURCES.items():
        df = read_csv_from_url(url)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "code" not in df.columns or "name" not in df.columns or "shares" not in df.columns:
            raise ValueError(f"{etf_code} latest CSV columns invalid: {list(df.columns)}")
        df["code"] = df["code"].map(normalize_code)
        df["name"] = df["name"].astype(str).str.strip()
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
        if "weight" in df.columns:
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        else:
            df["weight"] = np.nan
        df = df.sort_values(["weight", "shares"], ascending=[False, False]).reset_index(drop=True)
        df["etf_code"] = etf_code
        df["etf_rank"] = np.arange(1, len(df) + 1)
        df["is_top10"] = df["etf_rank"] <= 10
        frames.append(df[["etf_code", "code", "name", "shares", "weight", "etf_rank", "is_top10"]])
    all_holdings = pd.concat(frames, ignore_index=True)
    all_holdings.to_csv(RAW_DIR / "etf_holdings_latest.csv", index=False, encoding="utf-8-sig")
    return all_holdings


def build_candidate_pool(holdings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in holdings.groupby("code"):
        name = g["name"].dropna().astype(str).iloc[0]
        etfs = sorted(g["etf_code"].unique().tolist())
        row = {
            "code": code,
            "name": name,
            "etf_list": "+".join(etfs),
            "etf_count": len(etfs),
            "in_00981A": "00981A" in etfs,
            "in_00992A": "00992A" in etfs,
            "00981A_shares": 0,
            "00992A_shares": 0,
            "00981A_weight": np.nan,
            "00992A_weight": np.nan,
            "00981A_rank": np.nan,
            "00992A_rank": np.nan,
            "00981A_top10": False,
            "00992A_top10": False,
        }
        for _, r in g.iterrows():
            etf = r["etf_code"]
            row[f"{etf}_shares"] = int(r["shares"])
            row[f"{etf}_weight"] = r["weight"]
            row[f"{etf}_rank"] = int(r["etf_rank"])
            row[f"{etf}_top10"] = bool(r["is_top10"])
        rows.append(row)
    pool = pd.DataFrame(rows)
    pool = pool.sort_values(["etf_count", "00992A_top10", "00981A_top10", "00992A_weight", "00981A_shares"], ascending=[False, False, False, False, False]).reset_index(drop=True)
    pool.to_csv(OUTPUT_DIR / "candidate_pool.csv", index=False, encoding="utf-8-sig")
    return pool


def setup_finmind() -> DataLoader:
    api = DataLoader()
    token = os.getenv("FINMIND_TOKEN", "").strip()
    if token:
        api.login_by_token(api_token=token)
    return api


def fetch_stock_price(api: DataLoader, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["stock_id"] = df["stock_id"].astype(str)
    return df


def fetch_all_prices(pool: pd.DataFrame, days_back: int = 430) -> pd.DataFrame:
    api = setup_finmind()
    end = date.today()
    start = end - timedelta(days=days_back)
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    frames = []
    ids = pool["code"].dropna().astype(str).unique().tolist()
    # Add TAIEX benchmark for RS calculations.
    ids_with_benchmark = ids + [TAIEX_SYMBOL]
    for i, sid in enumerate(ids_with_benchmark, 1):
        print(f"[INFO] Fetch price {i}/{len(ids_with_benchmark)}: {sid}")
        try:
            df = fetch_stock_price(api, sid, start_s, end_s)
        except Exception as exc:
            print(f"[WARN] Failed to fetch {sid}: {exc}")
            continue
        if df.empty:
            print(f"[WARN] Empty price data: {sid}")
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("No price data fetched from FinMind.")
    prices = pd.concat(frames, ignore_index=True)
    prices.to_csv(RAW_DIR / "price_data.csv", index=False, encoding="utf-8-sig")
    return prices


def add_indicators_for_one(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["Trading_Volume"], errors="coerce")

    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()
    df["vol_ma20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["vol_ma20"]
    df["ret_5d"] = close / close.shift(5) - 1
    df["ret_20d"] = close / close.shift(20) - 1
    df["ret_60d"] = close / close.shift(60) - 1
    df["ret_120d"] = close / close.shift(120) - 1

    ema8 = close.ewm(span=8, adjust=False).mean()
    ema17 = close.ewm(span=17, adjust=False).mean()
    df["dif"] = ema8 - ema17
    df["macd"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["osc"] = df["dif"] - df["macd"]
    df["osc_prev"] = df["osc"].shift(1)
    df["macd_cross_up"] = (df["dif"] > df["macd"]) & (df["dif"].shift(1) <= df["macd"].shift(1))
    df["osc_turn_positive"] = (df["osc"] > 0) & (df["osc_prev"] <= 0)
    df["above_ma20"] = close > df["ma20"]
    df["dist_ma20_pct"] = close / df["ma20"] - 1
    df["high_20d"] = close.rolling(20).max()
    df["is_20d_high"] = close >= df["high_20d"]
    return df


def build_indicators(prices: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for sid, g in prices.groupby("stock_id"):
        pieces.append(add_indicators_for_one(g))
    ind = pd.concat(pieces, ignore_index=True)
    latest = ind.sort_values("date").groupby("stock_id").tail(1).copy()

    bench = latest[latest["stock_id"] == TAIEX_SYMBOL]
    if bench.empty:
        raise RuntimeError("Missing TAIEX benchmark; cannot calculate RS.")
    bench_ret20 = float(bench["ret_20d"].iloc[0])
    bench_ret60 = float(bench["ret_60d"].iloc[0])

    latest = latest[latest["stock_id"] != TAIEX_SYMBOL].copy()
    latest["rs20"] = latest["ret_20d"] - bench_ret20
    latest["rs60"] = latest["ret_60d"] - bench_ret60
    latest["rs_score"] = 0.4 * latest["rs20"] + 0.6 * latest["rs60"]
    latest["rs_rank"] = latest["rs_score"].rank(pct=True, ascending=True) * 100
    latest["rs_rank"] = latest["rs_rank"].round(1)

    latest = latest.rename(columns={"stock_id": "code"})
    keep_cols = [
        "code", "date", "close", "Trading_Volume", "ma5", "ma20", "ma60", "vol_ma20", "volume_ratio",
        "ret_5d", "ret_20d", "ret_60d", "ret_120d", "rs20", "rs60", "rs_score", "rs_rank",
        "dif", "macd", "osc", "osc_prev", "macd_cross_up", "osc_turn_positive",
        "above_ma20", "dist_ma20_pct", "is_20d_high"
    ]
    latest = latest[keep_cols]
    merged = pool.merge(latest, on="code", how="left")
    return merged


def add_research_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rs_component = out["rs_rank"].fillna(0)
    etf_component = out["etf_count"].fillna(0).clip(upper=2) / 2 * 100
    top10_component = ((out["00981A_top10"].fillna(False)) | (out["00992A_top10"].fillna(False))).astype(int) * 100
    timing_component = (
        out["above_ma20"].fillna(False).astype(int) * 35
        + out["osc_turn_positive"].fillna(False).astype(int) * 35
        + out["macd_cross_up"].fillna(False).astype(int) * 30
    )
    out["research_score"] = (0.45 * rs_component + 0.20 * etf_component + 0.20 * top10_component + 0.15 * timing_component).round(1)
    return out.sort_values(["research_score", "rs_rank", "etf_count"], ascending=[False, False, False]).reset_index(drop=True)


def export_excel(screen: pd.DataFrame, pool: pd.DataFrame, holdings: pd.DataFrame) -> pathlib.Path:
    today = date.today().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"daily_screen_{today}.xlsx"
    latest_path = OUTPUT_DIR / "daily_screen_latest.xlsx"

    watch_cols = [
        "code", "name", "research_score", "rs_rank", "rs20", "rs60", "etf_list", "etf_count",
        "00981A_top10", "00992A_top10", "00981A_rank", "00992A_rank", "00992A_weight",
        "date", "close", "ret_5d", "ret_20d", "ret_60d", "volume_ratio",
        "ma5", "ma20", "ma60", "above_ma20", "dist_ma20_pct", "is_20d_high",
        "dif", "macd", "osc", "macd_cross_up", "osc_turn_positive"
    ]
    watch_cols = [c for c in watch_cols if c in screen.columns]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        screen[watch_cols].to_excel(writer, sheet_name="Ranking", index=False)
        pool.to_excel(writer, sheet_name="Candidate_Pool", index=False)
        holdings.to_excel(writer, sheet_name="ETF_Holdings", index=False)
        screen.to_excel(writer, sheet_name="Full_Data", index=False)

    # Also write stable latest file.
    with pd.ExcelWriter(latest_path, engine="openpyxl") as writer:
        screen[watch_cols].to_excel(writer, sheet_name="Ranking", index=False)
        pool.to_excel(writer, sheet_name="Candidate_Pool", index=False)
        holdings.to_excel(writer, sheet_name="ETF_Holdings", index=False)
        screen.to_excel(writer, sheet_name="Full_Data", index=False)

    screen.to_csv(OUTPUT_DIR / "daily_screen_latest.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] Exported {out_path}")
    print(f"[OK] Exported {latest_path}")
    return out_path


def main() -> None:
    ensure_dirs()
    holdings = load_etf_holdings()
    pool = build_candidate_pool(holdings)
    prices = fetch_all_prices(pool)
    screen = build_indicators(prices, pool)
    screen = add_research_score(screen)
    export_excel(screen, pool, holdings)


if __name__ == "__main__":
    main()
