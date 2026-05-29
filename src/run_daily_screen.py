import pathlib
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

ETF_SOURCES = {
    "00981A": "https://github.com/4ru1013/united-etf-00981a-portfolio/blob/main/data/out/00981A_latest.csv?raw=1",
    "00992A": "https://github.com/4ru1013/capital-etf-00992a-portfolio/blob/main/data/out/00992A_latest.csv?raw=1",
}

TAIEX_SYMBOL = "TAIEX"
OUTPUT_DIR = pathlib.Path("output")
RAW_DIR = pathlib.Path("data/raw")
PREVIOUS_HOLDINGS_PATH = RAW_DIR / "etf_holdings_latest.csv"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def normalize_code(code) -> str:
    return str(code).strip().replace(".0", "")


def read_csv_from_url(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(resp.text), dtype={"code": "string"})


def load_previous_holding_codes() -> dict[str, set[str]]:
    if not PREVIOUS_HOLDINGS_PATH.exists():
        return {etf: set() for etf in ETF_SOURCES}
    try:
        prev = pd.read_csv(PREVIOUS_HOLDINGS_PATH, dtype={"code": "string"})
    except Exception as exc:
        print(f"[WARN] Failed to read previous holdings: {exc}")
        return {etf: set() for etf in ETF_SOURCES}
    if not {"etf_code", "code"}.issubset(set(prev.columns)):
        print("[WARN] Previous holdings format invalid; new-holding flags disabled for this run.")
        return {etf: set() for etf in ETF_SOURCES}
    prev["code"] = prev["code"].map(normalize_code)
    return {
        etf: set(prev.loc[prev["etf_code"].astype(str) == etf, "code"].dropna().astype(str))
        for etf in ETF_SOURCES
    }


def load_etf_holdings() -> pd.DataFrame:
    previous_codes = load_previous_holding_codes()
    frames = []
    for etf_code, url in ETF_SOURCES.items():
        df = read_csv_from_url(url)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "code" not in df.columns or "name" not in df.columns or "shares" not in df.columns:
            raise ValueError(f"{etf_code} latest CSV columns invalid: {list(df.columns)}")
        df["code"] = df["code"].map(normalize_code)
        df["name"] = df["name"].astype(str).str.strip()
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce") if "weight" in df.columns else np.nan
        df = df.sort_values(["weight", "shares"], ascending=[False, False]).reset_index(drop=True)
        df["etf_code"] = etf_code
        df["etf_rank"] = np.arange(1, len(df) + 1)
        df["is_top10"] = df["etf_rank"] <= 10
        prior = previous_codes.get(etf_code, set())
        df["is_new"] = False if not prior else ~df["code"].isin(prior)
        frames.append(df[["etf_code", "code", "name", "shares", "weight", "etf_rank", "is_top10", "is_new"]])
    all_holdings = pd.concat(frames, ignore_index=True)
    all_holdings.to_csv(PREVIOUS_HOLDINGS_PATH, index=False, encoding="utf-8-sig")
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
            "00981A_new": False,
            "00992A_new": False,
        }
        for _, r in g.iterrows():
            etf = r["etf_code"]
            row[f"{etf}_shares"] = int(r["shares"])
            row[f"{etf}_weight"] = r["weight"]
            row[f"{etf}_rank"] = int(r["etf_rank"])
            row[f"{etf}_top10"] = bool(r["is_top10"])
            row[f"{etf}_new"] = bool(r["is_new"])
        rows.append(row)
    pool = pd.DataFrame(rows)
    pool = pool.sort_values(
        ["etf_count", "00992A_top10", "00981A_top10", "00992A_new", "00981A_new", "00992A_weight", "00981A_shares"],
        ascending=[False, False, False, False, False, False, False],
    ).reset_index(drop=True)
    pool.to_csv(OUTPUT_DIR / "candidate_pool.csv", index=False, encoding="utf-8-sig")
    return pool


def yahoo_symbols_for_stock_id(stock_id: str) -> list[str]:
    if stock_id == TAIEX_SYMBOL:
        return ["^TWII"]
    return [f"{stock_id}.TW", f"{stock_id}.TWO"]


def fetch_yahoo_chart(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=timezone.utc)
    params = {
        "period1": int(start_dt.timestamp()),
        "period2": int(end_dt.timestamp()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(YAHOO_CHART_URL.format(symbol=symbol), params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    error = payload.get("chart", {}).get("error")
    if error:
        raise RuntimeError(error)
    result = payload.get("chart", {}).get("result") or []
    if not result:
        return pd.DataFrame()
    r = result[0]
    ts = r.get("timestamp") or []
    quote = (r.get("indicators", {}).get("quote") or [{}])[0]
    adj = (r.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or []
    if not ts or not adj:
        return pd.DataFrame()
    df = pd.DataFrame({
        "date": [datetime.fromtimestamp(x, tz=timezone.utc).date().isoformat() for x in ts],
        "close_adj": adj,
        "close_raw": quote.get("close", [np.nan] * len(ts)),
        "Trading_Volume": quote.get("volume", [np.nan] * len(ts)),
    })
    df = df.dropna(subset=["close_adj"])
    return df


def fetch_stock_price(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    errors = []
    for symbol in yahoo_symbols_for_stock_id(stock_id):
        try:
            df = fetch_yahoo_chart(symbol, start_date, end_date)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
            continue
        if not df.empty:
            df["stock_id"] = stock_id
            df["yahoo_symbol"] = symbol
            return df
    print(f"[WARN] Yahoo adjusted price failed for {stock_id}: {' | '.join(errors)}")
    return pd.DataFrame()


def fetch_all_prices(pool: pd.DataFrame, days_back: int = 430) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=days_back)
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    frames = []
    ids = pool["code"].dropna().astype(str).unique().tolist()
    ids_with_benchmark = ids + [TAIEX_SYMBOL]
    for i, sid in enumerate(ids_with_benchmark, 1):
        print(f"[INFO] Fetch adjusted price {i}/{len(ids_with_benchmark)}: {sid}")
        df = fetch_stock_price(sid, start_s, end_s)
        if df.empty:
            print(f"[WARN] Empty adjusted price data: {sid}")
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("No adjusted price data fetched.")
    prices = pd.concat(frames, ignore_index=True)
    stock_count = prices.loc[prices["stock_id"] != TAIEX_SYMBOL, "stock_id"].nunique()
    expected_count = len(ids)
    if stock_count < max(1, int(expected_count * 0.8)):
        raise RuntimeError(f"Adjusted price coverage too low: {stock_count}/{expected_count}")
    prices.to_csv(RAW_DIR / "price_data.csv", index=False, encoding="utf-8-sig")
    return prices


def add_indicators_for_one(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    close = pd.to_numeric(df["close_adj"], errors="coerce")
    volume = pd.to_numeric(df["Trading_Volume"], errors="coerce")

    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()
    df["vol_ma20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["vol_ma20"]
    df["ret_5d"] = close / close.shift(5) - 1
    df["ret_20d"] = close / close.shift(20) - 1
    df["ret_60d"] = close / close.shift(60) - 1

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
    pieces = [add_indicators_for_one(g) for _, g in prices.groupby("stock_id")]
    ind = pd.concat(pieces, ignore_index=True)
    latest = ind.sort_values("date").groupby("stock_id").tail(1).copy()

    bench = latest[latest["stock_id"] == TAIEX_SYMBOL]
    if bench.empty:
        raise RuntimeError("Missing TAIEX benchmark; cannot calculate RS.")
    bench_ret20 = float(bench["ret_20d"].iloc[0])
    bench_ret60 = float(bench["ret_60d"].iloc[0])
    bench_date = str(bench["date"].iloc[0])
    print(f"[INFO] Benchmark latest date: {bench_date}, ret20={bench_ret20:.4f}, ret60={bench_ret60:.4f}")

    latest = latest[latest["stock_id"] != TAIEX_SYMBOL].copy()
    latest["rs20"] = latest["ret_20d"] - bench_ret20
    latest["rs60"] = latest["ret_60d"] - bench_ret60
    latest["rs_accel"] = latest["rs20"] - latest["rs60"]
    latest["rs_score"] = 0.7 * latest["rs20"] + 0.3 * latest["rs_accel"]
    latest["rs_rank"] = latest["rs_score"].rank(pct=True, ascending=True) * 100
    latest["rs_rank"] = latest["rs_rank"].round(1)

    latest = latest.rename(columns={"stock_id": "code"})
    keep_cols = [
        "code", "date", "yahoo_symbol", "close_raw", "close_adj", "Trading_Volume", "ma5", "ma20", "ma60", "vol_ma20", "volume_ratio",
        "ret_5d", "ret_20d", "ret_60d", "rs20", "rs60", "rs_accel", "rs_score", "rs_rank",
        "dif", "macd", "osc", "osc_prev", "macd_cross_up", "osc_turn_positive",
        "above_ma20", "dist_ma20_pct", "is_20d_high"
    ]
    latest = latest[keep_cols]
    merged = pool.merge(latest, on="code", how="left")
    missing_price = merged["close_adj"].isna().sum()
    if missing_price > 0:
        print(f"[WARN] Missing adjusted price rows after merge: {missing_price}")
    return merged


def add_research_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rs_component = out["rs_rank"].fillna(0)
    accel_component = out["rs_accel"].fillna(0).rank(pct=True, ascending=True) * 100
    etf_component = out["etf_count"].fillna(0).clip(upper=2) / 2 * 100
    top10_component = ((out["00981A_top10"].fillna(False)) | (out["00992A_top10"].fillna(False))).astype(int) * 100
    new_component = ((out["00981A_new"].fillna(False)) | (out["00992A_new"].fillna(False))).astype(int) * 100
    timing_component = (
        out["above_ma20"].fillna(False).astype(int) * 35
        + out["osc_turn_positive"].fillna(False).astype(int) * 35
        + out["macd_cross_up"].fillna(False).astype(int) * 30
    )
    out["research_score"] = (
        0.35 * rs_component
        + 0.15 * accel_component
        + 0.15 * etf_component
        + 0.15 * top10_component
        + 0.10 * new_component
        + 0.10 * timing_component
    ).round(1)
    return out.sort_values(["research_score", "rs_rank", "etf_count"], ascending=[False, False, False]).reset_index(drop=True)


def export_excel(screen: pd.DataFrame, pool: pd.DataFrame, holdings: pd.DataFrame) -> pathlib.Path:
    today = date.today().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"daily_screen_{today}.xlsx"
    latest_path = OUTPUT_DIR / "daily_screen_latest.xlsx"

    watch_cols = [
        "code", "name", "research_score", "rs_rank", "rs20", "rs60", "rs_accel", "etf_list", "etf_count",
        "00981A_top10", "00992A_top10", "00981A_new", "00992A_new",
        "00981A_rank", "00992A_rank", "00992A_weight",
        "date", "yahoo_symbol", "close_raw", "close_adj", "ret_5d", "ret_20d", "ret_60d", "volume_ratio",
        "ma5", "ma20", "ma60", "above_ma20", "dist_ma20_pct", "is_20d_high",
        "dif", "macd", "osc", "macd_cross_up", "osc_turn_positive"
    ]
    watch_cols = [c for c in watch_cols if c in screen.columns]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        screen[watch_cols].to_excel(writer, sheet_name="Ranking", index=False)
        pool.to_excel(writer, sheet_name="Candidate_Pool", index=False)
        holdings.to_excel(writer, sheet_name="ETF_Holdings", index=False)
        screen.to_excel(writer, sheet_name="Full_Data", index=False)

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
