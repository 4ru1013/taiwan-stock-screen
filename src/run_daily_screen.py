import pathlib
import re
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

ETF_CONFIG = {
    "00981A": {
        "repo": "4ru1013/united-etf-00981a-portfolio",
        "prefix": "00981A_holdings_",
    },
}

PRIMARY_ETF = "00981A"
TAIEX_SYMBOL = "TAIEX"
OUTPUT_DIR = pathlib.Path("output")
RAW_DIR = pathlib.Path("data/raw")
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
GITHUB_CONTENTS_URL = "https://api.github.com/repos/{repo}/contents/data/out"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def normalize_code(code) -> str:
    return str(code).strip().replace(".0", "")


def read_csv_from_url(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(resp.text), dtype={"code": "string"})


def list_latest_two_holding_files(etf_code: str) -> tuple[dict, dict | None]:
    cfg = ETF_CONFIG[etf_code]
    resp = requests.get(GITHUB_CONTENTS_URL.format(repo=cfg["repo"]), timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    items = resp.json()
    pattern = re.compile(rf"^{re.escape(cfg['prefix'])}(\d{{8}})\.csv$")
    matches = []
    for item in items:
        name = item.get("name", "")
        m = pattern.match(name)
        if not m:
            continue
        matches.append({
            "date": m.group(1),
            "name": name,
            "download_url": item.get("download_url"),
        })
    matches = sorted(matches, key=lambda x: x["date"])
    if not matches:
        raise RuntimeError(f"No historical holdings files found for {etf_code}.")
    latest = matches[-1]
    previous = matches[-2] if len(matches) >= 2 else None
    print(f"[INFO] {etf_code} latest holdings: {latest['name']}")
    if previous:
        print(f"[INFO] {etf_code} previous holdings: {previous['name']}")
    else:
        print(f"[WARN] {etf_code} has no previous holdings file; ETF flow disabled for this ETF.")
    return latest, previous


def normalize_holding_df(df: pd.DataFrame, etf_code: str, source_date: str, is_previous: bool = False) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "code" not in df.columns or "name" not in df.columns or "shares" not in df.columns:
        raise ValueError(f"{etf_code} holdings columns invalid: {list(df.columns)}")
    df["code"] = df["code"].map(normalize_code)
    df["name"] = df["name"].astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce") if "weight" in df.columns else np.nan
    df = df.sort_values(["weight", "shares"], ascending=[False, False]).reset_index(drop=True)
    df["etf_code"] = etf_code
    df["source_date"] = source_date
    rank_col = "prev_rank" if is_previous else "etf_rank"
    df[rank_col] = np.arange(1, len(df) + 1)
    return df


def load_one_etf_holdings(etf_code: str) -> pd.DataFrame:
    latest_file, previous_file = list_latest_two_holding_files(etf_code)
    latest_df = normalize_holding_df(read_csv_from_url(latest_file["download_url"]), etf_code, latest_file["date"])

    if previous_file:
        prev_df = normalize_holding_df(read_csv_from_url(previous_file["download_url"]), etf_code, previous_file["date"], is_previous=True)
        prev_df = prev_df[["code", "shares", "prev_rank"]].rename(columns={"shares": "prev_shares"})
    else:
        prev_df = pd.DataFrame(columns=["code", "prev_shares", "prev_rank"])

    latest_df = latest_df.merge(prev_df, on="code", how="left")
    latest_df["prev_shares"] = latest_df["prev_shares"].fillna(0).astype(int)
    latest_df["prev_rank"] = pd.to_numeric(latest_df["prev_rank"], errors="coerce")
    latest_df["delta_shares"] = latest_df["shares"] - latest_df["prev_shares"]
    latest_df["delta_pct"] = np.where(latest_df["prev_shares"] > 0, latest_df["delta_shares"] / latest_df["prev_shares"], np.nan)
    latest_df["is_new"] = latest_df["prev_shares"].eq(0) & latest_df["shares"].gt(0)
    latest_df["is_top10"] = latest_df["etf_rank"] <= 10
    latest_df["latest_file"] = latest_file["name"]
    latest_df["previous_file"] = previous_file["name"] if previous_file else ""
    return latest_df[[
        "etf_code", "source_date", "latest_file", "previous_file", "code", "name", "shares", "weight", "etf_rank",
        "is_top10", "is_new", "prev_shares", "prev_rank", "delta_shares", "delta_pct"
    ]]


def load_etf_holdings() -> pd.DataFrame:
    frames = [load_one_etf_holdings(etf_code) for etf_code in ETF_CONFIG]
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
            "in_00981A": PRIMARY_ETF in etfs,
            "00981A_source_date": "",
            "00981A_shares": 0,
            "00981A_weight": np.nan,
            "00981A_rank": np.nan,
            "00981A_top10": False,
            "00981A_new": False,
            "00981A_prev_shares": 0,
            "00981A_delta_shares": 0,
            "00981A_delta_pct": np.nan,
            "00981A_flow_value": np.nan,
            "00981A_buy_flow_rank": np.nan,
            "00981A_sell_flow_rank": np.nan,
        }
        r = g[g["etf_code"] == PRIMARY_ETF].iloc[0]
        row["00981A_source_date"] = str(r["source_date"])
        row["00981A_shares"] = int(r["shares"])
        row["00981A_weight"] = r["weight"]
        row["00981A_rank"] = int(r["etf_rank"])
        row["00981A_top10"] = bool(r["is_top10"])
        row["00981A_new"] = bool(r["is_new"])
        row["00981A_prev_shares"] = int(r["prev_shares"])
        row["00981A_delta_shares"] = int(r["delta_shares"])
        row["00981A_delta_pct"] = r["delta_pct"]
        rows.append(row)

    pool = pd.DataFrame(rows)
    pool = pool.sort_values(
        ["00981A_top10", "00981A_new", "00981A_shares", "00981A_delta_shares"],
        ascending=[False, False, False, False],
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


def calc_next_day_osc(close_values: pd.Series, next_close: float) -> float:
    series = pd.concat([close_values, pd.Series([next_close])], ignore_index=True)
    ema8 = series.ewm(span=8, adjust=False).mean()
    ema17 = series.ewm(span=17, adjust=False).mean()
    dif = ema8 - ema17
    macd = dif.ewm(span=9, adjust=False).mean()
    return float((dif - macd).iloc[-1])


def find_osc_flip_price(close_values: pd.Series, current_close: float, current_osc: float):
    if pd.isna(current_osc) or pd.isna(current_close):
        return np.nan
    if current_osc > 0:
        return "Already Positive"
    low = float(current_close)
    high = float(current_close) * 1.15
    if calc_next_day_osc(close_values, high) <= 0:
        return np.nan
    for _ in range(30):
        mid = (low + high) / 2
        if calc_next_day_osc(close_values, mid) > 0:
            high = mid
        else:
            low = mid
    return round(high, 2)


def find_ma20_upturn_price(close_values: pd.Series, current_close: float):
    if len(close_values) < 20 or pd.isna(current_close):
        return np.nan
    ma20 = close_values.rolling(20).mean()
    if len(ma20) >= 2 and ma20.iloc[-1] > ma20.iloc[-2]:
        return "Already Upturn"
    today_ma20 = float(ma20.iloc[-1])
    last_19_sum = float(close_values.iloc[-19:].sum())
    required = today_ma20 * 20 - last_19_sum
    return round(max(required + 0.01, float(current_close)), 2)


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
    df["ma20_gt_ma60"] = df["ma20"] > df["ma60"]
    df["close_gt_ma60"] = close > df["ma60"]
    df["dist_ma20_pct"] = close / df["ma20"] - 1
    df["high_20d"] = close.rolling(20).max()
    df["is_20d_high"] = close >= df["high_20d"]

    if not df.empty:
        idx = df.index[-1]
        cur_close = float(close.iloc[-1]) if not pd.isna(close.iloc[-1]) else np.nan
        cur_osc = float(df["osc"].iloc[-1]) if not pd.isna(df["osc"].iloc[-1]) else np.nan
        df.loc[idx, "osc_flip_price"] = find_osc_flip_price(close.dropna().reset_index(drop=True), cur_close, cur_osc)
        df.loc[idx, "ma20_upturn_price"] = find_ma20_upturn_price(close.dropna().reset_index(drop=True), cur_close)
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
    print(f"[INFO] Benchmark latest date: {bench['date'].iloc[0]}, ret20={bench_ret20:.4f}, ret60={bench_ret60:.4f}")

    latest = latest[latest["stock_id"] != TAIEX_SYMBOL].copy()
    latest["rs20"] = latest["ret_20d"] - bench_ret20
    latest["rs60"] = latest["ret_60d"] - bench_ret60
    latest["rs_accel"] = latest["rs20"] - latest["rs60"]
    latest["rs_score"] = 0.7 * latest["rs20"] + 0.3 * latest["rs_accel"]
    latest["rs_rank"] = latest["rs_score"].rank(pct=True, ascending=True) * 100
    latest["rs_rank"] = latest["rs_rank"].round(1)
    latest["rs20_rank"] = latest["rs20"].rank(pct=True, ascending=True) * 100
    latest["rs20_rank"] = latest["rs20_rank"].round(1)

    latest = latest.rename(columns={"stock_id": "code"})
    keep_cols = [
        "code", "date", "yahoo_symbol", "close_raw", "close_adj", "Trading_Volume", "ma5", "ma20", "ma60", "vol_ma20", "volume_ratio",
        "ret_5d", "ret_20d", "ret_60d", "rs20", "rs60", "rs_accel", "rs_score", "rs_rank", "rs20_rank",
        "dif", "macd", "osc", "osc_prev", "macd_cross_up", "osc_turn_positive",
        "above_ma20", "ma20_gt_ma60", "close_gt_ma60", "dist_ma20_pct", "is_20d_high", "osc_flip_price", "ma20_upturn_price"
    ]
    latest = latest[keep_cols]
    merged = pool.merge(latest, on="code", how="left")
    missing_price = merged["close_adj"].isna().sum()
    if missing_price > 0:
        print(f"[WARN] Missing adjusted price rows after merge: {missing_price}")
    return merged


def add_flow_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    value_col = "00981A_flow_value"
    out[value_col] = out["00981A_delta_shares"].fillna(0) * out["close_adj"]
    buy_mask = out[value_col] > 0
    sell_mask = out[value_col] < 0
    out["00981A_buy_flow_rank"] = np.nan
    out["00981A_sell_flow_rank"] = np.nan
    if buy_mask.any():
        out.loc[buy_mask, "00981A_buy_flow_rank"] = out.loc[buy_mask, value_col].rank(ascending=False, method="min")
    if sell_mask.any():
        out.loc[sell_mask, "00981A_sell_flow_rank"] = (-out.loc[sell_mask, value_col]).rank(ascending=False, method="min")
    return out


def make_setup(row) -> str:
    if not bool(row.get("ma20_gt_ma60", False)) or not bool(row.get("close_gt_ma60", False)):
        return "D"
    if pd.notna(row.get("osc")) and row.get("osc") > 0:
        return "A"
    flip = row.get("osc_flip_price")
    close = row.get("close_adj")
    if isinstance(flip, (int, float, np.floating)) and pd.notna(flip) and pd.notna(close) and flip <= close * 1.05:
        return "B"
    return "C"


def make_etf_tag(row) -> str:
    tags = []
    if bool(row.get("00981A_top10", False)):
        tags.append("00981A Top10")
    if bool(row.get("00981A_new", False)):
        tags.append("00981A New Entry")
    buy_rank = row.get("00981A_buy_flow_rank")
    sell_rank = row.get("00981A_sell_flow_rank")
    if pd.notna(buy_rank) and buy_rank <= 10:
        tags.append("00981A Heavy Buy")
    if pd.notna(sell_rank) and sell_rank <= 10:
        tags.append("00981A Heavy Sell")
    return " | ".join(tags)


def make_market_tag(row) -> str:
    tags = []
    if pd.notna(row.get("rs20_rank")) and row.get("rs20_rank") >= 80:
        tags.append("Leader")
    if pd.notna(row.get("rs_accel")):
        if row.get("rs_accel") > 0:
            tags.append("Accelerating")
        elif row.get("rs_accel") < 0:
            tags.append("Decelerating")
    return " | ".join(tags)


def add_strategy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = add_flow_ranks(df)
    out["setup"] = out.apply(make_setup, axis=1)
    out["etf_tag"] = out.apply(make_etf_tag, axis=1)
    out["market_tag"] = out.apply(make_market_tag, axis=1)

    timing_map = {"A": 100, "B": 90, "C": 55, "D": 0}
    timing_component = out["setup"].map(timing_map).fillna(0)
    market_component = (0.7 * out["rs20_rank"].fillna(0)) + (0.3 * out["rs_accel"].fillna(0).rank(pct=True, ascending=True) * 100)
    etf_component = (
        out["00981A_top10"].fillna(False).astype(int) * 40
        + out["00981A_new"].fillna(False).astype(int) * 20
        + (out["00981A_buy_flow_rank"].fillna(999) <= 10).astype(int) * 40
    )
    out["trading_score"] = (0.60 * timing_component + 0.30 * market_component + 0.10 * etf_component).round(1)
    return out.sort_values(["setup", "trading_score", "rs20_rank"], ascending=[True, False, False]).reset_index(drop=True)


def export_excel(screen: pd.DataFrame, pool: pd.DataFrame, holdings: pd.DataFrame) -> pathlib.Path:
    today = date.today().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"daily_screen_{today}.xlsx"
    latest_path = OUTPUT_DIR / "daily_screen_latest.xlsx"

    ranking_cols = [
        "code", "name", "setup", "etf_tag", "market_tag", "trading_score",
        "rs20_rank", "rs_accel", "close_adj", "osc_flip_price", "ma20_upturn_price",
        "ma20_gt_ma60", "close_gt_ma60", "dif", "macd", "osc",
        "00981A_source_date", "00981A_rank", "00981A_weight", "00981A_top10", "00981A_new",
        "00981A_delta_shares", "00981A_flow_value", "00981A_buy_flow_rank", "00981A_sell_flow_rank",
        "date", "yahoo_symbol", "close_raw", "ret_5d", "ret_20d", "ret_60d", "volume_ratio",
        "ma5", "ma20", "ma60", "is_20d_high"
    ]
    ranking_cols = [c for c in ranking_cols if c in screen.columns]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        screen[ranking_cols].to_excel(writer, sheet_name="Ranking", index=False)
        pool.to_excel(writer, sheet_name="Candidate_Pool", index=False)
        holdings.to_excel(writer, sheet_name="ETF_Holdings", index=False)
        screen.to_excel(writer, sheet_name="Full_Data", index=False)

    with pd.ExcelWriter(latest_path, engine="openpyxl") as writer:
        screen[ranking_cols].to_excel(writer, sheet_name="Ranking", index=False)
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
    screen = add_strategy_columns(screen)
    export_excel(screen, pool, holdings)


if __name__ == "__main__":
    main()
