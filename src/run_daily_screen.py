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
GITHUB_CONTENTS_URL = "https://api.github.com/repos/{repo}/contents/data/out/holdings"


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
    prices.to_csv(RAW_DIR / "prices_latest.csv", index=False, encoding="utf-8-sig")
    return prices


def calc_macd(close: pd.Series, fast: int = 8, slow: int = 17, signal: int = 9):
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    macd = dif.ewm(span=signal, adjust=False).mean()
    osc = dif - macd
    return dif, macd, osc


def calc_setup(row) -> str:
    if not bool(row["osc_expanding"]):
        return "D"
    if row["close_above_ma20"] and row["ma20_up"] and row["macd_bull"] and row["osc_positive"]:
        return "A"
    if row["close_above_ma20"] and row["macd_near_cross"] and row["osc_improving"]:
        return "B"
    if row["close_above_ma20"]:
        return "C"
    return "D"


def estimate_osc_flip_price(latest, close_series: pd.Series) -> str | float:
    if latest["osc"] > 0:
        return "Already > 0"
    recent = close_series.dropna().tail(120)
    if recent.empty:
        return np.nan
    base = float(latest["close_adj"])
    candidates = np.linspace(base * 0.90, base * 1.20, 151)
    history = recent.iloc[:-1]
    for p in candidates:
        test = pd.concat([history, pd.Series([p])], ignore_index=True)
        _, _, osc = calc_macd(test)
        if osc.iloc[-1] > 0:
            return round(float(p), 2)
    return np.nan


def estimate_ma20_upturn_price(close_series: pd.Series) -> str | float:
    s = close_series.dropna()
    if len(s) < 21:
        return np.nan
    curr_ma20 = s.tail(20).mean()
    prev_ma20 = s.iloc[-21:-1].mean()
    if curr_ma20 > prev_ma20:
        return "Already Up"
    prior_19 = s.iloc[-19:].sum()
    prev_window = s.iloc[-20:-1].sum()
    required = (prev_window / 19) * 20 - prior_19
    return round(float(required), 2)


def compute_indicators(pool: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    benchmark = prices[prices["stock_id"] == TAIEX_SYMBOL].copy().sort_values("date")
    if benchmark.empty or len(benchmark) < 80:
        raise RuntimeError("Benchmark data insufficient.")
    bench_latest = benchmark.iloc[-1]
    bench_ret20 = bench_latest["close_adj"] / benchmark.iloc[-21]["close_adj"] - 1
    bench_ret60 = bench_latest["close_adj"] / benchmark.iloc[-61]["close_adj"] - 1
    print(f"[INFO] Benchmark latest date: {bench_latest['date']}, ret20={bench_ret20:.4f}, ret60={bench_ret60:.4f}")

    rows = []
    for _, item in pool.iterrows():
        code = item["code"]
        p = prices[prices["stock_id"] == code].copy().sort_values("date")
        if len(p) < 80:
            print(f"[WARN] Skip {code} insufficient price rows: {len(p)}")
            continue
        p["ma5"] = p["close_adj"].rolling(5).mean()
        p["ma20"] = p["close_adj"].rolling(20).mean()
        p["ma60"] = p["close_adj"].rolling(60).mean()
        p["dif"], p["macd"], p["osc"] = calc_macd(p["close_adj"])
        latest = p.iloc[-1]
        prev = p.iloc[-2]
        ret20 = latest["close_adj"] / p.iloc[-21]["close_adj"] - 1
        ret60 = latest["close_adj"] / p.iloc[-61]["close_adj"] - 1
        rs20 = ret20 - bench_ret20
        rs60 = ret60 - bench_ret60

        row = item.to_dict()
        row.update({
            "date": latest["date"],
            "close_adj": latest["close_adj"],
            "close_raw": latest["close_raw"],
            "volume": latest["Trading_Volume"],
            "ma5": latest["ma5"],
            "ma20": latest["ma20"],
            "ma60": latest["ma60"],
            "ma20_prev": prev["ma20"],
            "dif": latest["dif"],
            "macd": latest["macd"],
            "osc": latest["osc"],
            "osc_prev": prev["osc"],
            "osc_expanding": latest["osc"] > prev["osc"],
            "ret20": ret20,
            "ret60": ret60,
            "rs20": rs20,
            "rs60": rs60,
            "rs_accel": rs20 - rs60,
            "close_above_ma20": latest["close_adj"] >= latest["ma20"],
            "ma20_up": latest["ma20"] > prev["ma20"],
            "macd_bull": latest["dif"] > latest["macd"],
            "osc_positive": latest["osc"] > 0,
            "osc_improving": latest["osc"] > prev["osc"],
            "macd_near_cross": (latest["dif"] <= latest["macd"]) and ((latest["macd"] - latest["dif"]) / latest["close_adj"] < 0.015),
        })
        row["osc_flip_price"] = estimate_osc_flip_price(latest, p["close_adj"])
        row["ma20_upturn_price"] = estimate_ma20_upturn_price(p["close_adj"])
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No candidates after indicator calculation.")
    df["rs20_rank"] = df["rs20"].rank(pct=True) * 100
    df["rs60_rank"] = df["rs60"].rank(pct=True) * 100
    df["00981A_buy_flow_rank"] = df["00981A_delta_shares"].rank(pct=True) * 100
    df["00981A_sell_flow_rank"] = (-df["00981A_delta_shares"]).rank(pct=True) * 100
    df["setup"] = df.apply(calc_setup, axis=1)
    df["trading_score"] = (
        df["setup"].map({"A": 40, "B": 25, "C": 10, "D": 0}).fillna(0)
        + df["rs20_rank"] * 0.30
        + df["rs60_rank"] * 0.10
        + df["00981A_buy_flow_rank"] * 0.20
        + np.where(df["00981A_top10"], 10, 0)
        + np.where(df["00981A_new"], 8, 0)
    )

    df["etf_tag"] = np.select(
        [
            df["00981A_new"],
            df["00981A_delta_shares"] > 0,
            df["00981A_delta_shares"] < 0,
        ],
        ["New Entry", "Heavy Buy", "Heavy Sell"],
        default="Neutral",
    )
    return df.sort_values(["setup", "trading_score"], ascending=[True, False]).reset_index(drop=True)


def export_outputs(df: pd.DataFrame) -> None:
    latest_date = str(df["date"].max()).replace("-", "")
    xlsx_path = OUTPUT_DIR / f"daily_screen_{latest_date}.xlsx"
    latest_path = OUTPUT_DIR / "daily_screen_latest.xlsx"
    csv_path = OUTPUT_DIR / f"daily_screen_{latest_date}.csv"
    latest_csv_path = OUTPUT_DIR / "daily_screen_latest.csv"

    cols = [
        "date", "code", "name", "setup", "trading_score", "etf_tag",
        "00981A_source_date", "00981A_shares", "00981A_weight", "00981A_rank", "00981A_top10", "00981A_new",
        "00981A_prev_shares", "00981A_delta_shares", "00981A_delta_pct",
        "close_raw", "close_adj", "volume", "ma5", "ma20", "ma20_prev", "ma60",
        "dif", "macd", "osc", "osc_prev", "osc_expanding", "ret20", "ret60", "rs20", "rs60", "rs_accel",
        "rs20_rank", "rs60_rank", "close_above_ma20", "ma20_up", "macd_bull", "osc_positive", "osc_improving",
        "macd_near_cross", "osc_flip_price", "ma20_upturn_price",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    export_df = df[existing_cols].copy()

    export_df.to_excel(xlsx_path, index=False)
    export_df.to_excel(latest_path, index=False)
    export_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    export_df.to_csv(latest_csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Exported {xlsx_path}")
    print(f"[OK] Exported {latest_path}")
    print(f"[OK] Exported {csv_path}")
    print(f"[OK] Exported {latest_csv_path}")


def main() -> None:
    ensure_dirs()
    holdings = load_etf_holdings()
    pool = build_candidate_pool(holdings)
    prices = fetch_all_prices(pool)
    result = compute_indicators(pool, prices)
    export_outputs(result)


if __name__ == "__main__":
    main()
