import math
import pathlib
from datetime import date

import numpy as np
import pandas as pd

SOURCE_CSV = pathlib.Path("output/daily_screen_latest.csv")
REPORT_DIR = pathlib.Path("output/reports")


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def read_screen() -> pd.DataFrame:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE_CSV}")
    df = pd.read_csv(SOURCE_CSV, dtype={"code": "string"})
    df["code"] = df["code"].astype(str).str.zfill(4)
    if "osc_expanding" not in df.columns:
        df["osc_expanding"] = pd.to_numeric(df["osc"], errors="coerce") > pd.to_numeric(df["osc_prev"], errors="coerce")
    return df


def as_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def fmt_num(value, digits=2, signed=False):
    value = as_float(value)
    if pd.isna(value):
        return ""
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:.{digits}f}"


def fmt_price(value):
    if isinstance(value, str):
        return value
    value = as_float(value)
    if pd.isna(value):
        return ""
    return f"{value:.2f}"


def fmt_pct(value):
    value = as_float(value)
    if pd.isna(value):
        return ""
    return f"{value:.2f}%"


def flow_label(row) -> str:
    tag = str(row.get("etf_tag", "") or "")
    if "Heavy Sell" in tag:
        return "Heavy Sell"
    if "Heavy Buy" in tag:
        return "Heavy Buy"
    if "New Entry" in tag:
        return "New Entry"
    return "Neutral"


def flow_score(label: str) -> int:
    return {
        "Heavy Buy": 3,
        "New Entry": 2,
        "Neutral": 0,
        "Heavy Sell": -3,
    }.get(label, 0)


def rating(row) -> str:
    rs20 = as_float(row.get("rs20_rank"))
    rs_accel = as_float(row.get("rs_accel"))
    flow = flow_label(row)
    if rs20 >= 95 and rs_accel > 0 and flow in {"Heavy Buy", "New Entry"}:
        return "Leader"
    if rs20 >= 90:
        return "強"
    if rs20 >= 80:
        return "可觀察"
    return "一般"


def distance_to_price(row) -> float:
    close = as_float(row.get("close_adj"))
    candidates = []
    for col in ["osc_flip_price", "ma20_upturn_price"]:
        v = row.get(col)
        if isinstance(v, str):
            if v.startswith("Already"):
                candidates.append(close)
            continue
        fv = as_float(v)
        if not pd.isna(fv):
            candidates.append(fv)
    if pd.isna(close) or close <= 0 or not candidates:
        return np.nan
    target = max(candidates)
    return (target / close - 1) * 100


def market_regime(a_count: int, b_count: int, total: int) -> str:
    active = a_count + b_count
    if active >= max(10, total * 0.25):
        return "Risk On"
    if active >= max(5, total * 0.12):
        return "Neutral"
    return "Risk Off"


def build_a_table(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df["setup"].eq("A")].copy()
    if a.empty:
        return pd.DataFrame(columns=["Rank", "代號", "股票", "RS20", "RS Accel", "ETF Flow", "評價"])
    a["_flow_label"] = a.apply(flow_label, axis=1)
    a["_flow_score"] = a["_flow_label"].map(flow_score)
    a["_rating"] = a.apply(rating, axis=1)
    a = a.sort_values(
        ["rs20_rank", "rs_accel", "_flow_score", "trading_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    out = pd.DataFrame({
        "Rank": np.arange(1, len(a) + 1),
        "代號": a["code"],
        "股票": a["name"],
        "RS20": a["rs20_rank"].map(lambda x: fmt_num(x, 0)),
        "RS Accel": a["rs_accel"].map(lambda x: fmt_num(x, 2, signed=True)),
        "ETF Flow": a["_flow_label"],
        "評價": a["_rating"],
    })
    return out


def build_b_table(df: pd.DataFrame) -> pd.DataFrame:
    b = df[df["setup"].eq("B")].copy()
    if b.empty:
        return pd.DataFrame(columns=["Rank", "代號", "股票", "OSC翻正價", "MA20上彎價", "距離現價", "ETF Flow"])
    b["_flow_label"] = b.apply(flow_label, axis=1)
    b["_flow_score"] = b["_flow_label"].map(flow_score)
    b["_distance"] = b.apply(distance_to_price, axis=1)
    b = b.sort_values(
        ["_distance", "rs20_rank", "_flow_score", "trading_score"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    out = pd.DataFrame({
        "Rank": np.arange(1, len(b) + 1),
        "代號": b["code"],
        "股票": b["name"],
        "OSC翻正價": b["osc_flip_price"].map(fmt_price),
        "MA20上彎價": b["ma20_upturn_price"].map(fmt_price),
        "距離現價": b["_distance"].map(fmt_pct),
        "ETF Flow": b["_flow_label"],
    })
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "無\n"
    return df.to_markdown(index=False)


def build_trade_lists(a_table: pd.DataFrame, b_table: pd.DataFrame):
    tier1 = a_table.head(3).copy()
    tier2 = a_table.iloc[3:5].copy()
    watch = b_table.head(5).copy()
    return tier1, tier2, watch


def bullet_list(df: pd.DataFrame) -> str:
    if df.empty:
        return "- 無"
    return "\n".join([f"- {row['代號']} {row['股票']}" for _, row in df.iterrows()])


def deep_analysis(df: pd.DataFrame, tier1: pd.DataFrame) -> str:
    if tier1.empty:
        return "無\n"
    parts = []
    lookup = df.set_index("code", drop=False)
    for _, item in tier1.iterrows():
        code = str(item["代號"]).zfill(4)
        if code not in lookup.index:
            continue
        row = lookup.loc[code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        flow = flow_label(row)
        parts.append(
            f"## {code} {row.get('name', '')}\n\n"
            f"- Setup：{row.get('setup', '')}\n"
            f"- RS20：{fmt_num(row.get('rs20_rank'), 0)}\n"
            f"- RS Accel：{fmt_num(row.get('rs_accel'), 2, signed=True)}\n"
            f"- ETF Flow：{flow}\n"
            f"- OSC：{fmt_num(row.get('osc_prev'), 2)} → {fmt_num(row.get('osc'), 2)}\n"
            f"- OSC Expansion：{bool(row.get('osc_expanding'))}\n"
            f"- MA20上彎價：{fmt_price(row.get('ma20_upturn_price'))}\n"
            f"- 判斷：{item.get('評價', '')}\n"
        )
    return "\n".join(parts)


def write_outputs(df: pd.DataFrame, a_table: pd.DataFrame, b_table: pd.DataFrame) -> None:
    report_date = str(df["date"].dropna().iloc[0]) if "date" in df.columns and df["date"].notna().any() else date.today().isoformat()
    date_key = report_date.replace("-", "")

    setup_counts = df["setup"].value_counts().to_dict()
    a_count = int(setup_counts.get("A", 0))
    b_count = int(setup_counts.get("B", 0))
    c_count = int(setup_counts.get("C", 0))
    d_count = int(setup_counts.get("D", 0))
    regime = market_regime(a_count, b_count, len(df))

    tier1, tier2, watch = build_trade_lists(a_table, b_table)

    csv_rows = []
    for group_name, table in [("A", a_table), ("B", b_table), ("Priority1", tier1), ("Priority2", tier2), ("Watchlist", watch)]:
        temp = table.copy()
        temp.insert(0, "Group", group_name)
        csv_rows.append(temp)
    report_csv = pd.concat(csv_rows, ignore_index=True, sort=False) if csv_rows else pd.DataFrame()

    csv_latest = REPORT_DIR / "daily_selection_report_latest.csv"
    csv_dated = REPORT_DIR / f"daily_selection_report_{date_key}.csv"
    report_csv.to_csv(csv_latest, index=False, encoding="utf-8-sig")
    report_csv.to_csv(csv_dated, index=False, encoding="utf-8-sig")

    md = f"""# 00981A 每日選股報告

日期：{report_date}

## 1. 今日系統狀況

- 00981A 持股數：{len(df)}
- A：{a_count}
- B：{b_count}
- C：{c_count}
- D：{d_count}
- Market Regime：{regime}

## 2. A組

{markdown_table(a_table)}

## 3. B組

{markdown_table(b_table)}

## 4. 明日交易名單

### Priority 1

{bullet_list(tier1)}

### Priority 2

{bullet_list(tier2)}

### Watchlist

{bullet_list(watch)}

## 5. 深度分析

{deep_analysis(df, tier1)}

## 回測索引

- Date：{report_date}
- A Count：{a_count}
- B Count：{b_count}
- Priority 1：{', '.join(tier1['代號'].astype(str).tolist()) if not tier1.empty else ''}
- Priority 2：{', '.join(tier2['代號'].astype(str).tolist()) if not tier2.empty else ''}
- Watchlist：{', '.join(watch['代號'].astype(str).tolist()) if not watch.empty else ''}
"""
    md_latest = REPORT_DIR / "daily_selection_report_latest.md"
    md_dated = REPORT_DIR / f"daily_selection_report_{date_key}.md"
    md_latest.write_text(md, encoding="utf-8")
    md_dated.write_text(md, encoding="utf-8")
    print(f"[OK] Wrote {csv_latest}")
    print(f"[OK] Wrote {csv_dated}")
    print(f"[OK] Wrote {md_latest}")
    print(f"[OK] Wrote {md_dated}")


def main() -> None:
    ensure_dirs()
    df = read_screen()
    a_table = build_a_table(df)
    b_table = build_b_table(df)
    write_outputs(df, a_table, b_table)


if __name__ == "__main__":
    main()
