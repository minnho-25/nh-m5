"""Tải, kiểm tra, lưu dữ liệu ETH/USD daily (REAL DATA)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def download_eth_daily(ticker, start_date, end_date=None, drop_incomplete_today=True):
    """Tải daily OHLCV từ Yahoo Finance. Trả về DataFrame cột REQUIRED_COLUMNS."""
    import yfinance as yf

    kwargs = dict(start=start_date, interval="1d", auto_adjust=False, progress=False)
    if end_date:
        kwargs["end"] = end_date
    raw = yf.download(ticker, **kwargs)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo Finance không trả dữ liệu cho {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"index": "date", "datetime": "date"})
    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    if drop_incomplete_today:
        # Nến của ngày hiện tại (UTC) chưa đóng -> close chưa phải close thật
        today_utc = pd.Timestamp(datetime.now(timezone.utc).date())
        df = df[df["date"] < today_utc]

    return df.sort_values("date").reset_index(drop=True)


def validate_daily(df):
    """Trả về dict report. report['ok'] = False nếu có lỗi nghiêm trọng."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return {"ok": False, "failures": [f"missing columns: {missing_cols}"], "warnings": []}
    if len(df) == 0:
        return {"ok": False, "failures": ["empty dataframe"], "warnings": []}

    failures, warnings = [], []
    report = {
        "first_date": str(df["date"].min().date()),
        "last_date": str(df["date"].max().date()),
        "row_count": int(len(df)),
        "missing_values": int(df[REQUIRED_COLUMNS].isna().sum().sum()),
        "duplicate_dates": int(df["date"].duplicated().sum()),
        "all_prices_positive": bool((df[["open", "high", "low", "close"]] > 0).all().all()),
        "date_sorted": bool(df["date"].is_monotonic_increasing),
        "min_close": float(df["close"].min()),
        "max_close": float(df["close"].max()),
    }

    expected = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    report["missing_calendar_days"] = int(len(expected.difference(df["date"])))

    if report["missing_values"] > 0:
        failures.append("có giá trị NaN")
    if report["duplicate_dates"] > 0:
        failures.append("có ngày trùng")
    if not report["all_prices_positive"]:
        failures.append("có giá <= 0")
    if not report["date_sorted"]:
        failures.append("date không tăng dần")
    if report["missing_calendar_days"] > 0:
        warnings.append(f"thiếu {report['missing_calendar_days']} ngày (crypto giao dịch 24/7)")

    report["failures"], report["warnings"] = failures, warnings
    report["ok"] = len(failures) == 0
    return report


def save_raw_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def write_metadata(path, *, source, ticker, start_date, df, frequency,
                   price_column, monthly_price_method):
    meta = {
        "source": source,
        "ticker": ticker,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date,
        "first_date_in_data": str(df["date"].min().date()),
        "end_date": str(df["date"].max().date()),
        "frequency": frequency,
        "price_column": price_column,
        "monthly_price_method": monthly_price_method,
        "row_count": int(len(df)),
        "note": "Nến của ngày tải (UTC) bị loại vì chưa đóng.",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def plot_daily_price(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["date"], df["close"], linewidth=1)
    ax.set_title("ETH/USD daily close (Yahoo Finance)")
    ax.set_xlabel("Date")
    ax.set_ylabel("ETH/USD")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------
# Phase 3: daily -> monthly
# ---------------------------------------------------------------
def to_monthly_last_close(daily, price_col="close", drop_incomplete_last_month=True):
    """Giá tháng = close của ngày cuối cùng có dữ liệu trong tháng.

    Tháng cuối chỉ được giữ nếu ngày cuối của nó là ngày cuối tháng dương lịch
    (tránh dùng close của tháng chưa đóng).
    """
    df = daily[["date", price_col]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    g = (
        df.groupby("month", sort=True)
        .agg(date=("date", "last"), eth_usd=(price_col, "last"),
             n_daily_obs=(price_col, "size"))
        .reset_index()
    )
    if drop_incomplete_last_month and len(g) > 0:
        if not g.iloc[-1]["date"].is_month_end:
            g = g.iloc[:-1]
    return g.reset_index(drop=True)
