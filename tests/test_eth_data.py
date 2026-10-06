import pandas as pd
import pytest

from src.eth_data import validate_daily
from src.utils import PROJECT_ROOT


def _good_df():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "date": dates, "open": [1.0] * 5, "high": [2.0] * 5,
        "low": [0.5] * 5, "close": [1.5] * 5, "volume": [10.0] * 5,
    })


def test_validate_good():
    assert validate_daily(_good_df())["ok"]


def test_validate_duplicate_dates():
    df = _good_df()
    df.loc[1, "date"] = df.loc[0, "date"]
    assert not validate_daily(df)["ok"]


def test_validate_nonpositive_price():
    df = _good_df()
    df.loc[2, "close"] = 0
    assert not validate_daily(df)["ok"]


def test_validate_nan():
    df = _good_df()
    df.loc[3, "close"] = float("nan")
    assert not validate_daily(df)["ok"]


def test_raw_file_on_disk():
    p = PROJECT_ROOT / "data/raw/eth_usd_daily_raw.csv"
    if not p.exists():
        pytest.skip("chưa chạy download_eth_data.py")
    df = pd.read_csv(p, parse_dates=["date"])
    assert validate_daily(df)["ok"]
    assert df["date"].min() >= pd.Timestamp("2020-01-01")
