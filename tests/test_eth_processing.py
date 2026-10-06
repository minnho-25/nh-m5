import numpy as np
import pandas as pd
import pytest

from src.eth_data import to_monthly_last_close
from src.scenarios import build_historical_paths, validate_mesa_file
from src.utils import PROJECT_ROOT, load_config
from src.volatility import calculate_log_returns, calculate_volatility


def _daily(start, end):
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": dates, "close": np.arange(len(dates)) + 1.0})


def test_monthly_last_close_and_drop_incomplete_month():
    m = to_monthly_last_close(_daily("2020-01-01", "2020-03-15"))
    assert list(m["month"]) == ["2020-01", "2020-02"]      # tháng 3 chưa đóng bị loại
    assert m.loc[0, "eth_usd"] == 31.0                     # close ngày 2020-01-31
    assert m.loc[1, "eth_usd"] == 60.0                     # close ngày 2020-02-29


def test_log_return_formula():
    monthly = pd.DataFrame({
        "month": ["a", "b", "c", "d"],
        "date": pd.date_range("2020-01-31", periods=4, freq="ME"),
        "eth_usd": [100.0, 110.0, 99.0, 120.0],
    })
    r = calculate_log_returns(monthly)
    assert len(r) == 3
    assert r["log_return"].iloc[0] == pytest.approx(np.log(110 / 100))
    expected = np.std(np.log([110 / 100, 99 / 110, 120 / 99]), ddof=1)
    assert calculate_volatility(r)["sigma_eth"] == pytest.approx(expected)


def test_sliding_windows_synthetic():
    monthly = pd.DataFrame({
        "month": [f"2020-{i:02d}" for i in range(1, 9)],
        "date": pd.date_range("2020-01-31", periods=8, freq="ME"),
        "eth_usd": [100.0 + i for i in range(8)],
    })
    mesa_df, summary = build_historical_paths(monthly, 5)
    assert summary["path_id"].nunique() == 4               # 8 - 5 + 1
    p2 = mesa_df[mesa_df["path_id"] == "PATH_0002"]
    assert list(p2["step"]) == [0, 1, 2, 3, 4]
    assert list(p2["eth_usd"]) == [101.0, 102.0, 103.0, 104.0, 105.0]
    assert validate_mesa_file(mesa_df, 5)["ok"]


def test_validate_mesa_rejects_bad_file():
    df = pd.DataFrame({"path_id": ["P", "P"], "step": [0, 0],
                       "date": ["2020-01-31", "2020-02-29"], "eth_usd": [1.0, 2.0]})
    assert not validate_mesa_file(df, 2)["ok"]


def test_mesa_file_on_disk():
    p = PROJECT_ROOT / "data/mesa/eth_price_mesa.csv"
    m = PROJECT_ROOT / "data/processed/eth_usd_monthly.csv"
    if not p.exists() or not m.exists():
        pytest.skip("chưa chạy prepare_eth_data.py")
    N = load_config()["pool"]["N"]
    df = pd.read_csv(p)
    assert validate_mesa_file(df, N)["ok"]
    # giá trong Mesa file phải khớp đúng giá tháng thật, không có giá bịa
    monthly = pd.read_csv(m)
    merged = df.merge(monthly[["date", "eth_usd"]], on="date", suffixes=("", "_m"))
    assert len(merged) == len(df)
    assert np.allclose(merged["eth_usd"], merged["eth_usd_m"])
    assert df["path_id"].nunique() == len(monthly) - N + 1
