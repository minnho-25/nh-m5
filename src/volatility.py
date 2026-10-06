"""Log return và sigma_ETH (F9) tính từ ETH/USD lịch sử."""
import numpy as np
import pandas as pd


def calculate_log_returns(monthly, price_col="eth_usd"):
    """log_return_t = ln(P_t / P_(t-1)). Dòng đầu (không có P_(t-1)) bị bỏ."""
    out = monthly[["month", "date", price_col]].copy()
    out["log_return"] = np.log(out[price_col] / out[price_col].shift(1))
    out["simple_return"] = out[price_col] / out[price_col].shift(1) - 1.0
    return out.dropna(subset=["log_return"]).reset_index(drop=True)


def calculate_volatility(returns, ddof=1, periods_per_year=12):
    """sigma_ETH = std(log_return) theo đơn vị chu kỳ (monthly nếu cycle=1 tháng)."""
    r = returns["log_return"]
    sigma = float(r.std(ddof=ddof))
    return {
        "sigma_eth": sigma,
        "sigma_eth_annualized": sigma * float(np.sqrt(periods_per_year)),
        "periods_per_year": periods_per_year,
        "ddof": ddof,
        "n_obs": int(len(r)),
        "first_return_date": str(pd.to_datetime(returns["date"]).min().date()),
        "last_return_date": str(pd.to_datetime(returns["date"]).max().date()),
        "method": "std of monthly log returns, full sample",
    }


def calculate_rolling_volatility(returns, window=12, ddof=1):
    out = returns[["month", "date", "log_return"]].copy()
    out[f"rolling_sigma_{window}m"] = out["log_return"].rolling(window).std(ddof=ddof)
    return out.dropna().reset_index(drop=True)


from pathlib import Path  # noqa: E402

from src.utils import PROJECT_ROOT  # noqa: E402


def load_sigma_eth(path=None):
    """Đọc sigma_ETH (tháng) đã tính ở prepare_eth_data.py. Không hard-code."""
    p = Path(path) if path else PROJECT_ROOT / "data/processed/volatility.csv"
    if not p.exists():
        raise FileNotFoundError(f"Chưa có {p}. Chạy: python prepare_eth_data.py")
    return float(pd.read_csv(p)["sigma_eth"].iloc[0])
