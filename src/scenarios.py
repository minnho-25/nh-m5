"""Historical paths: cửa sổ trượt N tháng liên tiếp của ETH/USD thật."""
import numpy as np
import pandas as pd

MESA_COLUMNS = ["path_id", "step", "date", "eth_usd"]


def build_historical_paths(monthly, n_rounds, cycle_months=1):
    """Trả về (mesa_df, summary_df). Không tạo giá mới, không random."""
    if cycle_months != 1:
        raise NotImplementedError("Hiện chỉ hỗ trợ cycle_months = 1")
    if len(monthly) < n_rounds:
        raise ValueError(f"Chỉ có {len(monthly)} tháng, cần >= {n_rounds}")

    mesa_rows, summary_rows = [], []
    for i in range(len(monthly) - n_rounds + 1):
        w = monthly.iloc[i:i + n_rounds]
        pid = f"PATH_{i + 1:04d}"
        prices = w["eth_usd"].to_numpy(dtype=float)
        peak = np.maximum.accumulate(prices)
        for step, (_, row) in enumerate(w.iterrows()):
            mesa_rows.append({
                "path_id": pid,
                "step": step,
                "date": row["date"].strftime("%Y-%m-%d"),
                "eth_usd": float(row["eth_usd"]),
            })
        summary_rows.append({
            "path_id": pid,
            "start_month": w.iloc[0]["month"],
            "end_month": w.iloc[-1]["month"],
            "start_date": w.iloc[0]["date"].strftime("%Y-%m-%d"),
            "end_date": w.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "n_steps": int(n_rounds),
            "start_price": float(prices[0]),
            "end_price": float(prices[-1]),
            "min_price": float(prices.min()),
            "path_return_total": float(prices[-1] / prices[0] - 1.0),
            "max_drawdown": float(((peak - prices) / peak).max()),
        })
    return pd.DataFrame(mesa_rows, columns=MESA_COLUMNS), pd.DataFrame(summary_rows)


def validate_mesa_file(df, n_rounds):
    failures = []
    if list(df.columns) != MESA_COLUMNS:
        return {"ok": False, "failures": [f"cột sai: {list(df.columns)}"]}
    if df.isna().any().any():
        failures.append("có NaN")
    if df.duplicated(["path_id", "step"]).any():
        failures.append("trùng (path_id, step)")
    if (df["eth_usd"] <= 0).any():
        failures.append("có eth_usd <= 0")

    bad_len, bad_date = [], []
    for pid, g in df.groupby("path_id"):
        if list(g["step"]) != list(range(n_rounds)):
            bad_len.append(pid)
        if not pd.to_datetime(g["date"]).is_monotonic_increasing:
            bad_date.append(pid)
    if bad_len:
        failures.append(f"path sai độ dài/step: {bad_len[:5]}")
    if bad_date:
        failures.append(f"path sai thứ tự ngày: {bad_date[:5]}")

    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "n_paths": int(df["path_id"].nunique()),
        "n_rows": int(len(df)),
    }


# ---------------------------------------------------------------
# Phase 4: đọc một path từ file Mesa-ready
# ---------------------------------------------------------------
from pathlib import Path  # noqa: E402

from src.utils import PROJECT_ROOT  # noqa: E402


def load_path(path_id, mesa_csv=None):
    """Đọc (prices, dates) của một path từ data/mesa/eth_price_mesa.csv."""
    p = Path(mesa_csv) if mesa_csv else PROJECT_ROOT / "data/mesa/eth_price_mesa.csv"
    df = pd.read_csv(p)
    g = df[df["path_id"] == path_id].sort_values("step")
    if g.empty:
        raise KeyError(f"Không có {path_id} trong {p}")
    return g["eth_usd"].astype(float).tolist(), g["date"].astype(str).tolist()
