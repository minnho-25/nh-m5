import json

import pandas as pd

from src import eth_data, scenarios, volatility
from src.utils import PROJECT_ROOT, load_config


def _plot_all(monthly, returns, rolling, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly["date"], monthly["eth_usd"], marker="o", markersize=3, linewidth=1)
    ax.set_title("ETH/USD monthly (last close of month)")
    ax.set_ylabel("ETH/USD"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "eth_usd_monthly.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(returns["date"], returns["log_return"], width=20)
    ax.set_title("ETH/USD monthly log return")
    ax.set_ylabel("log return"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "eth_usd_monthly_log_return.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(rolling["date"], rolling.iloc[:, -1], linewidth=1.2)
    ax.set_title("ETH/USD rolling 12-month volatility (std of monthly log return)")
    ax.set_ylabel("sigma"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "eth_usd_monthly_volatility.png", dpi=150); plt.close(fig)


def main():
    cfg = load_config()
    N = cfg["pool"]["N"]
    cycle = cfg["pool"]["cycle_months"]
    market = cfg["market"]
    if market["monthly_price_method"] != "last_close":
        raise SystemExit("Chỉ hỗ trợ monthly_price_method = last_close")

    raw_path = PROJECT_ROOT / "data/raw/eth_usd_daily_raw.csv"
    if not raw_path.exists():
        raise SystemExit("Chưa có raw CSV. Chạy: python download_eth_data.py")

    daily = pd.read_csv(raw_path, parse_dates=["date"])
    rep = eth_data.validate_daily(daily)
    if not rep["ok"]:
        raise SystemExit(f"Raw data không hợp lệ: {rep['failures']}")

    proc = PROJECT_ROOT / "data/processed"
    mesa_dir = PROJECT_ROOT / "data/mesa"
    proc.mkdir(parents=True, exist_ok=True)
    mesa_dir.mkdir(parents=True, exist_ok=True)

    daily.to_csv(proc / "eth_usd_daily.csv", index=False, encoding="utf-8")

    monthly = eth_data.to_monthly_last_close(daily, price_col=market["price_column"])
    monthly.to_csv(proc / "eth_usd_monthly.csv", index=False, encoding="utf-8")

    returns = volatility.calculate_log_returns(monthly)
    returns.to_csv(proc / "eth_usd_returns_monthly.csv", index=False, encoding="utf-8")

    vol = volatility.calculate_volatility(returns)
    pd.DataFrame([vol]).to_csv(proc / "volatility.csv", index=False, encoding="utf-8")
    rolling = volatility.calculate_rolling_volatility(returns, window=12)
    rolling.to_csv(proc / "eth_usd_rolling_volatility.csv", index=False, encoding="utf-8")

    mesa_df, summary = scenarios.build_historical_paths(monthly, N, cycle)
    mesa_df.to_csv(mesa_dir / "eth_price_mesa.csv", index=False, encoding="utf-8")
    summary.to_csv(mesa_dir / "historical_paths.csv", index=False, encoding="utf-8")

    check = scenarios.validate_mesa_file(
        pd.read_csv(mesa_dir / "eth_price_mesa.csv"), N)

    _plot_all(monthly, returns, rolling, PROJECT_ROOT / "results/figures")

    print("=== MONTHLY ===")
    print(f"months: {len(monthly)} | {monthly['month'].iloc[0]} -> {monthly['month'].iloc[-1]}")
    print(f"last monthly date used: {monthly['date'].iloc[-1].date()}")
    print("\n=== VOLATILITY (F9) ===")
    print(json.dumps(vol, indent=2))
    print("\n=== MESA FILE ===")
    print(json.dumps(check, indent=2))
    print("\nworst path by max_drawdown:")
    print(summary.sort_values("max_drawdown", ascending=False).head(3).to_string(index=False))
    if not check["ok"]:
        raise SystemExit("Mesa file validation FAILED")


if __name__ == "__main__":
    main()
