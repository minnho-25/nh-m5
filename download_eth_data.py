from src.utils import load_config, PROJECT_ROOT
from src import eth_data


def main():
    cfg = load_config()
    m = cfg["market"]

    print(f"Đang tải {m['ticker']} từ {m['start_date']} ...")
    df = eth_data.download_eth_daily(m["ticker"], m["start_date"], m.get("end_date"))

    report = eth_data.validate_daily(df)
    print("\n=== VALIDATION REPORT ===")
    for k, v in report.items():
        print(f"{k:>24}: {v}")
    if not report["ok"]:
        raise SystemExit("Validation FAILED, không lưu file.")

    raw_path = eth_data.save_raw_csv(df, PROJECT_ROOT / "data/raw/eth_usd_daily_raw.csv")
    eth_data.write_metadata(
        PROJECT_ROOT / "data/metadata.json",
        source=m["source"], ticker=m["ticker"], start_date=m["start_date"], df=df,
        frequency=m["frequency"], price_column=m["price_column"],
        monthly_price_method=m["monthly_price_method"],
    )
    fig_path = eth_data.plot_daily_price(df, PROJECT_ROOT / "results/figures/eth_usd_daily.png")

    print(f"\nĐã lưu: {raw_path}")
    print(f"Đã lưu: {PROJECT_ROOT / 'data/metadata.json'}")
    print(f"Đã lưu: {fig_path}")


if __name__ == "__main__":
    main()
