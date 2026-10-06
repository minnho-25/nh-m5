# HuiChain

Mô hình Mesa ABM cho dây hụi phi tập trung theo tài liệu `Hụichaincoban.pdf`.

## Chạy dashboard local

```bash
source .venv/bin/activate
./.venv/bin/python run_dashboard.py
```

Mở `http://127.0.0.1:8765`.

Dashboard chạy dữ liệu ETH/USD daily trong `data/processed/eth_usd_daily.csv`:

- 1 Mesa step = 1 ngày;
- contribution và auction diễn ra mỗi `pool.cycle_days` ngày;
- margin check diễn ra mỗi `margin.margin_check_interval_days` ngày;
- xem được ETH path, collateral, top-up, margin call, default, recovery và trạng thái hụi.

## Kiểm tra

```bash
./.venv/bin/python -m pytest -q tests/test_daily_model.py tests/test_metrics.py tests/test_basic_spec.py
```

Các tham số nghiên cứu và giả định nằm trong `config/config.yaml`.
