"""Local Solara dashboard for HuiChain experiments and scenario analysis."""
from __future__ import annotations

from html import escape
import math
from numbers import Number

import numpy as np
import pandas as pd
import solara

from src.utils import PROJECT_ROOT, load_config


SCENARIOS = [
    "Kịch bản bình thường",
    "ETH sập giá - không nạp thêm",
    "Người đã nhận hụi không thanh toán",
    "Thiếu ETH collateral",
]
TRANSLATION_MAP = {
    "HUI_BREAK": "VỠ HỤI (HUI_BREAK)", "HUI_COMPLETED": "HOÀN THÀNH (HUI_COMPLETED)",
    "COMPLETED": "Đã hoàn thành", "ACTIVE": "Đang hoạt động", "RECEIVED": "Đã nhận hụi", "DEFAULTED": "Đã vỡ nợ",
    "BEHAVIORAL_DEFAULT_SURPLUS_COLLATERAL": "Bỏ đóng do hành vi - còn dư tài sản đảm bảo",
    "BEHAVIORAL_DEFAULT_INSUFFICIENT_COLLATERAL": "Bỏ đóng do hành vi - thiếu tài sản đảm bảo",
    "RANDOM_DEFAULT": "Bỏ đóng ngẫu nhiên", "WINNER_SELECTED": "Đã chọn người thắng", "COLLATERAL_LOCKED": "Đã khóa tài sản đảm bảo",
    "BID": "Đặt thầu", "DEFAULT": "Bỏ đóng", "MARGIN_CALL": "Yêu cầu bổ sung tài sản đảm bảo", "ETH_TOPUP": "Nạp thêm ETH",
    "RECOVERY": "Thu hồi", "ETH_PRICE_CRASH": "ETH sập giá", "NO_ELIGIBLE_BIDDER_INSUFFICIENT_ETH": "Không đủ ETH để khóa tài sản đảm bảo",
    "BID_MAX": "Mức bid tối đa", "Event": "Sự kiện", "Value": "Giá trị (USD)", "Reason": "Nguyên nhân", "Agent": "Thành viên",
    "Bid rate": "Tỷ lệ bid", "Bid amount": "Số tiền bid (USD)", "Winner payout": "Tiền người thắng nhận (USD)",
    "Future obligation": "Nghĩa vụ tương lai (USD)", "Required collateral": "Tài sản đảm bảo yêu cầu (USD)",
    "Actual collateral": "Tài sản đảm bảo thực tế (USD)", "Coverage ratio": "Tỷ lệ bao phủ", "Margin call": "Margin call",
    "ETH value (USD)": "Giá trị ETH (USD)", "Recovery (USD)": "Thu hồi (USD)", "Residual loss": "Tổn thất còn lại (USD)", "Surplus": "Phần dư (USD)",
}
FIELD_HELP = {"Kỳ": ("Kỳ hụi", "Kỳ đóng góp và đấu giá."), "Ngày": ("Ngày", "Ngày mô phỏng."), "Người nhận": ("Người nhận hụi", "Thành viên thắng đấu giá."),
              "Trạng thái": ("Trạng thái", "Trạng thái hiện tại."), "Coverage ratio": ("Tỷ lệ bao phủ", "Tài sản đảm bảo thực tế chia yêu cầu."),
              "Margin call": ("Margin call", "Có nghĩa là cần bổ sung collateral."), "Event": ("Sự kiện", "Mốc quan trọng trong vòng đời dây hụi.")}
SUMMARY_HELP = {"Trạng thái": "Trạng thái cuối của dây hụi.", "Tổng failure": "Tổng số sự kiện vỡ hụi.", "Recovery (USD)": "Tổng tiền thu hồi.", "Tỷ lệ recovery": "Recovery chia nghĩa vụ bị bỏ lỡ."}


def translate(text):
    return "—" if text is None else TRANSLATION_MAP.get(str(text), str(text))


def _missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "nat", "none"}
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def format_or_dash(value):
    return "—" if _missing(value) else str(value)


def _number(value, decimals, suffix=""):
    if _missing(value):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return format_or_dash(value)
    return "—" if not math.isfinite(number) else f"{number:,.{decimals}f}{suffix}"


def format_usd(value): return _number(value, 2, " USD")
def format_eth(value): return _number(value, 4, " ETH")
def format_pct(value): return _number(float(value) * 100, 1, "%") if not _missing(value) else "—"
def format_ratio(value): return _number(value, 2)


USD_COLUMNS = {"Value", "eth_usd", "Giá ETH/USD", "bid_amount", "Bid amount", "winner_payout", "Winner payout", "future_obligation_usd", "Future obligation", "required_collateral_usd", "Required collateral", "actual_collateral_usd", "Actual collateral", "bid_collateral_usd", "Bid collateral (USD)", "total_recoverable_usd", "Total recoverable", "recovery_usd", "Recovery (USD)", "residual_loss_usd", "Residual loss", "Tổn thất còn lại (USD)", "surplus_usd", "Surplus", "eth_value_usd", "ETH value (USD)", "liquidity_benefit_usd", "Lợi ích thanh khoản", "capital_locked_usd", "Capital locked USD", "Vốn bị khóa (USD)", "Lợi ích thanh khoản (USD)", "Quỹ hệ thống (USD)", "Collateral đang khóa"}
ETH_COLUMNS = {"locked_eth", "ETH khóa", "ETH đang khóa", "eth_topup", "ETH đã nạp", "topups_eth", "ETH ban đầu", "initial_eth"}
PCT_COLUMNS = {"bid_rate", "Bid rate", "hui_break_rate", "Tỷ lệ HUI_BREAK", "default_rate", "Tỷ lệ default", "eth_crash_break_rate", "Tỷ lệ break do ETH", "recovery_rate", "Tỷ lệ recovery", "margin_call_rate", "Tỷ lệ margin call", "eth_topup_rate", "Tỷ lệ nạp ETH", "participation_rate", "Tỷ lệ tham gia", "default_probability", "Xác suất bỏ đóng", "crash_level", "Mức giảm ETH", "max_drawdown"}
RATIO_COLUMNS = {"coverage_ratio", "Coverage ratio", "r"}


def _table_value(column, value):
    if isinstance(value, str):
        return translate(value)
    if column == "margin_call":
        return "—" if _missing(value) else ("Có" if bool(value) else "Không")
    if column in USD_COLUMNS: return format_usd(value)
    if column in ETH_COLUMNS: return format_eth(value)
    if column in PCT_COLUMNS: return format_pct(value)
    if column in RATIO_COLUMNS: return format_ratio(value)
    return format_or_dash(value)


def _hover_table(frame):
    if frame.empty:
        return solara.HTML(unsafe_innerHTML="<p>Không có dữ liệu.</p>")
    headers = "".join(f'<th class="hui-head">{escape(translate(FIELD_HELP.get(c, (c, ""))[0]))}</th>' for c in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(f'<td class="hui-cell">{escape(_table_value(column, row[column]))}</td>' for column in frame.columns)
        rows.append(f"<tr>{cells}</tr>")
    markup = f'<div class="hui-table-wrap"><table class="hui-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    return solara.HTML(unsafe_innerHTML=markup)


def _reading(text):
    return solara.Markdown(f"**Cách đọc:** {text}")


def _run_simulation(scenario, seed):
    from src.model import HuiChainModel
    config = load_config(); pool, collateral, auction = config["pool"], config["collateral"], config["auction"]
    behavior, margin = config["behavior"], config["margin"]
    n = int(pool["N"]); cycle_days = int(pool.get("cycle_days", 30))
    daily = pd.read_csv(PROJECT_ROOT / "data/processed/eth_usd_daily.csv").sort_values("date").head(n * cycle_days)
    prices = daily["close"].astype(float).tolist()
    if scenario == "ETH sập giá - không nạp thêm": prices[10:] = [prices[0] * 0.40] * (len(prices) - 10)
    forced = {2: [1]} if scenario == "Người đã nhận hụi không thanh toán" else None
    schedule = None if not forced else {r: {member: (0.20 if member == r else 0.0) for member in range(r, n + 1)} for r in range(1, n)}
    return HuiChainModel(N=n, contribution_usd=float(pool["contribution_usd"]), r=float(collateral["r"]), bid_max=float(auction["bid_max"]), max_ratio=float(auction.get("max_ratio", 1.5)), initial_eth=0.01 if scenario == "Thiếu ETH collateral" else float(config["agent_endowment"]["initial_eth"]), initial_eth_source=config["agent_endowment"]["initial_eth_source"], cash_multiplier=float(config["agent_endowment"]["cash_multiplier"]), default_probability=float(behavior["default_probability"]), prices=prices, dates=daily["date"].astype(str).tolist(), seed=seed, bid_schedule=schedule, z=0.0, sigma_eth=0.0, haircut=0.0, insufficient_eth_policy=collateral.get("insufficient_eth_policy", "FILTER"), release_policy="AT_END", post_default_decision=config["post_default"]["th2_default_decision"], margin_check_interval_days=int(margin.get("margin_check_interval_days", 7)), cycle_days=cycle_days, margin_call_grace_periods=int(margin.get("margin_call_grace_periods", 0)), margin_topup_policy="NONE" if scenario == "ETH sập giá - không nạp thêm" else margin.get("margin_topup_policy", "AUTO"), crash_enabled=True, crash_threshold=float(config["crash"]["threshold"]), forced_defaults=forced).run()


def _research_tables(model):
    from src.metrics import agent_profile_rows, collateral_margin_series, default_recovery_rows, event_timeline, financial_flow
    flow = pd.DataFrame(financial_flow(model)).rename(columns={"round": "Kỳ", "date": "Ngày", "winner": "Người nhận", "bid_rate": "Bid rate", "bid_amount": "Bid amount", "winner_payout": "Winner payout", "future_obligation_usd": "Future obligation", "required_collateral_usd": "Required collateral"})
    margin = pd.DataFrame(collateral_margin_series(model)).rename(columns={"day": "Ngày mô phỏng", "date": "Ngày", "round": "Kỳ", "required_collateral_usd": "Required collateral", "actual_collateral_usd": "Actual collateral", "coverage_ratio": "Coverage ratio", "margin_call": "margin_call", "hui_status": "Trạng thái"})
    recovery = pd.DataFrame(default_recovery_rows(model)).rename(columns={"agent_id": "Agent", "round": "Kỳ", "date": "Ngày", "remaining_obligation_usd": "Future obligation", "bid_collateral_usd": "Bid collateral (USD)", "eth_value_usd": "ETH value (USD)", "total_recoverable_usd": "Total recoverable", "recovery_usd": "Recovery (USD)", "residual_loss_usd": "Residual loss", "surplus_usd": "Surplus", "reason": "Reason"})
    profile = pd.DataFrame(agent_profile_rows(model)).rename(columns={"agent_id": "Thành viên", "status": "Trạng thái", "received_round": "Kỳ nhận hụi", "bid_rate": "Bid rate", "bid_collateral_usd": "Bid collateral (USD)", "winner_payout_usd": "Winner payout", "contribution_paid_usd": "Contribution paid", "future_obligation_usd": "Future obligation", "required_collateral_usd": "Required collateral", "locked_eth": "ETH đang khóa", "eth_value_usd": "ETH value (USD)", "margin_calls": "Số lần margin call", "eth_topup": "ETH đã nạp", "recovery_usd": "Recovery (USD)", "residual_loss_usd": "Residual loss", "liquidity_benefit_usd": "Lợi ích thanh khoản"})
    events = pd.DataFrame(event_timeline(model)).rename(columns={"date": "Ngày", "round": "Kỳ", "agent_id": "Agent", "event": "Event", "value": "Value", "reason": "Reason"})
    if model.failure_type == "NO_ELIGIBLE_BIDDER_INSUFFICIENT_ETH" and not events.empty:
        row = model.round_log[-1]; events = pd.concat([events, pd.DataFrame([{"Ngày": row["date"], "Kỳ": row["round"], "Agent": (row.get("insufficient_eth_members") or [None])[0], "Event": "HUI_BREAK", "Value": 0.0, "Reason": "Không đủ ETH để khóa tài sản đảm bảo"}])], ignore_index=True)
    return flow, margin, recovery, profile, events[["Ngày", "Kỳ", "Agent", "Event", "Value", "Reason"]]


def _summary(model):
    from src.metrics import capital_locked_usd, liquidity_benefit_total, recovery_rate
    failures = model.post_default_summary()
    return {"Trạng thái": model.hui_status, "Số kỳ đã chạy": len(model.round_log), "Số ngày đã chạy": model.current_day if model.daily_mode else len(model.round_log), "Kỳ failure đầu tiên": model.first_failure_round or model.broken_round, "Tổng failure": failures["num_failures"], "Recovery (USD)": failures["recovery_usd"], "Collateral đang khóa": capital_locked_usd(model), "Lợi ích thanh khoản": liquidity_benefit_total(model), "Tỷ lệ recovery": recovery_rate(model)}


def _charts(model):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from src.metrics import collateral_margin_series
    plt.close("all")
    margin = pd.DataFrame(collateral_margin_series(model)); days = pd.DataFrame(model.daily_log)
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    x = days["day"] if not days.empty else [row["round"] for row in model.round_log]; price = days["eth_usd"] if not days.empty else [row["eth_usd"] for row in model.round_log]
    axes[0].plot(x, price, color="#e46b3a"); axes[0].set_title("ETH/USD và biến động giá"); axes[0].set_ylabel("USD"); axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_usd(value))); axes[0].grid(alpha=.2)
    axes[1].plot(margin["day"], margin["required_collateral_usd"], label="Yêu cầu", color="#d45745"); axes[1].plot(margin["day"], margin["actual_collateral_usd"], label="Thực tế", color="#0f6b68"); axes[1].set_title("Tài sản đảm bảo yêu cầu và thực tế"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    axes[2].plot(margin["day"], margin["coverage_ratio"].replace(float("inf"), 1.0), color="#4563a6"); axes[2].axhline(1.0, color="#78858a", linestyle="--"); axes[2].set_title("Tỷ lệ bao phủ tài sản đảm bảo"); axes[2].set_xlabel("Ngày"); axes[2].set_ylabel("Thực tế / Yêu cầu"); axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_ratio(value))); axes[2].grid(alpha=.2)
    return figure


SWEEP_PRESETS = {"Quick": 50, "Normal": 150, "Full": 500}
SWEEP_PARAMETER_LABELS = {"r": "Tỷ lệ r", "bid_max": "BID_MAX", "initial_eth": "ETH ban đầu", "default_probability": "Xác suất bỏ đóng", "crash_level": "Mức giảm ETH"}
SWEEP_METRICS = {"residual_loss_usd": "Tổn thất còn lại (USD)", "recovery_rate": "Tỷ lệ thu hồi", "hui_break_rate": "Tỷ lệ vỡ hụi", "capital_locked_usd": "Vốn bị khóa (USD)", "liquidity_benefit_usd": "Lợi ích thanh khoản (USD)"}
SWEEP_BASELINES = {"r": 1.0, "bid_max": 0.20, "initial_eth": 100.0, "default_probability": 0.05, "crash_level": None}
_SWEEP_CACHE = {}
_COMPARISON_MODEL_CACHE = {}


def _arange_values(minimum, maximum, step):
    try:
        minimum, maximum, step = float(minimum), float(maximum), float(step)
    except (TypeError, ValueError):
        return []
    if step <= 0 or maximum < minimum: return []
    return [round(float(value), 10) for value in np.arange(minimum, maximum + step * 0.5, step) if value <= maximum + 1e-9]


def _sweep_values(selected, ranges):
    return {name: ranges[name] if name in selected else [SWEEP_BASELINES[name]] for name in SWEEP_PARAMETER_LABELS}


def _experiment_output(seed, selected, ranges, repetitions):
    from src.experiments import run_sweep
    values = _sweep_values(selected, ranges)
    return run_sweep(r_values=values["r"], bid_max_values=values["bid_max"], initial_eth_values=values["initial_eth"], default_probability_values=values["default_probability"], crash_level_values=values["crash_level"], seeds=range(seed, seed + repetitions), output_dir=PROJECT_ROOT / "results")


def _sweep_model_from_row(row):
    from src.experiments import build_model
    run_id = str(row["run_id"])
    if run_id not in _COMPARISON_MODEL_CACHE:
        crash = None if _missing(row["crash_level"]) else float(row["crash_level"])
        _COMPARISON_MODEL_CACHE[run_id] = build_model(seed=int(row["seed"]), r=float(row["r"]), bid_max=float(row["BID_MAX"]), initial_eth=float(row["initial_eth"]), default_probability=float(row["default_probability"]), crash_level=crash)
    return _COMPARISON_MODEL_CACHE[run_id]


def _metric_value(value, metric):
    return format_pct(value) if metric.endswith("rate") else format_usd(value)


def _experiment_chart(aggregate, dimensions, metric="residual_loss_usd"):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.close("all")
    dimensions = dimensions[:2]
    columns = ["BID_MAX" if dimension == "bid_max" else dimension for dimension in dimensions]
    has_two_varying_dimensions = len(dimensions) == 2 and all(aggregate[column].nunique(dropna=False) > 1 for column in columns)
    if has_two_varying_dimensions:
        pivot = aggregate.pivot_table(index=columns[0], columns=columns[1], values=metric, aggfunc="mean")
        sample = aggregate.pivot_table(index=columns[0], columns=columns[1], values="n", aggfunc="sum")
        is_rate = metric.endswith("rate")
        data = pivot.fillna(0).to_numpy()
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True); image = axis.imshow(data, aspect="auto", cmap="RdYlGn_r" if is_rate else "YlOrRd", vmin=0 if is_rate else None, vmax=1 if is_rate else None)
        axis.set_xticks(range(len(pivot.columns)), labels=[format_or_dash(value) for value in pivot.columns]); axis.set_yticks(range(len(pivot.index)), labels=[format_or_dash(value) for value in pivot.index]); axis.set_xlabel(SWEEP_PARAMETER_LABELS[dimensions[1]]); axis.set_ylabel(SWEEP_PARAMETER_LABELS[dimensions[0]]); axis.set_title(f"Heatmap {SWEEP_METRICS[metric]}")
        for row_index in range(len(pivot.index)):
            for column_index in range(len(pivot.columns)):
                rate = pivot.iloc[row_index, column_index]
                sample_size = sample.iloc[row_index, column_index]
                axis.text(column_index, row_index, f"{_metric_value(rate, metric)}\n(n={int(sample_size)})", ha="center", va="center", fontsize=8)
            figure.colorbar(image, ax=axis, format=FuncFormatter(lambda value, _: _metric_value(value, metric))); return figure
    figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    if dimensions:
        plot_index = next((index for index, column in enumerate(columns) if aggregate[column].nunique(dropna=False) > 1), 0)
        plot_column = columns[plot_index]
        for value, group in aggregate.groupby(plot_column):
            axis.plot(group[plot_column], group[metric], marker="o", label=format_or_dash(value))
            for _, row in group.iterrows():
                axis.annotate(f"{_metric_value(row[metric], metric)}\n(n={int(row['n'])})", (row[plot_column], row[metric]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
        axis.set_xlabel(SWEEP_PARAMETER_LABELS[dimensions[plot_index]])
        if axis.lines:
            axis.legend(fontsize=8)
    else:
        rate = aggregate[metric].mean(); axis.bar(["Mức cơ sở"], [rate]); axis.annotate(f"{_metric_value(rate, metric)}\n(n={int(aggregate['n'].sum())})", (0, rate), ha="center", va="bottom", fontsize=8)
    axis.set_title(f"{SWEEP_METRICS[metric]} theo tham số"); axis.set_ylabel(SWEEP_METRICS[metric]); axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _metric_value(value, metric))); axis.grid(alpha=.2); return figure


def _experiment_table(aggregate):
    table = aggregate.copy()
    for column in ["hui_break_rate", "default_rate", "eth_crash_break_rate", "recovery_rate", "margin_call_rate", "eth_topup_rate", "participation_rate"]:
        if column in table: table[column] = table.apply(lambda row: f"{format_pct(row[column])} (n={int(row['n'])})", axis=1)
    return table.rename(columns={"n": "Cỡ mẫu", "initial_eth": "ETH ban đầu", "default_probability": "Xác suất bỏ đóng", "crash_level": "Mức giảm ETH", "hui_break_rate": "Tỷ lệ vỡ hụi", "default_rate": "Tỷ lệ bỏ đóng", "eth_crash_break_rate": "Tỷ lệ vỡ hụi do ETH", "recovery_rate": "Tỷ lệ thu hồi", "residual_loss_usd": "Tổn thất còn lại (USD)", "margin_call_rate": "Tỷ lệ yêu cầu bổ sung", "eth_topup_rate": "Tỷ lệ nạp ETH", "liquidity_benefit_usd": "Lợi ích thanh khoản", "capital_locked_usd": "Vốn bị khóa (USD)", "participation_rate": "Tỷ lệ tham gia"})


def _interpret(model):
    ending = "Đây là 1 lần chạy minh họa, không đại diện xu hướng chung - xem Parameter Sweep để kết luận."
    if model is None or not hasattr(model, "hui_status"): return f"Không đủ dữ liệu. {ending}"
    if model.hui_status != "HUI_BREAK": return f"Dây hụi hoàn thành trong lần chạy này; không ghi nhận vỡ hụi. {ending}"
    failures = getattr(model, "failure_events", []); failure = failures[0] if failures else {}; reason = failure.get("failure_type") or getattr(model, "failure_type", None)
    if reason in {"BEHAVIORAL_DEFAULT_INSUFFICIENT_COLLATERAL", "DEFAULT_WITH_INSUFFICIENT_COLLATERAL"}: text = "Dây hụi vỡ vì một thành viên bỏ đóng khi tài sản đảm bảo không đủ bù nghĩa vụ."
    elif reason == "ETH_PRICE_CRASH": text = "Dây hụi vỡ sau cú giảm giá ETH làm tỷ lệ bao phủ collateral xuống dưới mức yêu cầu."
    elif any(event.get("surplus_liquidation_usd", 0) > 0 for event in failures): text = "Có bỏ đóng nhưng collateral thanh lý vượt nghĩa vụ; phần vượt là surplus."
    elif reason == "NO_ELIGIBLE_BIDDER_INSUFFICIENT_ETH": text = "Dây hụi dừng vì người thắng không đủ ETH để khóa collateral; chưa phát sinh default hoặc recovery."
    else: text = "Không đủ dữ liệu để xác định nguyên nhân vỡ hụi một cách chắc chắn."
    return f"{text} {ending}"


def _comparison_chart(models):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.close("all")
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for label, model in models:
        daily = pd.DataFrame(model.daily_log); x = daily["day"] if not daily.empty else list(range(1, len(model.round_log) + 1)); prices = daily["eth_usd"] if not daily.empty else [row["eth_usd"] for row in model.round_log]; coverage = daily.get("coverage_ratio", pd.Series(1.0, index=daily.index)) if not daily.empty else [row.get("coverage_ratio", 1.0) for row in model.round_log]
        axes[0].plot(x, prices, label=label); axes[1].plot(x, coverage, label=label)
    axes[0].set_title("So sánh đường giá ETH/USD"); axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_usd(value))); axes[0].legend(fontsize=8); axes[0].grid(alpha=.2); axes[1].set_title("So sánh tỷ lệ bao phủ collateral"); axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_ratio(value))); axes[1].axhline(1.0, color="#78858a", linestyle="--"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.2); return figure


def _comparison_rows(models):
    from src.metrics import simulation_summary
    rows = []
    for label, model in models:
        summary = simulation_summary(model); failure = model.failure_events[0] if model.failure_events else {}
        rows.append({"Lần chạy": label, "Trạng thái cuối": translate(model.hui_status), "Nguyên nhân vỡ": translate(failure.get("failure_type") or model.failure_type) if (failure.get("failure_type") or model.failure_type) else "—", "Tổn thất còn lại (USD)": format_usd(summary["residual_loss_usd"]), "Thu hồi (USD)": format_usd(summary["recovery_usd"]), "Số lần margin call": summary["margin_call_count"], "Lợi ích thanh khoản (USD)": format_usd(summary["liquidity_benefit_usd"])})
    return rows


def _comparison_difference(models):
    losses = [sum(event.get("residual_loss_usd", 0) for event in model.failure_events) for _, model in models]; return f"Trong {len(models)} lần chạy, có {sum(model.hui_status == 'HUI_BREAK' for _, model in models)} lần vỡ hụi. Tổn thất thấp nhất thuộc về {models[losses.index(min(losses))][0]} ({format_usd(min(losses))})."


def _summary_grid(summary):
    return solara.HTML(unsafe_innerHTML="<div class='hui-summary-grid'>" + "".join(f"<div class='hui-summary-card'><div class='hui-summary-label'>{escape(translate(label))}</div><div class='hui-card-value'>{escape(_table_value(label, value))}</div></div>" for label, value in summary.items()) + "</div>")


def _result_panel(model):
    if model.hui_status == "COMPLETED": return solara.Markdown("### HOÀN THÀNH (HUI_COMPLETED)\n\nDây hụi đã hoàn thành.")
    reason = model.failure_events[0].get("failure_type") if model.failure_events else model.failure_type
    return solara.Markdown(f"### VỠ HỤI (HUI_BREAK)\n\nKỳ vỡ hụi: `{format_or_dash(model.broken_round)}`. Nguyên nhân: **{translate(reason)}**.")


@solara.component
def Page():
    mode, set_mode = solara.use_state("Một lần"); scenario, set_scenario = solara.use_state(SCENARIOS[0]); seed, set_seed = solara.use_state(42)
    history, set_history = solara.use_state([]); preset, set_preset = solara.use_state("Normal"); repetitions, set_repetitions = solara.use_state(150); dimensions, set_dimensions = solara.use_state(["r"]); metric, set_metric = solara.use_state("residual_loss_usd"); revision, set_revision = solara.use_state(0); comparison, set_comparison = solara.use_state([])
    r_min, set_r_min = solara.use_state(.5); r_max, set_r_max = solara.use_state(1.5); r_step, set_r_step = solara.use_state(.1)
    bid_min, set_bid_min = solara.use_state(.1); bid_max, set_bid_max = solara.use_state(.3); bid_step, set_bid_step = solara.use_state(.1)
    eth_min, set_eth_min = solara.use_state(50.); eth_max, set_eth_max = solara.use_state(150.); eth_step, set_eth_step = solara.use_state(50.)
    default_min, set_default_min = solara.use_state(0.); default_max, set_default_max = solara.use_state(.1); default_step, set_default_step = solara.use_state(.05)
    crash_min, set_crash_min = solara.use_state(.1); crash_max, set_crash_max = solara.use_state(.6); crash_step, set_crash_step = solara.use_state(.1)
    ranges = {"r": _arange_values(r_min, r_max, r_step), "bid_max": _arange_values(bid_min, bid_max, bid_step), "initial_eth": _arange_values(eth_min, eth_max, eth_step), "default_probability": _arange_values(default_min, default_max, default_step), "crash_level": _arange_values(crash_min, crash_max, crash_step)}
    values = _sweep_values(dimensions, ranges); total = repetitions * math.prod(len(items) for items in values.values()); result = _SWEEP_CACHE.get(revision); raw = result[0] if result else None; aggregate = result[1] if result else None
    solara.Style(":root{--hui-ink:#172b2d;--hui-line:#dbe6e4;--hui-teal:#0f6b68}*,*::before,*::after{box-sizing:border-box}html,body,#app,.v-application,.v-application__wrap,.v-main,.solara-content-main{width:100%!important;max-width:100%!important;min-width:0!important;overflow-x:hidden}.v-application__wrap,.v-main,.solara-content-main{flex:1 1 auto}.v-card,.v-sheet,.v-container,.v-row,.v-col,.v-input,.v-input__control,.v-field,.v-slider{max-width:100%;min-width:0}.hui-summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(155px,100%),1fr));gap:10px;margin:14px 0;max-width:100%}.hui-summary-card{min-width:0;padding:14px;border:1px solid var(--hui-line);border-radius:8px}.hui-summary-label{color:#617477;font-size:12px}.hui-card-value{font-size:20px;font-weight:700;color:var(--hui-teal);overflow-wrap:anywhere}.hui-table-wrap{width:100%;max-width:100%;min-width:0;overflow-x:auto;overflow-y:auto;max-height:520px;border:1px solid var(--hui-line)}.hui-table{border-collapse:collapse;min-width:760px;width:100%;font-size:12px}.hui-table th,.hui-table td{padding:8px;border:1px solid var(--hui-line);white-space:nowrap}.hui-table th{background:#e8f1ef}.hui-table-wrap+*{max-width:100%}")
    with solara.Card(title="Điều khiển phân tích"):
        solara.Select("Chế độ", values=["Một lần", "Thí nghiệm", "So sánh"], value=mode, on_value=set_mode); solara.SliderInt("Seed ngẫu nhiên", value=seed, min=0, max=999, on_value=set_seed)
        if mode == "Một lần":
            solara.Select("Kịch bản", values=SCENARIOS, value=scenario, on_value=set_scenario)
            solara.Button("Lưu lần chạy để so sánh", on_click=lambda: set_history(history + [{"label": f"Một lần #{len(history)+1} - seed {seed}", "model": _run_simulation(scenario, seed)}]), outlined=True)
    if mode == "Thí nghiệm":
        solara.Markdown("## Parameter Sweep\n\n**Kết quả mô phỏng, không phải giá trị tối ưu.**")
        solara.Select("Preset số lần lặp", values=list(SWEEP_PRESETS), value=preset, on_value=lambda value: (set_preset(value), set_repetitions(SWEEP_PRESETS[value])))
        solara.InputInt("Số lần lặp mỗi tổ hợp (tối thiểu 20)", value=repetitions, on_value=lambda value: set_repetitions(max(20, int(value or 20))), suffix="lần")
        solara.SelectMultiple("Chọn chiều sweep (tối đa 2)", values=dimensions, all_values=list(SWEEP_PARAMETER_LABELS), on_value=lambda selected: set_dimensions(selected[:2]))
        solara.Select("Chỉ số biểu đồ", values=list(SWEEP_METRICS), value=metric, on_value=set_metric)
        with solara.Card(title="Min / max / step"):
            for label, minimum, maximum, step, set_minimum, set_maximum, set_step in [("r", r_min, r_max, r_step, set_r_min, set_r_max, set_r_step), ("BID_MAX", bid_min, bid_max, bid_step, set_bid_min, set_bid_max, set_bid_step), ("ETH ban đầu", eth_min, eth_max, eth_step, set_eth_min, set_eth_max, set_eth_step), ("Xác suất bỏ đóng", default_min, default_max, default_step, set_default_min, set_default_max, set_default_step), ("Mức giảm ETH", crash_min, crash_max, crash_step, set_crash_min, set_crash_max, set_crash_step)]:
                solara.InputFloat(f"{label} min", value=minimum, on_value=set_minimum); solara.InputFloat(f"{label} max", value=maximum, on_value=set_maximum); solara.InputFloat(f"{label} step", value=step, on_value=set_step)
        solara.Markdown(f"**Tổng số simulation trước khi chạy:** {total:,} = số tổ hợp × {repetitions} lần lặp.")
        if total > 5000: solara.Markdown("**Cảnh báo:** vượt 5,000 simulation, hãy giảm dải hoặc số lần lặp.")
        def run_sweep():
            if 0 < total <= 5000:
                new_result = __import__("app")._experiment_output(seed, dimensions, ranges, repetitions); key = max(_SWEEP_CACHE, default=0) + 1; _SWEEP_CACHE[key] = new_result; set_revision(key)
        solara.Button("Chạy Parameter Sweep", on_click=run_sweep, disabled=total <= 0 or total > 5000, color="primary")
        if aggregate is not None:
            solara.FigureMatplotlib(_experiment_chart(aggregate, dimensions, metric), format="png"); _hover_table(_experiment_table(aggregate)); solara.Markdown(f"Đã chạy {len(raw)} simulation. Đang hiển thị: **{SWEEP_METRICS[metric]}**.")
        else: solara.Markdown("Chưa có kết quả sweep.")
        return
    if mode == "So sánh":
        candidates = {item["label"]: item["model"] for item in history}
        if raw is not None:
            for _, row in raw.iterrows():
                candidates[f"Sweep {row['run_id']}"] = None
        solara.Markdown(f"Có **{len(candidates)}** lần chạy có thể chọn. Chọn 2-4 dòng để dựng bảng và biểu đồ overlay.")
        solara.SelectMultiple("Chọn 2-4 lần chạy", values=[item for item in comparison if item in candidates][:4], all_values=list(candidates), on_value=lambda selected: set_comparison(selected[:4]))
        selected_models = []
        for label in comparison[:4]:
            if label in candidates and candidates[label] is not None: selected_models.append((label, candidates[label]))
        if raw is not None:
            for label in comparison[:4]:
                if label.startswith("Sweep "):
                    matching = raw[raw["run_id"] == label.replace("Sweep ", "")]
                    if not matching.empty:
                        selected_models.append((label, _sweep_model_from_row(matching.iloc[0])))
        if len(selected_models) >= 2:
            _hover_table(pd.DataFrame(_comparison_rows(selected_models))); solara.FigureMatplotlib(_comparison_chart(selected_models), format="png"); solara.Markdown(_comparison_difference(selected_models))
        else: solara.Markdown("Hãy chọn ít nhất 2 và tối đa 4 lần chạy.")
        return
    model = _run_simulation(scenario, seed); _result_panel(model); _summary_grid(_summary(model)); solara.FigureMatplotlib(_charts(model), format="png"); solara.Markdown(f"**Đọc tình huống:** {_interpret(model)}")
    flow, margin, recovery, profile, events = _research_tables(model)
    solara.Markdown("## Dòng tiền theo vòng"); _hover_table(flow); solara.Markdown("Mỗi dòng là một kỳ, gồm người thắng, bid, payout và nghĩa vụ còn lại.")
    solara.Markdown("## Collateral và margin"); _hover_table(margin); solara.Markdown("Tỷ lệ bao phủ dưới 1.00 nghĩa là cần margin call.")
    solara.Markdown("## Vỡ nợ và thu hồi"); _hover_table(recovery); solara.Markdown("Thu hồi là phần được bù từ collateral; tổn thất còn lại lớn hơn 0 nghĩa là collateral không đủ.")
    solara.Markdown("## Hồ sơ thành viên"); _hover_table(profile); solara.Markdown("Đang hoạt động, đã nhận hụi và đã vỡ nợ là các trạng thái chính của agent.")
    solara.Markdown("## Dòng thời gian sự kiện"); _hover_table(events); solara.Markdown("Dùng timeline để truy vết trình tự sự kiện.")