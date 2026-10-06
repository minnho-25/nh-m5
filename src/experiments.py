"""Single-run and parameter-sweep output for HuiChain research."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from src.metrics import simulation_summary
from src.utils import PROJECT_ROOT, load_config


def _daily_path(n, cycle_days):
	data = pd.read_csv(PROJECT_ROOT / "data/processed/eth_usd_daily.csv").sort_values("date")
	data = data.head(n * cycle_days)
	if len(data) < n * cycle_days:
		raise ValueError("Không đủ dữ liệu daily cho số kỳ yêu cầu")
	return data["close"].astype(float).tolist(), data["date"].astype(str).tolist()


def build_model(*, seed=42, r=None, bid_max=None, initial_eth=None,
				default_probability=None, crash_level=None, path_id="DAILY_0001"):
	from src.model import HuiChainModel

	config = load_config()
	pool = config["pool"]
	endowment = config["agent_endowment"]
	collateral = config["collateral"]
	auction = config["auction"]
	behavior = config["behavior"]
	margin = config["margin"]
	crash = config["crash"]
	n = int(pool["N"])
	cycle_days = int(pool.get("cycle_days", 30))
	prices, dates = _daily_path(n, cycle_days)
	if crash_level is not None:
		prices[10:] = [prices[0] * (1.0 - float(crash_level))] * (len(prices) - 10)
	return HuiChainModel(
		N=n,
		contribution_usd=float(pool["contribution_usd"]),
		r=float(collateral["r"] if r is None else r),
		bid_max=float(auction["bid_max"] if bid_max is None else bid_max),
		max_ratio=float(auction.get("max_ratio", 1.5)),
		initial_eth=float(endowment["initial_eth"] if initial_eth is None else initial_eth),
		initial_eth_source=endowment["initial_eth_source"],
		cash_multiplier=float(endowment["cash_multiplier"]),
		default_probability=float(behavior["default_probability"] if default_probability is None else default_probability),
		prices=prices,
		dates=dates,
		path_id=path_id,
		seed=seed,
		z=0.0,
		sigma_eth=0.0,
		haircut=0.0,
		insufficient_eth_policy=collateral.get("insufficient_eth_policy", "FILTER"),
		release_policy="AT_END",
		post_default_decision=config["post_default"]["th2_default_decision"],
		margin_check_interval_days=int(margin.get("margin_check_interval_days", 7)),
		cycle_days=cycle_days,
		margin_call_grace_periods=int(margin.get("margin_call_grace_periods", 0)),
		margin_topup_policy=margin.get("margin_topup_policy", "AUTO"),
		crash_enabled=bool(crash.get("enabled", True)),
		crash_threshold=float(crash.get("threshold", 0.20)),
	).run()


def run_single(*, seed=42, output_dir=None, **params):
	model = build_model(seed=seed, **params)
	row = simulation_summary(model, run_id=f"single_{seed}", seed=seed)
	if output_dir is not None:
		path = Path(output_dir)
		path.mkdir(parents=True, exist_ok=True)
		pd.DataFrame([row]).to_csv(path / "simulation_results.csv", index=False)
	return model, row


def run_sweep(*, r_values=(0.8, 1.0, 1.2), bid_max_values=(0.1, 0.2),
			  initial_eth_values=(50.0, 100.0), default_probability_values=(0.0, 0.05),
			  crash_level_values=(None, 0.6), seeds=(42,), output_dir=None):
	rows = []
	for values in product(r_values, bid_max_values, initial_eth_values,
						default_probability_values, crash_level_values, seeds):
		r, bid_max, initial_eth, default_probability, crash_level, seed = values
		model = build_model(seed=seed, r=r, bid_max=bid_max, initial_eth=initial_eth,
						   default_probability=default_probability, crash_level=crash_level)
		row = simulation_summary(model, run_id=f"sweep_{len(rows):05d}", seed=seed)
		row.update({
			"crash_level": crash_level,
			"initial_eth": initial_eth,
			"default_probability": default_probability,
		})
		rows.append(row)
	raw = pd.DataFrame(rows)
	group_cols = ["r", "BID_MAX", "initial_eth", "default_probability", "crash_level"]
	aggregated = raw.groupby(group_cols, dropna=False).agg(
		n=("hui_break", "size"),
		hui_break_rate=("hui_break", "mean"),
		default_rate=("default_count", lambda values: (values > 0).mean()),
		eth_crash_break_rate=("eth_crash_break", "mean"),
		recovery_rate=("recovery_rate", "mean"),
		residual_loss_usd=("residual_loss_usd", "mean"),
		margin_call_rate=("margin_call_count", lambda values: (values > 0).mean()),
		eth_topup_rate=("eth_topup_count", lambda values: (values > 0).mean()),
		liquidity_benefit_usd=("liquidity_benefit_usd", "mean"),
		capital_locked_usd=("capital_locked_usd", "mean"),
		participation_rate=("participation_rate", "mean"),
	).reset_index()
	if output_dir is None:
		output_dir = PROJECT_ROOT / "results"
	output_dir = Path(output_dir)
	(output_dir / "raw").mkdir(parents=True, exist_ok=True)
	(output_dir / "aggregated").mkdir(parents=True, exist_ok=True)
	raw.to_csv(output_dir / "raw/simulation_results.csv", index=False)
	aggregated.to_csv(output_dir / "aggregated/experiment_summary.csv", index=False)
	return raw, aggregated
