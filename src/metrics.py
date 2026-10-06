"""Observable metrics for the Basic HuiChain model."""

from __future__ import annotations


def capital_locked_usd(model, price=None):
	"""Current ETH collateral value plus retained bid collateral."""
	mark = model.current_eth_usd if price is None else price
	return sum(
		agent.bid_collateral_usd + agent.locked_eth * mark
		for agent in model.members
	)


def total_collateral_usd(model, price=None):
	return capital_locked_usd(model, price)


def liquidity_benefit_total(model):
	return sum(agent.liquidity_benefit_usd for agent in model.members)


def recovery_rate(model):
	summary = model.post_default_summary()
	missed = summary["obligation_missed_usd"]
	return 1.0 if missed <= 0 else summary["recovery_usd"] / missed


def daily_metrics(model):
	"""Return one compact row per daily observation for dashboard use."""
	rows = []
	for item in model.daily_log:
		rows.append({
			"day": item["day"],
			"date": item["date"],
			"round": item["round"],
			"eth_usd": item["eth_usd"],
			"locked_eth": item["locked_eth_total"],
			"margin_calls": len(item["margin_calls"]),
			"topups_eth": sum(item["topups"].values()),
			"crash_flag": item["crash_flag"],
			"hui_status": item["hui_status"],
		})
	return rows


def financial_flow(model):
	"""Round-level cash flow and obligation table."""
	return [{
		"round": row["round"],
		"date": row["date"],
		"winner": row["winner"],
		"bid_rate": row["bid_rate"],
		"bid_amount": row.get("bid_collateral", row.get("discount", 0.0)),
		"winner_payout": row["payout"],
		"future_obligation_usd": row["future_obligation_usd"],
		"required_collateral_usd": row.get("collateral_usd", 0.0),
	} for row in model.round_log]


def collateral_margin_series(model):
	"""Daily required/actual collateral and coverage ratio."""
	if model.daily_log:
		return [{
			"day": row["day"],
			"date": row["date"],
			"round": row["round"],
			"required_collateral_usd": row.get("required_collateral_usd", 0.0),
			"actual_collateral_usd": row.get("total_collateral_usd", 0.0),
			"coverage_ratio": row.get("coverage_ratio", float("inf")),
			"margin_call": bool(row["margin_calls"]),
			"hui_status": row["hui_status"],
		} for row in model.daily_log]
	return [{
		"day": row["round"],
		"date": row["date"],
		"round": row["round"],
		"required_collateral_usd": row.get("collateral_usd", 0.0),
		"actual_collateral_usd": row.get("bid_collateral", 0.0) + row.get("collateral_eth_actual", 0.0) * row["eth_usd"],
		"coverage_ratio": 1.0,
		"margin_call": bool(row.get("margin_calls")),
		"hui_status": row["hui_status"],
	} for row in model.round_log]


def default_recovery_rows(model):
	return [{
		"agent_id": event["member"],
		"round": event["round"],
		"date": event.get("date", model.current_date),
		"remaining_obligation_usd": event["obligation_missed_usd"],
		"bid_collateral_usd": event["bid_collateral_usd"],
		"eth_value_usd": event["eth_collateral_value_usd"],
		"total_recoverable_usd": event["total_recoverable_usd"],
		"recovery_usd": event["recovery_usd"],
		"residual_loss_usd": event["residual_loss_usd"],
		"surplus_usd": event["surplus_liquidation_usd"],
		"reason": event["reason"],
	} for event in model.failure_events]


def agent_profile_rows(model):
	price = model.current_eth_usd or 0.0
	return [{
		"agent_id": agent.join_order,
		"status": agent.status,
		"received_round": agent.received_round,
		"bid_rate": agent.bid_rate,
		"bid_collateral_usd": agent.bid_collateral_usd,
		"winner_payout_usd": agent.winner_payout_usd,
		"contribution_paid_usd": agent.contribution_paid_usd,
		"future_obligation_usd": agent.future_obligation_usd,
		"required_collateral_usd": (agent.collateral_ratio or 0.0) * agent.future_obligation_usd,
		"locked_eth": agent.locked_eth,
		"eth_value_usd": agent.locked_eth * price,
		"margin_calls": agent.margin_call_count,
		"eth_topup": agent.collateral_topup_eth,
		"recovery_usd": agent.recovery_usd,
		"residual_loss_usd": agent.residual_loss_usd,
		"liquidity_benefit_usd": agent.liquidity_benefit_usd,
	} for agent in model.members]


def event_timeline(model):
	return [event for event in model.event_log if event["event"] in {
		"BID", "WINNER_SELECTED", "COLLATERAL_LOCKED", "MARGIN_CALL",
		"ETH_TOPUP", "DEFAULT", "RECOVERY", "HUI_BREAK", "HUI_COMPLETED",
	}]


def eth_risk_summary(model):
	prices = list(model.prices[:model.current_day] if model.daily_mode else model.prices[:len(model.round_log)])
	if not prices:
		return {"peak_price": None, "trough_price": None, "max_drawdown": 0.0, "price_at_failure": None}
	failure_price = model.failure_events[0]["price"] if model.failure_events else None
	return {
		"peak_price": max(prices),
		"trough_price": min(prices),
		"max_drawdown": model.max_drawdown,
		"price_at_failure": failure_price,
	}


def simulation_summary(model, *, run_id="single", seed=None):
	failure = model.failure_events[0] if model.failure_events else {}
	post = model.post_default_summary()
	return {
		"run_id": run_id,
		"seed": seed if seed is not None else model.seed,
		"eth_path_id": model.path_id or "local",
		"N": model.N,
		"C": model.C,
		"r": model.r,
		"BID_MAX": model.bid_max,
		"completed": model.hui_status == "COMPLETED",
		"hui_break": model.hui_status == "HUI_BREAK",
		"failure_reason": failure.get("failure_type"),
		"failure_round": failure.get("round"),
		"default_count": post["num_behavioral_defaults"],
		"margin_call_count": sum(agent.margin_call_count for agent in model.members),
		"eth_topup_count": sum(1 for event in model.event_log if event["event"] == "ETH_TOPUP"),
		"max_drawdown": model.max_drawdown,
		"recovery_usd": post["recovery_usd"],
		"recovery_rate": recovery_rate(model),
		"residual_loss_usd": post["residual_loss_usd"],
		"surplus_usd": post["surplus_liquidation_usd"],
		"liquidity_benefit_usd": liquidity_benefit_total(model),
		"capital_locked_usd": max((row.get("total_collateral_usd", 0.0) for row in model.daily_log), default=capital_locked_usd(model)),
		"eth_crash_break": bool(failure.get("failure_type") == "ETH_PRICE_CRASH"),
		"participation_rate": sum(agent.status != "DEFAULTED" for agent in model.members) / model.N,
	}
