import pytest

from src.model import HuiChainModel


def make_daily(**overrides):
    prices = [3000.0] * 150
    params = dict(
        N=5,
        contribution_usd=100.0,
        r=1.0,
        bid_max=0.20,
        max_ratio=1.5,
        initial_eth=1.0,
        initial_eth_source="PRE_OWNED",
        cash_multiplier=2.0,
        default_probability=0.0,
        prices=prices,
        dates=[f"2026-01-{i + 1:03d}" for i in range(150)],
        cycle_days=30,
        margin_check_interval_days=7,
        margin_topup_policy="AUTO",
        post_default_decision="CONTINUE_SUB_HUI",
    )
    params.update(overrides)
    return HuiChainModel(**params)


def test_daily_timeline_runs_rounds_every_cycle_days():
    model = make_daily().run()
    assert model.daily_mode
    assert model.current_round == 5
    assert model.current_day == 121
    assert [row["day"] for row in model.daily_log if row["round"] > 0][:3] == [1, 2, 3]
    assert model.hui_status == "COMPLETED"


def test_margin_check_happens_on_day_seven():
    model = make_daily().run()
    checkpoints = [row for row in model.daily_log if row["day"] % 7 == 0]
    assert checkpoints[0]["day"] == 7
    assert len(checkpoints) >= 17


def test_eth_crash_break_is_recorded_as_price_failure():
    prices = [3000.0] * 150
    prices[10:] = [1200.0] * 140
    model = make_daily(prices=prices, margin_topup_policy="NONE").run()
    assert model.hui_status == "HUI_BREAK"
    assert model.failure_type == "ETH_PRICE_CRASH"
    assert model.failure_events[0]["reason"] == "ETH_PRICE_CRASH"
    assert model.failure_events[0]["kind"] == "PRICE_DRIVEN"


def test_daily_insufficient_eth_refunds_cancelled_contribution():
    model = make_daily(initial_eth=0.001).run()
    assert model.hui_status == "HUI_BREAK"
    assert model.current_day == 1
    assert sum(agent.contribution_paid_usd for agent in model.members) == pytest.approx(0.0)