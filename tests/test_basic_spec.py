import pytest

from src.model import HuiChainModel
from src import mechanisms as mech


def make_model(**overrides):
    params = dict(
        N=5, contribution_usd=100.0, r=1.0, bid_max=0.20, max_ratio=1.5,
        initial_eth=1.0, initial_eth_source="PRE_OWNED", cash_multiplier=2.0,
        default_probability=0.0, prices=[3000.0] * 5,
        dates=[f"2026-10-0{i + 1}" for i in range(5)],
        bid_schedule={1: {1: 0.20, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
                      2: {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
                      3: {3: 0.0, 4: 0.0, 5: 0.0},
                      4: {4: 0.0, 5: 0.0}},
    )
    params.update(overrides)
    return HuiChainModel(**params)


def test_basic_bid_collateral_is_subtracted_from_eth_requirement():
    model = make_model(initial_eth=0.1).run()
    first = model.round_log[0]
    assert first["bid_collateral"] == pytest.approx(100.0)
    assert first["additional_collateral_usd"] == pytest.approx(300.0)
    assert first["collateral_eth_actual"] == pytest.approx(0.1)
    assert first["discount_shares"] == {}


def test_tied_bid_uses_earliest_join_order():
    model = make_model(bid_schedule={1: {1: 0.1, 2: 0.1, 3: 0, 4: 0, 5: 0},
                                     2: {2: 0, 3: 0, 4: 0, 5: 0},
                                     3: {3: 0, 4: 0, 5: 0},
                                     4: {4: 0, 5: 0}}).run()
    assert model.round_log[0]["winner"] == 1
    assert model.round_log[0]["tie_break"]["rule"] == "earliest_join_order"


def test_completion_refunds_bid_collateral_and_keeps_liquidity_metric():
    model = make_model().run()
    assert model.hui_status == "COMPLETED"
    assert model.system_fund_usd == pytest.approx(0.0)
    assert sum(a.cash_balance_usd for a in model.members) == pytest.approx(5000.0)
    assert model.members[0].liquidity_benefit_usd == pytest.approx(300.0)


def test_basic_collateral_formulas():
    assert mech.bid_collateral(100) == 100
    assert mech.additional_collateral(400, 100) == 300
    assert mech.eth_collateral_value(0.1, 3000) == pytest.approx(300)
    assert mech.total_collateral_value(100, 0.1, 3000) == pytest.approx(400)


def test_margin_check_is_periodic():
    model = make_model(N=14, prices=[3000.0] * 14,
                       dates=[str(i) for i in range(14)],
                       initial_eth=10.0, bid_schedule=None,
                       margin_check_interval_days=7).run()
    assert all(record["margin_ratios"] == {} for record in model.round_log[:6])
    assert model.round_log[6]["margin_ratios"] or model.round_log[13]["margin_ratios"]