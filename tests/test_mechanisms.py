import pytest

from src import mechanisms as m


def test_f1_total_pot():
    assert m.total_pot(5, 1000) == 5000


def test_f2_bid_range():
    assert m.validate_bid_rate(0.0, 0.2) == 0.0
    assert m.validate_bid_rate(0.2, 0.2) == 0.2
    with pytest.raises(ValueError):
        m.validate_bid_rate(0.21, 0.2)
    with pytest.raises(ValueError):
        m.validate_bid_rate(-0.01, 0.2)


def test_f3_f4_discount_and_payout():
    pot = 5000
    d = m.discount_amount(0.10, pot)
    assert d == pytest.approx(500)
    assert m.winner_payout(pot, d) == pytest.approx(pot - d) == pytest.approx(4500)


def test_f5_distribution_conserves_discount():
    s = m.distribute_discount(500, [1, 2, 4, 5])
    assert s == {1: 125, 2: 125, 4: 125, 5: 125}
    assert sum(s.values()) == pytest.approx(500)
    assert m.distribute_discount(0, []) == {}


def test_f6_future_obligation():
    assert m.future_obligation(1, 5, 1000) == 4000
    assert m.future_obligation(4, 5, 1000) == 1000
    assert m.future_obligation(5, 5, 1000) == 0
    with pytest.raises(ValueError):
        m.future_obligation(0, 5, 1000)
    with pytest.raises(ValueError):
        m.future_obligation(6, 5, 1000)


def test_f7_collateral_usd():
    assert m.collateral_usd(1, 5, 1000, 1.0) == 4000
    assert m.collateral_usd(1, 5, 1000, 1.1) == pytest.approx(4400)
    assert m.collateral_usd(5, 5, 1000, 1.5) == 0     # round N: không collateral


def test_f8_eth_conversion():
    assert m.collateral_eth_base(4000, 2000) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        m.collateral_eth_base(4000, 0)


def test_f9_buffer_and_actual_eth():
    assert m.safety_buffer(1.645, 0.2319, 5, 1) == pytest.approx(1.645 * 0.2319 * 2)
    assert m.safety_buffer(1.645, 0.2319, 5, 5) == 0.0          # round N: sqrt(0)
    assert m.safety_buffer(0.0, 0.2319, 5, 1) == 0.0            # z = 0
    assert m.collateral_eth_actual(2.0, 0.5) == pytest.approx(3.0)


def test_f10_liquidation_not_above_spot():
    spot = 10 * 100
    assert m.liquidation_value_usd(10, 100, 0.2) == pytest.approx(800)
    assert m.liquidation_value_usd(10, 100, 0.2) <= spot
    assert m.liquidation_value_usd(10, 100, 0.0) == pytest.approx(spot)
    with pytest.raises(ValueError):
        m.liquidation_value_usd(10, 100, 1.0)


def test_f11_margin_ratio_and_call():
    ratio = m.margin_ratio(2, 2000, 3000)
    assert ratio == pytest.approx(4000 / 3000)
    assert not m.is_margin_call(ratio, 1.15)
    assert m.is_margin_call(1.0, 1.15)
    assert m.margin_ratio(2, 2000, 0) == float("inf")


# ---------------------- Phase 6 ----------------------
def test_f12_collateral_tie_break():
    r = m.collateral_tie_break([(1, 1.4), (2, 1.2)], 1.5, 0.10)
    assert r["winner"] == 1 and r["winning_ratio"] == pytest.approx(1.3)
    r = m.collateral_tie_break([(1, 1.2), (2, 1.2)], 1.5, 0.10)       # hòa tiếp -> join_order nhỏ
    assert r["winner"] == 1 and r["winning_ratio"] == pytest.approx(1.2)
    r = m.collateral_tie_break([(3, 1.5), (4, 1.45)], 1.5, 0.10)      # bị cap bởi MAX_RATIO
    assert r["winner"] == 3 and r["winning_ratio"] == pytest.approx(1.5)
    r = m.collateral_tie_break([(1, 2.0), (2, 1.0)], 1.5, 0.10)       # risk_tolerance > MAX_RATIO
    assert r["willing_max"][1] == 1.5 and r["winning_ratio"] == pytest.approx(1.1)
    with pytest.raises(ValueError):
        m.collateral_tie_break([(1, 1.2)], 1.5, 0.10)


def test_f13_recovery_and_residual():
    assert m.recovery_and_residual(4000, 6000) == (4000, 0)
    assert m.recovery_and_residual(4000, 2500) == (2500, 1500)
    assert m.recovery_and_residual(4000, 4000) == (4000, 0)
    with pytest.raises(ValueError):
        m.recovery_and_residual(-1, 100)


def test_f14_liquidity_benefit():
    assert m.liquidity_benefit(4500, 1000, 4000) == pytest.approx(-500)
    assert m.liquidity_benefit(5000, 1000, 0) == pytest.approx(4000)


def test_f15_binomial_and_suspicion():
    assert m.binomial_sf(10, 0.5, 10) == pytest.approx(1 / 1024)
    assert m.binomial_sf(3, 0.5, 2) == pytest.approx(0.5)
    assert m.binomial_sf(5, 0.3, 0) == 1.0
    assert m.binomial_sf(5, 0.3, 6) == 0.0
    res = m.collusion_suspicion(10, 0.2, 8, 0.05)
    assert res["p_value"] < 0.001 and res["collusion_suspected"] is True
    res0 = m.collusion_suspicion(0, 0.0, 0, 0.05)
    assert res0["p_value"] == 1.0 and res0["collusion_suspected"] is False


def test_f16_required_and_release():
    assert m.required_eth_safe(1.0, 3000, 2000, 0.5) == pytest.approx(2.25)
    assert m.release_amount(5.0, 3.0) == pytest.approx(2.0)
    assert m.release_amount(2.0, 3.0) == 0.0          # required > locked -> không release


# ---------------------- Phase 7 ----------------------
def test_classify_post_default():
    assert m.classify_post_default(6000, 4000) == ("TH2", 2000)
    assert m.classify_post_default(2500, 4000) == ("TH1", 0.0)
    assert m.classify_post_default(4000, 4000) == ("TH1", 0.0)       # bằng nhau -> TH1
    rec, res = m.recovery_and_residual(4000, 6000)
    assert rec + m.classify_post_default(6000, 4000)[1] == pytest.approx(6000)   # không double-count


def test_refund_distribution():
    refunds, tot, sys_s = m.refund_distribution(200, {1: 100, 2: 300})
    assert refunds == {1: pytest.approx(50), 2: pytest.approx(150)}
    assert tot == pytest.approx(200) and sys_s == pytest.approx(0)
    refunds, tot, sys_s = m.refund_distribution(2000, {1: 1000, 2: 1000})
    assert refunds == {1: 1000, 2: 1000} and tot == 2000 and sys_s == 0 + 0.0 or sys_s == pytest.approx(0)
    refunds, tot, sys_s = m.refund_distribution(2000, {1: 100, 2: 300})
    assert tot == pytest.approx(400) and sys_s == pytest.approx(1600)
    assert tot + sys_s == pytest.approx(2000)
    assert m.refund_distribution(500, {}) == ({}, 0.0, 500)
