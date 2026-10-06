import pytest

from src.agents import DEFAULTED
from src.model import HuiChainModel

SCHEDULE = {
    1: {3: 0.10, 1: 0.05, 2: 0.05, 4: 0.05, 5: 0.05},
    2: {1: 0.08, 2: 0.02, 4: 0.02, 5: 0.02},
    3: {5: 0.06, 2: 0.01, 4: 0.01},
    4: {2: 0.04, 4: 0.01},
}
SIGMA, Z = 0.2318747, 1.645
N, C, INITIAL_ETH, CASH_MULT = 5, 1000.0, 100.0, 2.0


def _margin_kw(grace=0, topup="NONE", threshold=1.15):
    """!!! Sua ten tham so o DAY neu __init__ cua ban dat ten khac."""
    return dict(
        margin_call_grace_periods=grace,
        margin_topup_policy=topup,
        maintenance_threshold=threshold,
    )


def _model(schedule=SCHEDULE, seed=42, **kw):
    params = dict(
        N=N, contribution_usd=C, r=1.0, bid_max=0.20, max_ratio=1.5,
        initial_eth=INITIAL_ETH, initial_eth_source="PRE_OWNED",
        cash_multiplier=CASH_MULT, default_probability=0.0,
        prices=[2000.0] * 5,
        dates=[f"2020-0{i + 1}-28" for i in range(5)],
        seed=seed, bid_schedule=schedule, z=Z, sigma_eth=SIGMA, haircut=0.2,
    )
    params.update(kw)
    return HuiChainModel(**params)


def _lock1():
    """ETH member 3 khoa o round 1 (gia 2000, r=1, N=5)."""
    return 2.0 * (1 + Z * SIGMA * 2)


# ---------- Round N tuong minh ----------
def test_round_n_no_bid_no_collateral_full_pot():
    m = _model().run()
    last = m.round_log[-1]
    assert last["bid_rate"] == 0.0
    assert last["pot"] == pytest.approx(N * C)
    assert last["payout"] == pytest.approx(N * C)       # doi key neu round_log dat ten khac
    assert all(a.locked_eth == pytest.approx(0, abs=1e-9) for a in m.members)


# ---------- TH2 + terminate khi refund_pool > tong contribution ----------
def test_terminate_refund_with_system_surplus():
    prices = [2000.0, 4000.0, 4000.0, 4000.0, 4000.0]
    m = _model(prices=prices, forced_defaults={2: [3]},
               post_default_decision="TERMINATE_AND_REFUND").run()
    ev = m.failure_events[0]
    liq = _lock1() * 4000.0 * 0.8
    pool = liq - 4000.0
    assert pool > 4000.0
    assert ev["refund_pool_usd"] == pytest.approx(pool)
    assert ev["total_refund_usd"] == pytest.approx(4000.0)
    assert ev["system_surplus_usd"] == pytest.approx(pool - 4000.0)
    assert ev["total_refund_usd"] + ev["system_surplus_usd"] == pytest.approx(pool)
    assert all(v <= 1000.0 + 1e-9 for v in ev["refunds"].values())
    assert m.hui_broken and not m.sub_hui_created


# ---------- F10 dung gia luc thanh ly, khong dung gia luc khoa ----------
def test_f10_uses_liquidation_price_not_lock_price():
    prices = [2000.0, 3000.0, 3000.0, 3000.0, 3000.0]
    m = _model(prices=prices, forced_defaults={2: [3]},
               post_default_decision="CONTINUE_SUB_HUI").run()
    ev = m.failure_events[0]
    assert ev["liquidation_value_usd"] == pytest.approx(_lock1() * 3000.0 * 0.8)
    assert ev["liquidation_value_usd"] != pytest.approx(_lock1() * 2000.0 * 0.8)


# ---------- Bao toan tien, bao toan ETH, khong nhan hui 2 lan ----------
@pytest.mark.parametrize("decision", ["CONTINUE_SUB_HUI", "TERMINATE_AND_REFUND"])
@pytest.mark.parametrize("seed", [1, 7, 42])
def test_money_eth_conservation_after_default(decision, seed):
    m = _model(seed=seed, forced_defaults={2: [3]}, post_default_decision=decision).run()
    m.check_invariants()
    liq_total = sum(e["liquidation_value_usd"] for e in m.failure_events)
    cash_total = sum(a.cash_balance_usd for a in m.members) + m.system_fund_usd
    assert cash_total == pytest.approx(N * N * C * CASH_MULT + liq_total)
    for a in m.members:
        assert a.locked_eth == pytest.approx(0, abs=1e-9)
        assert a.available_eth >= -1e-9 and a.cash_balance_usd >= -1e-9
        assert a.available_eth + a.locked_eth == pytest.approx(a.total_eth)
        assert a.total_eth + a.collateral_liquidated_eth == pytest.approx(INITIAL_ETH)
    winners = [x["winner"] for x in m.round_log if x.get("winner")]
    assert len(winners) == len(set(winners))
    assert m.members[2].status == DEFAULTED


# ---------- Nhieu default lien tiep ----------
def test_multiple_sequential_defaults_continue_sub_hui():
    m = _model(forced_defaults={2: [3], 3: [1]},
               post_default_decision="CONTINUE_SUB_HUI").run()
    m.check_invariants()
    assert len(m.failure_events) == 2
    assert all(e["kind"] == "BEHAVIORAL" for e in m.failure_events)
    assert m.members[2].status == DEFAULTED and m.members[0].status == DEFAULTED
    s = m.post_default_summary()
    assert s["num_behavioral_defaults"] == 2 and s["num_price_driven_failures"] == 0
    for e in m.failure_events:
        assert e["recovery_usd"] + e["residual_loss_usd"] == pytest.approx(e["obligation_missed_usd"])
        if e["branch"] == "TH2":
            assert e["recovery_usd"] + e["surplus_liquidation_usd"] == pytest.approx(e["liquidation_value_usd"])
    assert m.failure_events[1]["num_remaining_members"] == 3


# ---------- Grace 0/1/2, price-driven khong tinh la behavioral ----------
CRASH = [2000.0, 2000.0, 300.0, 300.0, 300.0]


def _first_failure(grace):
    m = _model(prices=CRASH, **_margin_kw(grace=grace)).run()
    m.check_invariants()
    return m, m.post_default_summary()["first_failure_round"]


def test_grace_zero_is_immediate_and_price_driven():
    m, ff0 = _first_failure(0)
    assert ff0 is not None
    s = m.post_default_summary()
    assert s["num_price_driven_failures"] >= 1
    assert s["num_behavioral_defaults"] == 0
    assert all(a.behavioral_default_count == 0 for a in m.members)
    assert m.failure_events[0]["kind"] == "PRICE_DRIVEN"


@pytest.mark.parametrize("grace", [1, 2])
def test_grace_delays_failure(grace):
    _, ff0 = _first_failure(0)
    m, ffg = _first_failure(grace)
    assert ffg is None or ffg > ff0
    assert m.post_default_summary()["num_behavioral_defaults"] == 0


# ---------- Crash manh: nhieu failure / cascade ----------
def test_severe_crash_multiple_locked_members():
    prices = [2000.0, 2000.0, 100.0, 100.0, 100.0]
    m = _model(prices=prices, **_margin_kw(grace=0)).run()
    m.check_invariants()
    s = m.post_default_summary()
    assert s["num_failures"] >= 1
    assert s["num_behavioral_defaults"] == 0
    print("cascade_count =", s["cascade_count"], "| failures =", s["num_failures"])
