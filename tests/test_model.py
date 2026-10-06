import pytest

from src.agents import COMPLETED
from src.model import HuiChainModel
from src.scenarios import load_path
from src.utils import PROJECT_ROOT

# Lịch bid deterministic: winner lần lượt là member 3, 1, 5, 2, rồi 4 (round N)
SCHEDULE = {
    1: {3: 0.10, 1: 0.05, 2: 0.05, 4: 0.05, 5: 0.05},
    2: {1: 0.08, 2: 0.02, 4: 0.02, 5: 0.02},
    3: {5: 0.06, 2: 0.01, 4: 0.01},
    4: {2: 0.04, 4: 0.01},
}


def _model(schedule=None, seed=42, **kw):
    params = dict(
        N=5, contribution_usd=1000.0, r=1.0, bid_max=0.20, max_ratio=1.5,
        initial_eth=100.0, initial_eth_source="PRE_OWNED", cash_multiplier=2.0,
        default_probability=0.0, prices=[2000.0] * 5,
        dates=[f"2020-0{i + 1}-28" for i in range(5)], seed=seed, bid_schedule=schedule,
    )
    params.update(kw)
    return HuiChainModel(**params)


def test_happy_path_deterministic():
    m = _model(SCHEDULE).run()
    log = m.round_log
    assert [x["winner"] for x in log] == [3, 1, 5, 2, 4]
    assert [x["pot"] for x in log] == [5000] * 5
    assert [x["discount"] for x in log] == pytest.approx([500, 400, 300, 200, 0])
    assert [x["payout"] for x in log] == pytest.approx([4500, 4600, 4700, 4800, 5000])
    assert [x["future_obligation_usd"] for x in log] == [4000, 3000, 2000, 1000, 0]
    assert [x["collateral_usd"] for x in log] == [4000, 3000, 2000, 1000, 0]
    assert log[-1]["bid_rate"] == 0.0                        # round N: không bid
    cash = {a.join_order: a.cash_balance_usd for a in m.members}
    assert cash[1] == pytest.approx(9725)
    assert cash[2] == pytest.approx(10208.3333333, rel=1e-6)
    assert cash[3] == pytest.approx(9500)
    assert cash[4] == pytest.approx(10608.3333333, rel=1e-6)
    assert cash[5] == pytest.approx(9958.3333333, rel=1e-6)
    assert sum(cash.values()) == pytest.approx(50000)        # không tạo tiền
    assert all(a.status == COMPLETED for a in m.members)
    assert m.hui_status == "COMPLETED"


def test_invariants_many_seeds():
    for seed in range(30):
        m = _model(seed=seed).run()
        winners = [x["winner"] for x in m.round_log]
        assert sorted(winners) == [1, 2, 3, 4, 5]            # mỗi người nhận đúng 1 lần
        assert all(x["payout"] >= 0 for x in m.round_log)
        assert m.round_log[-1]["bid_rate"] == 0.0
        assert m.round_log[-1]["payout"] == pytest.approx(5000)
        assert sum(a.cash_balance_usd for a in m.members) == pytest.approx(50000)
        assert all(a.locked_eth >= 0 and a.available_eth >= 0 for a in m.members)


def test_reproducible_same_seed():
    a = _model(seed=7).run().round_log
    b = _model(seed=7).run().round_log
    assert [x["winner"] for x in a] == [x["winner"] for x in b]
    assert [x["bid_rate"] for x in a] == [x["bid_rate"] for x in b]


def test_bid_above_bid_max_rejected():
    bad = {1: {1: 0.50}}
    with pytest.raises(ValueError):
        _model(bad).run()



def test_runs_on_real_historical_path():
    if not (PROJECT_ROOT / "data/mesa/eth_price_mesa.csv").exists():
        pytest.skip("chưa có file Mesa")
    prices, dates = load_path("PATH_0001")
    m = _model(prices=prices, dates=dates, path_id="PATH_0001").run()
    assert m.hui_status == "COMPLETED"
    assert [x["eth_usd"] for x in m.round_log] == pytest.approx(prices)


# ---------------------- Phase 5 ----------------------
from src.volatility import load_sigma_eth  # noqa: E402

SIGMA, Z = 0.2318747, 1.645


def test_eth_collateral_locked_after_round_1():
    m = _model(SCHEDULE, z=Z, sigma_eth=SIGMA)
    m.step()
    rec = m.round_log[0]
    buf = Z * SIGMA * 2
    assert rec["collateral_eth_base"] == pytest.approx(2.0)          # 4000 / 2000
    assert rec["buffer"] == pytest.approx(buf)
    assert rec["collateral_eth_actual"] == pytest.approx(2.0 * (1 + buf))
    w = m.members[2]                                                  # member 3 thắng round 1
    assert w.locked_eth == pytest.approx(2.0 * (1 + buf))
    assert w.available_eth == pytest.approx(100 - 2.0 * (1 + buf))
    assert w.total_eth == pytest.approx(100.0)
    assert m.members[0].locked_eth == 0.0


def test_all_collateral_released_at_end():
    m = _model(SCHEDULE, z=Z, sigma_eth=SIGMA).run()
    assert all(a.locked_eth == 0 for a in m.members)
    assert all(a.available_eth == pytest.approx(100.0) for a in m.members)
    locked_total = sum(x["collateral_eth_actual"] for x in m.round_log)
    assert sum(a.collateral_released_eth for a in m.members) == pytest.approx(locked_total)
    assert m.round_log[-1]["collateral_eth_actual"] == 0.0           # round N: không collateral


def test_margin_call_on_price_crash_is_not_default():
    m = _model(SCHEDULE, z=Z, sigma_eth=SIGMA,
               prices=[2000.0, 2000.0, 500.0, 500.0, 500.0]).run()
    assert 3 in m.round_log[2]["margin_calls"]                       # ratio ~0.88 < 1.15
    assert m.members[2].margin_call_count >= 1
    assert all(a.behavioral_default_count == 0 for a in m.members)
    assert m.hui_status == "COMPLETED"


def test_insufficient_eth_breaks_hui_without_moving_money():
    m = _model(SCHEDULE, initial_eth=1.0).run()                      # cần 2.0 ETH, chỉ có 1.0
    assert m.hui_status == "BROKEN"
    assert m.failure_type == "NO_ELIGIBLE_BIDDER_INSUFFICIENT_ETH"
    assert m.broken_round == 1
    assert m.round_log[0]["winner"] is None
    assert all(a.cash_balance_usd == pytest.approx(10000) for a in m.members)


def test_error_policy_raises():
    with pytest.raises(RuntimeError):
        _model(SCHEDULE, initial_eth=1.0, insufficient_eth_policy="ERROR").run()


def test_sigma_loaded_from_data_and_real_crash_path():
    if not (PROJECT_ROOT / "data/processed/volatility.csv").exists():
        pytest.skip("chưa có volatility.csv")
    sigma = load_sigma_eth()
    assert 0.05 < sigma < 0.6
    if not (PROJECT_ROOT / "data/mesa/eth_price_mesa.csv").exists():
        pytest.skip("chưa có file Mesa")
    prices, dates = load_path("PATH_0026")
    m = _model(prices=prices, dates=dates, path_id="PATH_0026", z=Z, sigma_eth=sigma).run()
    assert m.hui_status == "COMPLETED"
    assert all(a.locked_eth == 0 for a in m.members)


# ---------------------- Phase 6 ----------------------
import math  # noqa: E402


def test_f12_tie_break_in_model():
    sched = {1: {1: 0.10, 2: 0.10, 3: 0.05, 4: 0.05, 5: 0.05}}
    m = _model(sched)
    m.members[0].risk_tolerance = 1.4
    m.members[1].risk_tolerance = 1.2
    m.step()
    rec = m.round_log[0]
    assert rec["winner"] == 1
    assert rec["tie_break"]["winning_ratio"] == pytest.approx(1.3)
    assert rec["collateral_ratio"] == pytest.approx(1.3)
    assert rec["collateral_usd"] == pytest.approx(5200)              # 1.3 * 4000
    assert rec["collateral_eth_base"] == pytest.approx(2.6)          # 5200 / 2000


def test_f14_liquidity_benefit_in_model():
    m = _model(SCHEDULE).run()
    lb = {a.join_order: a.liquidity_benefit_usd for a in m.members}
    assert lb == pytest.approx({3: -500, 1: 600, 5: 1700, 2: 2800, 4: 4000})


def test_f16_progressive_release_in_model():
    m = _model(SCHEDULE, z=Z, sigma_eth=SIGMA)
    m.step()
    m.step()
    lock1 = 2.0 * (1 + Z * SIGMA * 2)
    required2 = 1.5 * (1 + Z * SIGMA * math.sqrt(3))
    assert lock1 > required2
    assert m.members[2].locked_eth == pytest.approx(required2)
    assert m.round_log[1]["released"][3] == pytest.approx(lock1 - required2)
    assert m.members[2].collateral_released_eth == pytest.approx(lock1 - required2)


def test_release_at_end_policy():
    m = _model(SCHEDULE, z=Z, sigma_eth=SIGMA, release_policy="AT_END")
    for _ in range(4):
        m.step()
    assert m.members[2].locked_eth == pytest.approx(2.0 * (1 + Z * SIGMA * 2))   # chưa release
    m.run()
    assert all(a.locked_eth == 0 for a in m.members)
    locked_total = sum(x["collateral_eth_actual"] for x in m.round_log)
    assert sum(a.collateral_released_eth for a in m.members) == pytest.approx(locked_total)


def test_f15_collusion_check_on_run():
    m = _model(SCHEDULE).run()
    res = m.collusion_check([1, 2])
    assert (res["n"], res["k"]) == (4, 2)
    assert res["p_bar"] == pytest.approx((0.4 + 0.5 + 1 / 3 + 0.5) / 4)
    assert res["collusion_suspected"] is False


def test_bid_tick_creates_ties_and_ratio_in_range():
    n_tie = 0
    for seed in range(30):
        m = _model(seed=seed, bid_tick=0.05, z=Z, sigma_eth=SIGMA).run()
        assert m.hui_status == "COMPLETED"
        for rec in m.round_log:
            if rec["tie_break"]:
                n_tie += 1
                assert 1.0 <= rec["collateral_ratio"] <= 1.5 + 1e-9
        assert all(a.locked_eth == 0 for a in m.members)
        assert sum(a.cash_balance_usd for a in m.members) == pytest.approx(50000)
    assert n_tie > 0
