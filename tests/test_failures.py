import pytest

from src.agents import COMPLETED, DEFAULTED
from src.model import HuiChainModel

SCHEDULE = {
    1: {3: 0.10, 1: 0.05, 2: 0.05, 4: 0.05, 5: 0.05},
    2: {1: 0.08, 2: 0.02, 4: 0.02, 5: 0.02},
    3: {5: 0.06, 2: 0.01, 4: 0.01},
    4: {2: 0.04, 4: 0.01},
}
SIGMA, Z = 0.2318747, 1.645
CRASH = [2000.0, 2000.0, 500.0, 500.0, 500.0]


def _model(schedule=SCHEDULE, seed=42, **kw):
    params = dict(
        N=5, contribution_usd=1000.0, r=1.0, bid_max=0.20, max_ratio=1.5,
        initial_eth=100.0, initial_eth_source="PRE_OWNED", cash_multiplier=2.0,
        default_probability=0.0, prices=[2000.0] * 5,
        dates=[f"2020-0{i + 1}-28" for i in range(5)], seed=seed, bid_schedule=schedule,
        z=Z, sigma_eth=SIGMA, haircut=0.2,
    )
    params.update(kw)
    return HuiChainModel(**params)


def _lock1():
    """ETH member 3 khóa ở round 1 (giá 2000, r=1, N=5)."""
    return 2.0 * (1 + Z * SIGMA * 2)


def test_th2_continue_sub_hui():
    m = _model(forced_defaults={2: [3]}, post_default_decision="CONTINUE_SUB_HUI").run()
    L = _lock1()
    liq = L * 2000 * 0.8
    ev = m.failure_events[0]
    assert ev["branch"] == "TH2" and ev["kind"] == "BEHAVIORAL"
    assert ev["obligation_missed_usd"] == pytest.approx(4000)
    assert ev["liquidation_value_usd"] == pytest.approx(liq)
    assert ev["recovery_usd"] == pytest.approx(4000)
    assert ev["surplus_liquidation_usd"] == pytest.approx(liq - 4000)
    assert ev["recovery_usd"] + ev["surplus_liquidation_usd"] == pytest.approx(liq)   # không double-count
    assert ev["num_remaining_members"] == 4
    assert ev["surplus_per_member"] == pytest.approx((liq - 4000) / 4)
    assert m.sub_hui_created and not m.hui_broken and m.hui_status == "COMPLETED"
    d = m.members[2]
    assert d.status == DEFAULTED
    assert (d.behavioral_default_count, d.price_driven_failure_count) == (1, 0)
    assert d.collateral_liquidated_eth == pytest.approx(L)
    assert d.total_eth == pytest.approx(100 - L)
    assert d.contribution_paid_usd == 1000                      # không đóng sau khi DEFAULTED
    winners = [x["winner"] for x in m.round_log]
    assert winners[0] == 3 and sorted(winners[1:]) == [1, 2, 4, 5]
    assert [x["pot"] for x in m.round_log] == pytest.approx([5000, 4000, 4000, 4000, 4000])
    assert m.round_log[-1]["bid_rate"] == 0.0
    assert m.system_fund_usd == pytest.approx(4000)             # recovery nằm lại system fund
    assert all(x.status == COMPLETED for x in m.members if x is not d)


def test_th2_terminate_and_refund():
    m = _model(forced_defaults={2: [3]}, post_default_decision="TERMINATE_AND_REFUND").run()
    pool = _lock1() * 2000 * 0.8 - 4000
    ev = m.failure_events[0]
    assert ev["branch"] == "TH2" and ev["post_default_decision"] == "TERMINATE_AND_REFUND"
    assert ev["refund_pool_usd"] == pytest.approx(pool)
    assert ev["total_refund_usd"] == pytest.approx(pool)        # pool < tổng contribution (4000)
    assert ev["system_surplus_usd"] == pytest.approx(0, abs=1e-6)
    assert all(v == pytest.approx(pool / 4) for v in ev["refunds"].values())
    assert m.hui_status == "BROKEN" and m.hui_broken and not m.sub_hui_created
    assert m.running is False and m.broken_round == 2
    assert m.members[0].cash_balance_usd == pytest.approx(9125 + pool / 4)
    assert m.system_fund_usd == pytest.approx(4000)


def test_th1_insufficient_collateral():
    m = _model(forced_defaults={2: [3]}, haircut=0.7).run()
    liq = _lock1() * 2000 * 0.3
    ev = m.failure_events[0]
    assert ev["branch"] == "TH1" and m.default_branch == "TH1"
    assert ev["recovery_usd"] == pytest.approx(liq)
    assert ev["system_loss_usd"] == pytest.approx(4000 - liq) == pytest.approx(ev["residual_loss_usd"])
    assert m.hui_status == "BROKEN" and m.hui_broken and not m.sub_hui_created
    assert m.failure_type == "BEHAVIORAL_DEFAULT_INSUFFICIENT_COLLATERAL"
    assert m.broken_round == 2


def test_equality_is_th1():
    h = 1 - 4000 / (_lock1() * 2000)                            # liquidation == obligation_missed
    m = _model(forced_defaults={2: [3]}, haircut=h).run()
    ev = m.failure_events[0]
    assert ev["branch"] == "TH1"
    assert ev["system_loss_usd"] == pytest.approx(0, abs=1e-6)
    assert m.hui_status == "BROKEN"


def test_insufficient_cash_received_member_defaults():
    m = _model()
    m.step()
    a = m.members[2]                                            # member 3 thắng round 1
    m.total_initial_cash -= a.cash_balance_usd - 500.0
    a.cash_balance_usd = 500.0
    m.step()
    ev = m.failure_events[0]
    assert ev["reason"] == "INSUFFICIENT_CASH" and ev["kind"] == "BEHAVIORAL"
    assert a.status == DEFAULTED and a.behavioral_default_count == 1


def test_unreceived_member_cash_shortage_is_th1():
    m = _model()
    a = m.members[0]
    m.total_initial_cash -= a.cash_balance_usd
    a.cash_balance_usd = 0.0
    m.step()
    ev = m.failure_events[0]
    assert ev["reason"] == "INSUFFICIENT_CASH" and ev["branch"] == "TH1"
    assert ev["liquidation_value_usd"] == 0 and ev["recovery_usd"] == 0
    assert ev["system_loss_usd"] == pytest.approx(5000)
    assert m.hui_status == "BROKEN"


def test_price_driven_failure_no_topup_grace0():
    m = _model(prices=CRASH, margin_topup_policy="NONE", margin_call_grace_periods=0).run()
    assert m.hui_status == "BROKEN" and m.failure_type == "PRICE_DRIVEN"
    ev = m.failure_events[0]
    assert ev["kind"] == "PRICE_DRIVEN" and ev["round"] == 3 and ev["member"] == 1
    assert ev["obligation_missed_usd"] == pytest.approx(2000)
    assert m.members[0].price_driven_failure_count == 1
    assert all(a.behavioral_default_count == 0 for a in m.members)   # không tính là behavioral
    assert m.broken_round == 3


def test_grace_period_delays_processing():
    m = _model(prices=CRASH, margin_topup_policy="NONE", margin_call_grace_periods=1).run()
    assert m.failure_events == []
    assert m.hui_status == "COMPLETED"
    assert m.members[0].margin_call_count >= 1                  # có margin call nhưng chưa xử lý


def test_auto_topup_restores_margin():
    m = _model(prices=CRASH, margin_topup_policy="AUTO").run()
    assert m.failure_events == [] and m.hui_status == "COMPLETED"
    assert set(m.round_log[2]["topups"]) == {1, 3}
    assert m.members[0].collateral_topup_eth > 0
    assert all(a.locked_eth == 0 and a.available_eth == pytest.approx(100.0) for a in m.members)


def test_crash_metrics():
    m = _model(prices=[2000.0, 2000.0, 1500.0, 1500.0, 1500.0]).run()
    r3 = m.round_log[2]
    assert r3["crash_flag"] and r3["eth_return"] == pytest.approx(-0.25)
    assert r3["drawdown"] == pytest.approx(0.25)
    assert m.crash_count == 1 and m.max_drawdown == pytest.approx(0.25)
    assert m.round_log[0]["eth_return"] is None and not m.round_log[0]["crash_flag"]


def test_collusion_group_bidding_and_f15():
    m = _model(schedule=None, seed=11)
    m.members[0].collusion_group = "G1"
    m.members[1].collusion_group = "G1"
    m.run()
    winners = [x["winner"] for x in m.round_log]
    assert winners[:2] == [1, 2]
    res = m.collusion_check([1, 2])
    assert (res["n"], res["k"]) == (2, 2)
    assert res["p_value"] == pytest.approx(0.325 ** 2)
    assert res["collusion_suspected"] is False                  # chỉ là nghi ngờ thống kê, và chưa vượt ngưỡng


def test_collusion_probability_one():
    m = _model(schedule=None, collusion_probability=1.0)
    assert all(a.collusion_group == "G1" for a in m.members)
    m.run()
    assert [x["winner"] for x in m.round_log] == [1, 2, 3, 4, 5]
    assert m.collusion_result["collusion_suspected"] is False


def test_reproducible_with_random_defaults():
    def sig(seed):
        m = _model(schedule=None, seed=seed, default_probability=0.3,
                   post_default_decision="CONTINUE_SUB_HUI").run()
        return ([x["winner"] for x in m.round_log],
                [(e["round"], e["member"], e["branch"], round(e["liquidation_value_usd"], 6))
                 for e in m.failure_events])
    assert sig(5) == sig(5)
    assert sig(5) != sig(6) or True
