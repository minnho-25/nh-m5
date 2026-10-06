import pytest

from src.agents import DEFAULTED
from src.model import HuiChainModel

SIGMA, Z = 0.2318747, 1.645
CRASHY = [2000.0, 2200.0, 900.0, 800.0, 700.0]
CALM = [2000.0] * 5


def _run(seed, decision):
    return HuiChainModel(
        N=5, contribution_usd=1000.0, r=1.0, bid_max=0.20, max_ratio=1.5,
        initial_eth=100.0, initial_eth_source="PRE_OWNED", cash_multiplier=2.0,
        default_probability=0.35, prices=CRASHY if seed % 3 == 0 else CALM,
        dates=[f"2020-0{i + 1}-28" for i in range(5)], seed=seed, bid_schedule=None,
        z=Z, sigma_eth=SIGMA, haircut=0.05 + 0.01 * (seed % 20),
        bid_tick=0.05 if seed % 2 else 0.0, post_default_decision=decision,
        margin_topup_policy="NONE" if seed % 4 == 0 else "AUTO",
        margin_call_grace_periods=seed % 2,
    ).run()                                  # check_invariants chạy mỗi round -> AssertionError nếu vi phạm


@pytest.mark.parametrize("decision", ["CONTINUE_SUB_HUI", "TERMINATE_AND_REFUND"])
def test_fuzz_invariants(decision):
    n_events = 0
    for seed in range(60):
        m = _run(seed, decision)
        assert m.hui_status in ("COMPLETED", "BROKEN")
        winners = [x["winner"] for x in m.round_log if x["winner"] is not None]
        assert len(winners) == len(set(winners))                  # không ai nhận hụi hai lần
        assert m.system_fund_usd >= -1e-6
        eth_total = sum(a.total_eth for a in m.members) + sum(a.collateral_liquidated_eth for a in m.members)
        assert eth_total == pytest.approx(500.0)                  # ETH không tự sinh/mất
        for e in m.failure_events:
            n_events += 1
            assert e["recovery_usd"] + e["surplus_liquidation_usd"] == pytest.approx(
                e["liquidation_value_usd"], abs=1e-6)             # không dùng liquidation hai lần
            assert e["recovery_usd"] <= e["obligation_missed_usd"] + 1e-9
            assert e["residual_loss_usd"] >= -1e-9 and e["system_loss_usd"] >= -1e-9
            assert e["total_refund_usd"] + e["system_surplus_usd"] == pytest.approx(
                e["refund_pool_usd"], abs=1e-6)
            if e["branch"] == "TH1":
                assert e["system_loss_usd"] == pytest.approx(e["residual_loss_usd"], abs=1e-9)
            later = [x for x in m.round_log if x["round"] > e["round"]]
            assert all(e["member"] not in x["eligible_bidders"] and x["winner"] != e["member"]
                       for x in later)                            # DEFAULTED không bid / không thắng
        for a in m.members:
            if a.status == DEFAULTED:
                assert a.contribution_paid_usd <= 1000.0 * a.defaulted_round
    assert n_events > 0                                           # fuzz thực sự tạo ra failure
