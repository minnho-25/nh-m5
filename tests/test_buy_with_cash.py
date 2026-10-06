import pytest

from src.model import HuiChainModel

SCHEDULE = {
    1: {3: 0.10, 1: 0.05, 2: 0.05, 4: 0.05, 5: 0.05},
    2: {1: 0.08, 2: 0.02, 4: 0.02, 5: 0.02},
    3: {5: 0.06, 2: 0.01, 4: 0.01},
    4: {2: 0.04, 4: 0.01},
}
SIGMA, Z = 0.2318747, 1.645


def _model(**kw):
    params = dict(
        N=5, contribution_usd=1000.0, r=1.0, bid_max=0.20, max_ratio=1.5,
        initial_eth=100.0, initial_eth_source="BUY_WITH_CASH", cash_multiplier=4.0,
        default_probability=0.0, prices=[1000.0] * 5,
        dates=[f"2020-0{i + 1}-28" for i in range(5)], seed=42,
        bid_schedule=SCHEDULE, z=Z, sigma_eth=SIGMA, haircut=0.2,
        eth_purchase_fraction=0.5,
    )
    params.update(kw)
    return HuiChainModel(**params)


def test_purchase_accounting_at_start():
    m = _model()
    for a in m.members:
        assert a.cash_balance_usd == pytest.approx(10000.0)     # 20000 - 10000
        assert a.total_eth == pytest.approx(10.0)               # 10000 / 1000
        assert a.available_eth == pytest.approx(10.0)
        assert a.locked_eth == pytest.approx(0.0)
        assert a.initial_eth == pytest.approx(10.0)
    assert m.eth_purchase_total_usd == pytest.approx(50000.0)
    assert m.eth_purchase_total_eth == pytest.approx(50.0)
    assert m.total_initial_cash == pytest.approx(50000.0)


def test_purchase_uses_first_path_price():
    m = _model(prices=[1000.0, 4000.0, 4000.0, 4000.0, 4000.0])
    assert all(a.total_eth == pytest.approx(10.0) for a in m.members)   # khong phai 2.5


def test_pre_owned_has_no_purchase():
    m = _model(initial_eth_source="PRE_OWNED", cash_multiplier=2.0)
    assert m.eth_purchase_total_usd == 0.0
    assert all(a.total_eth == pytest.approx(100.0) for a in m.members)
    assert all(a.cash_balance_usd == pytest.approx(10000.0) for a in m.members)


def test_full_run_after_purchase_conserves_money_and_eth():
    m = _model().run()
    m.check_invariants()
    assert m.hui_status == "COMPLETED"
    cash_total = sum(a.cash_balance_usd for a in m.members) + m.system_fund_usd
    assert cash_total == pytest.approx(50000.0)                 # khong tao them tien
    for a in m.members:
        assert a.total_eth == pytest.approx(10.0)
        assert a.locked_eth == pytest.approx(0.0, abs=1e-9)
        assert a.cash_balance_usd >= 0


def test_default_after_purchase_conserves_eth():
    m = _model(forced_defaults={2: [3]}, post_default_decision="CONTINUE_SUB_HUI").run()
    m.check_invariants()
    liq = sum(e["liquidation_value_usd"] for e in m.failure_events)
    cash_total = sum(a.cash_balance_usd for a in m.members) + m.system_fund_usd
    assert cash_total == pytest.approx(50000.0 + liq)
    for a in m.members:
        assert a.total_eth + a.collateral_liquidated_eth == pytest.approx(10.0)


def test_invalid_source_and_fraction():
    with pytest.raises(ValueError):
        _model(initial_eth_source="MAGIC")
    with pytest.raises(ValueError):
        _model(eth_purchase_fraction=1.5)
    with pytest.raises(ValueError):
        _model(eth_purchase_fraction=-0.1)
