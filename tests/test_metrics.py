import pytest

from src.metrics import capital_locked_usd, liquidity_benefit_total, recovery_rate
from src.model import HuiChainModel


def test_metrics_on_completed_daily_model():
    model = HuiChainModel(
        N=5, contribution_usd=100, r=1, bid_max=.2, max_ratio=1.5,
        initial_eth=1, initial_eth_source="PRE_OWNED", cash_multiplier=2,
        default_probability=0, prices=[3000.0] * 150,
        dates=[str(i) for i in range(150)], cycle_days=30,
    ).run()
    assert capital_locked_usd(model) == pytest.approx(0.0)
    assert liquidity_benefit_total(model) > 0
    assert recovery_rate(model) == pytest.approx(1.0)