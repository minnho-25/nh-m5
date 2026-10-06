"""MemberAgent (Mesa 3.x: Agent(model); unique_id do Mesa tự gán)."""
from mesa import Agent

from src.mechanisms import validate_bid_rate

ACTIVE = "ACTIVE"
RECEIVED = "RECEIVED"
DEFAULTED = "DEFAULTED"
COMPLETED = "COMPLETED"
MARGIN_CALL = "MARGIN_CALL"  # MARGIN_CALL != DEFAULT (margin call được theo dõi bằng streak, không đổi status)


class MemberAgent(Agent):
    def __init__(self, model, join_order, initial_cash_usd, initial_eth,
                 risk_tolerance, default_probability):
        super().__init__(model)
        self.join_order = join_order

        self.initial_cash_usd = float(initial_cash_usd)
        self.cash_balance_usd = float(initial_cash_usd)

        self.initial_eth = float(initial_eth)
        self.total_eth = float(initial_eth)
        self.available_eth = float(initial_eth)
        self.locked_eth = 0.0

        self.received_hui = False
        self.received_round = None
        self.status = ACTIVE

        self.future_obligation_usd = 0.0
        self.collateral_usd = 0.0
        self.collateral_ratio = None
        self.collateral_eth_base = 0.0
        self.collateral_buffer = 0.0
        self.lock_price = None
        self.margin_ratio = None
        self.bid_rate = 0.0
        self.bid_amount = 0.0
        self.winner_payout_usd = 0.0
        self.bid_collateral_usd = 0.0
        self.required_collateral_usd = 0.0
        self.additional_collateral_usd = 0.0
        self.eth_collateral_value_usd = 0.0
        self.margin_call_status = None
        self.risk_tolerance = risk_tolerance
        self.default_probability = default_probability

        self.margin_call_count = 0
        self.margin_call_streak = 0
        self.behavioral_default_count = 0
        self.price_driven_failure_count = 0
        self.collateral_released_eth = 0.0
        self.collateral_liquidated_eth = 0.0
        self.collateral_topup_eth = 0.0
        self.contribution_paid_usd = 0.0
        self.recovery_usd = 0.0
        self.residual_loss_usd = 0.0
        self.liquidity_benefit_usd = 0.0
        self.collusion_group = None

        self.failure_type = None
        self.default_reason = None
        self.defaulted_round = None

    @property
    def eligible_to_bid(self):
        return self.status == ACTIVE and not self.received_hui

    def pay_contribution(self, amount):
        if self.cash_balance_usd + 1e-12 < amount:
            raise RuntimeError(f"member {self.join_order} thiếu cash nhưng vẫn bị yêu cầu đóng tiền")
        self.cash_balance_usd -= amount
        self.contribution_paid_usd += amount

    def _collusion_role(self):
        if self.collusion_group is None:
            return None
        mates = [x.join_order for x in self.model.members
                 if x.collusion_group == self.collusion_group and x.eligible_to_bid]
        return "DESIGNATED" if mates and min(mates) == self.join_order else "YIELD"

    def make_bid(self, t):
        m = self.model
        if m.bid_schedule is not None:
            b = m.bid_schedule.get(t, {}).get(self.join_order, 0.0)
        else:
            role = self._collusion_role()
            if role == "DESIGNATED":            # MODEL ASSUMPTION: nhóm luân phiên thắng
                b = m.bid_max
            elif role == "YIELD":
                b = 0.0
            else:
                b = m.py_rng.uniform(0.0, m.bid_max)
                if m.bid_tick > 0:
                    b = min(m.bid_max, round(b / m.bid_tick) * m.bid_tick)
        validate_bid_rate(b, m.bid_max)
        self.bid_rate = b
        return b
