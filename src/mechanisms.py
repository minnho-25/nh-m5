"""Công thức thuần (pure functions) của HuiChain. Mọi số tiền: USD."""


def total_pot(n, c):
    """F1: TOTAL_POT = N * C."""
    return n * c


def validate_bid_rate(bid_rate, bid_max):
    """F2: bid_rate in [0, BID_MAX]."""
    if not (0.0 <= bid_rate <= bid_max + 1e-12):
        raise ValueError(f"bid_rate {bid_rate} ngoài [0, {bid_max}]")
    return bid_rate


def discount_amount(bid_rate, pot):
    """F3: DISCOUNT_AMOUNT = bid_rate * TOTAL_POT."""
    return bid_rate * pot


def winner_payout(pot, discount):
    """F4: WINNER_PAYOUT = TOTAL_POT - DISCOUNT_AMOUNT."""
    return pot - discount


def distribute_discount(discount, recipient_ids):
    """F5: chia đều cho recipients (assumption: chia đều). Trả về {id: share}."""
    if discount == 0:
        return {i: 0.0 for i in recipient_ids}
    if len(recipient_ids) == 0:
        raise ValueError("discount > 0 nhưng không có người nhận")
    share = discount / len(recipient_ids)
    return {i: share for i in recipient_ids}


def bid_collateral(bid_amount):
    """F5: bid amount is retained as the winner's USD collateral."""
    if bid_amount < 0:
        raise ValueError("bid_amount phải >= 0")
    return bid_amount


def additional_collateral(required_usd, bid_collateral_usd):
    """Required ETH collateral after applying the retained bid collateral."""
    if required_usd < 0 or bid_collateral_usd < 0:
        raise ValueError("collateral phải >= 0")
    return max(0.0, required_usd - bid_collateral_usd)


def eth_collateral_value(locked_eth, price):
    """ETH collateral is marked to the current ETH/USD price without haircut."""
    if locked_eth < 0 or price <= 0:
        raise ValueError("locked_eth phải >= 0 và price phải > 0")
    return locked_eth * price


def total_collateral_value(bid_collateral_usd, locked_eth, price):
    return bid_collateral_usd + eth_collateral_value(locked_eth, price)


def future_obligation(t, n, c):
    """F6: (N - t) * C, t = 1..N. Round N -> 0."""
    if not (1 <= t <= n):
        raise ValueError(f"round t={t} ngoài [1, {n}]")
    return (n - t) * c


def collateral_usd(t, n, c, r):
    """F7: collateral_USD = r * future_obligation(t). Round N -> 0."""
    return r * future_obligation(t, n, c)


# ---------------------------------------------------------------
# Phase 5: F8-F11 (ETH collateral)
# ---------------------------------------------------------------
import math  # noqa: E402


def collateral_eth_base(collateral_usd_value, price):
    """F8: collateral_ETH_base = collateral_USD / P_t (P_t = ETH/USD lúc khóa)."""
    if price <= 0:
        raise ValueError("price phải > 0")
    return collateral_usd_value / price


def safety_buffer(z, sigma_eth, n, t):
    """F9: buffer(t) = z * sigma_ETH * sqrt(N - t)."""
    if not (1 <= t <= n):
        raise ValueError(f"round t={t} ngoài [1, {n}]")
    return z * sigma_eth * math.sqrt(n - t)


def collateral_eth_actual(base_eth, buffer):
    """F9: collateral_ETH_actual = base * (1 + buffer)."""
    return base_eth * (1.0 + buffer)


def liquidation_value_usd(eth_held, p_liquidation, haircut):
    """F10: eth_held * P_liquidation * (1 - haircut). P_liquidation là giá lúc thanh lý."""
    if not (0.0 <= haircut < 1.0):
        raise ValueError("haircut phải trong [0, 1)")
    return eth_held * p_liquidation * (1.0 - haircut)


def margin_ratio(eth_held, price, remaining_obligation_usd):
    """F11: eth_held * P_t / remaining_obligation_USD. Nghĩa vụ = 0 -> inf."""
    if remaining_obligation_usd <= 0:
        return math.inf
    return eth_held * price / remaining_obligation_usd


def is_margin_call(ratio, threshold):
    """MARGIN_CALL khi ratio < threshold. Không đồng nghĩa behavioral default."""
    return ratio < threshold


# ---------------------------------------------------------------
# Phase 6: F12-F16
# ---------------------------------------------------------------
def collateral_tie_break(tied, max_ratio, increment):
    """F12. tied = [(join_order, risk_tolerance), ...] (>= 2 người hòa bid cao nhất).

    willing_max_i = min(MAX_RATIO, risk_tolerance_i); winner = argmax(willing_max),
    hòa tiếp -> join_order nhỏ hơn (không random).
    winning_ratio = min(MAX_RATIO, willing_max_winner, runner_up_willing_max + increment).
    """
    if len(tied) < 2:
        raise ValueError("tie-break cần >= 2 người")
    willing = {j: min(max_ratio, rt) for j, rt in tied}
    order = sorted(willing, key=lambda j: (-willing[j], j))
    winner, runner_up = order[0], order[1]
    ratio = min(max_ratio, willing[winner], willing[runner_up] + increment)
    return {"winner": winner, "runner_up": runner_up,
            "winning_ratio": ratio, "willing_max": willing}


def recovery_and_residual(obligation_missed_usd, liquidation_value_usd_):
    """F13: recovery = min(obligation, liquidation); residual = obligation - recovery >= 0."""
    if obligation_missed_usd < 0 or liquidation_value_usd_ < 0:
        raise ValueError("obligation và liquidation phải >= 0")
    recovery = min(obligation_missed_usd, liquidation_value_usd_)
    return recovery, obligation_missed_usd - recovery


def liquidity_benefit(payout, c, collateral_usd_value):
    """F14: liquidity benefit excludes collateral because it remains the owner's asset."""
    return payout - c


def binomial_sf(n, p, k):
    """P(X >= k), X ~ Binomial(n, p). Không cần scipy."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, sum(math.comb(n, i) * p ** i * (1.0 - p) ** (n - i) for i in range(k, n + 1)))


def collusion_suspicion(n, p_bar, k, threshold):
    """F15: nghi ngờ thống kê. p_value < threshold -> collusion_suspected (KHÔNG phải bằng chứng)."""
    if not (0.0 <= p_bar <= 1.0) or not (0 <= k <= n):
        raise ValueError("tham số F15 không hợp lệ")
    p_value = binomial_sf(n, p_bar, k) if n > 0 else 1.0
    return {"n": n, "p_bar": p_bar, "k": k, "p_value": p_value,
            "suspicion_threshold": threshold, "collusion_suspected": p_value < threshold}


def required_eth_safe(ratio, future_obligation_usd, price, buffer):
    """F16: required_ETH_safe = ratio * future_obligation / P_t * (1 + buffer)."""
    return ratio * future_obligation_usd / price * (1.0 + buffer)


def release_amount(locked_prev, required):
    """F16: release = max(0, locked_prev - required_ETH_safe)."""
    return max(0.0, locked_prev - required)


# ---------------------------------------------------------------
# Phase 7: post-default
# ---------------------------------------------------------------
def classify_post_default(liquidation, missed):
    """TH1 nếu liquidation <= obligation_missed (kể cả bằng nhau); TH2 nếu lớn hơn.

    Trả về (branch, surplus_liquidation_usd). Không double-count:
    liquidation = recovery + surplus.
    """
    if liquidation <= missed + 1e-9:
        return "TH1", 0.0
    return "TH2", liquidation - missed


def refund_distribution(pool, paid):
    """TH2 terminate. paid = {member: contribution_paid_to_date}.

    refund_i = min(pool * paid_i / total, paid_i). Trả về (refunds, total_refund, system_surplus).
    total_refund + system_surplus = pool (không tạo thêm tiền).
    """
    if pool < 0:
        raise ValueError("pool phải >= 0")
    total = sum(paid.values())
    if total <= 0:
        return {}, 0.0, pool
    refunds = {j: min(pool * p / total, p) for j, p in paid.items()}
    total_refund = sum(refunds.values())
    return refunds, total_refund, pool - total_refund
