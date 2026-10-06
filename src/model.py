"""HuiChainModel (Mesa 3.x) for the Basic HuiChain daily simulation."""
from __future__ import annotations

import random

from mesa import Model
from mesa.datacollection import DataCollector

from src import mechanisms as mech
from src.agents import ACTIVE, COMPLETED, DEFAULTED, RECEIVED, MemberAgent

TOL = 1e-6
DECISIONS = ("CONTINUE_SUB_HUI", "TERMINATE_AND_REFUND")


class HuiChainModel(Model):
    def __init__(self, *, N, contribution_usd, r, bid_max, max_ratio, initial_eth,
                 initial_eth_source, cash_multiplier, default_probability,
                 prices, dates, path_id="", seed=42, bid_schedule=None, verbose=False,
                 z=0.0, sigma_eth=0.0, haircut=0.0, maintenance_threshold=1.15,
                 insufficient_eth_policy="FILTER", increment=0.10, bid_tick=0.0,
                 release_policy="PROGRESSIVE", margin_check_interval_days=7,
                 cycle_days=30,
                 post_default_decision="TERMINATE_AND_REFUND",
                 margin_call_grace_periods=0, margin_topup_policy="AUTO",
                 crash_enabled=True, crash_threshold=0.20,
                 collusion_probability=0.0, collusion_suspicion_threshold=0.05,
                 eth_purchase_fraction=0.5, forced_defaults=None):
        super().__init__()
        if initial_eth_source not in ("PRE_OWNED", "BUY_WITH_CASH"):
            raise ValueError("initial_eth_source phải là PRE_OWNED hoặc BUY_WITH_CASH")
        if not (0.0 <= eth_purchase_fraction <= 1.0):
            raise ValueError("eth_purchase_fraction phải trong [0, 1]")
        if insufficient_eth_policy not in ("FILTER", "ERROR"):
            raise ValueError("insufficient_eth_policy phải là FILTER hoặc ERROR")
        if release_policy not in ("PROGRESSIVE", "AT_END"):
            raise ValueError("release_policy phải là PROGRESSIVE hoặc AT_END")
        if post_default_decision not in DECISIONS:
            raise ValueError(f"post_default_decision phải thuộc {DECISIONS}")
        if margin_topup_policy not in ("AUTO", "NONE"):
            raise ValueError("margin_topup_policy phải là AUTO hoặc NONE")
        if margin_call_grace_periods < 0:
            raise ValueError("margin_call_grace_periods phải >= 0")
        if cycle_days <= 0:
            raise ValueError("cycle_days phải > 0")
        if len(prices) != len(dates) or len(prices) < N:
            raise ValueError("prices và dates phải cùng độ dài và có ít nhất N giá")
        if not (0.0 <= haircut < 1.0):
            raise ValueError("haircut phải trong [0, 1)")

        self.N, self.C, self.r = N, float(contribution_usd), r
        self.cycle_days = int(cycle_days)
        self.daily_mode = len(prices) >= N * self.cycle_days
        self.current_day = 0
        self._daily_price = None
        self._daily_date = None
        self.bid_max, self.max_ratio, self.increment = bid_max, max_ratio, increment
        self.bid_tick, self.release_policy = bid_tick, release_policy
        if margin_check_interval_days <= 0:
            raise ValueError("margin_check_interval_days phải > 0")
        self.margin_check_interval_days = int(margin_check_interval_days)
        self.z, self.sigma_eth = z, sigma_eth
        self.haircut, self.maintenance_threshold = haircut, maintenance_threshold
        self.insufficient_eth_policy = insufficient_eth_policy
        self.post_default_decision = post_default_decision
        self.grace = int(margin_call_grace_periods)
        self.margin_topup_policy = margin_topup_policy
        self.crash_enabled, self.crash_threshold = crash_enabled, crash_threshold
        self.collusion_suspicion_threshold = collusion_suspicion_threshold
        self.forced_defaults = {int(k): list(v) for k, v in (forced_defaults or {}).items()}
        self.prices, self.dates, self.path_id = list(prices), list(dates), path_id
        self.seed, self.bid_schedule, self.verbose = seed, bid_schedule, verbose
        self.py_rng = random.Random(seed)                         # bid + risk_tolerance
        self.beh_rng = random.Random(f"{seed}-behavior")          # default ngẫu nhiên
        col_rng = random.Random(f"{seed}-collusion")              # nhóm collusion

        initial_cash = N * self.C * cash_multiplier               # MODEL ASSUMPTION
        self.members = [
            MemberAgent(self, j, initial_cash, (initial_eth if initial_eth_source == "PRE_OWNED" else 0.0),
                        self.py_rng.uniform(1.0, max_ratio), default_probability)
            for j in range(1, N + 1)
        ]
        for a in self.members:                                    # luôn rút N số
            if col_rng.random() < collusion_probability:
                a.collusion_group = "G1"
        if sum(1 for a in self.members if a.collusion_group) < 2:
            for a in self.members:
                a.collusion_group = None

        self.total_initial_cash = initial_cash * N
        self.eth_purchase_total_usd = 0.0
        self.eth_purchase_total_eth = 0.0
        self.eth_purchase_fraction = eth_purchase_fraction
        if initial_eth_source == "BUY_WITH_CASH":               # MODEL ASSUMPTION: mua tai prices[0]
            p0 = float(prices[0])
            for a in self.members:
                spend = a.cash_balance_usd * eth_purchase_fraction
                bought = spend / p0
                a.cash_balance_usd -= spend
                a.total_eth += bought
                a.available_eth += bought
                a.initial_eth = a.total_eth
                self.eth_purchase_total_usd += spend
                self.eth_purchase_total_eth += bought
            self.total_initial_cash -= self.eth_purchase_total_usd   # USD roi khoi he thong
        self.system_fund_usd = 0.0
        self.bid_collateral_total_usd = 0.0
        self.liquidation_proceeds_usd = 0.0

        self.end_round = N
        self.current_round = 0
        self.current_eth_usd = None
        self.current_date = None
        self.hui_status = "ACTIVE"
        self.hui_broken = False
        self.sub_hui_created = False
        self.default_branch = None
        self.failure_type = None
        self.broken_round = None
        self.first_failure_round = None
        self.failure_events = []
        self.round_failures = {}
        self.cascade_count = 0
        self.max_drawdown = 0.0
        self.crash_count = 0
        self.collusion_result = None
        self._mkt = {}
        self.round_log = []
        self.daily_log = []
        self.event_log = []
        self._daily_margin_calls = []
        self._daily_topups = {}
        self._daily_margin_ratios = {}
        self.running = True
        for agent in self.members:
            self._log_event("JOIN", agent.join_order, 0.0, "Thành viên tham gia dây hụi",
                            round_number=0, date=self.dates[0], price=self.prices[0])

        self.datacollector = DataCollector(
            model_reporters={
                "round": lambda m: m.current_round,
                "eth_usd": lambda m: m.current_eth_usd,
                "hui_status": lambda m: m.hui_status,
            },
            agent_reporters={
                "status": "status",
                "cash_usd": "cash_balance_usd",
                "locked_eth": "locked_eth",
                "future_obligation_usd": "future_obligation_usd",
            },
        )
        self.datacollector.collect(self)

    # ------------------------------------------------------------------
    def step(self):
        if not self.running:
            return
        t = self.current_round + 1
        self.current_round = t
        C = self.C
        price = self._daily_price if self.daily_mode else self.prices[t - 1]
        self.current_eth_usd = price
        self.current_date = self._daily_date if self.daily_mode else self.dates[t - 1]
        self._set_market_fields(t, price)
        n0 = len(self.failure_events)

        # A) Behavioral default (đầu round, trước contribution)
        for a, reason in self._collect_defaulters(t):
            if not self.running:
                break
            if a.status == DEFAULTED:
                continue
            missed = (self.end_round - t + 1) * C
            self._resolve_failure(a, "BEHAVIORAL", reason, missed, t - 1, price, t)
        if not self.running:
            return self._finalize_round(t, self._empty_rec(t, price), n0)

        end = self.end_round

        # B) Lọc người có đủ ETH cho phần collateral còn thiếu sau bid.
        fut = mech.future_obligation(t, end, C)
        coll = mech.collateral_usd(t, end, C, self.r)
        base = mech.collateral_eth_base(coll, price)  # Original line retained for context
        buf = 0.0
        req = base

        candidates = [a for a in self.members if a.eligible_to_bid]
        # Bid collateral reduces the ETH requirement, so eligibility is checked
        # after the winner's bid is known.
        eligible = candidates
        short = []

        # C) Contribution: mọi member ACTIVE/RECEIVED đóng C
        payers = [a for a in self.members if a.status in (ACTIVE, RECEIVED)]
        for a in payers:
            a.pay_contribution(C)
        pot = C * len(payers)  # Original line retained for context
        if abs(pot - mech.total_pot(len(payers), C)) > TOL:
            raise AssertionError("TOTAL_POT != (số người đóng) * C")

        # D) Auction (round cuối: không bid)
        is_last = (t == end)
        if is_last:
            if len(candidates) != 1:
                raise AssertionError("round cuối phải còn đúng 1 member chưa nhận")
            bids = {eligible[0].join_order: 0.0}
        else:
            bids = {a.join_order: a.make_bid(t) for a in eligible}
        for agent_id, bid_rate in bids.items():
            self._log_event("BID", agent_id, mech.discount_amount(bid_rate, pot),
                            "Bid hợp lệ", round_number=t, date=self.current_date, price=price)

        top = max(bids.values())
        tied = [a for a in eligible if bids[a.join_order] >= top - 1e-12]
        ratio, tie_break = self.r, None
        if len(tied) >= 2 and not is_last:
            winner = min(tied, key=lambda a: a.join_order)
            tie_break = {"winner": winner.join_order, "rule": "earliest_join_order"}
        else:
            winner = min(tied, key=lambda a: a.join_order)
        bid = bids[winner.join_order]
        self._log_event("WINNER_SELECTED", winner.join_order, bid,
                "Bid cao nhất; hòa bid ưu tiên join_order sớm hơn",
                round_number=t, date=self.current_date, price=price)

        coll_eff = mech.collateral_usd(t, end, C, ratio)
        base_eff = mech.collateral_eth_base(coll_eff, price)
        bid_amount = mech.discount_amount(bid, pot)
        bid_collateral = mech.bid_collateral(bid_amount)
        additional_usd = mech.additional_collateral(coll_eff, bid_collateral)
        req_eff = mech.collateral_eth_base(additional_usd, price)
        if fut > 0 and req_eff > winner.available_eth:
            short = [winner.join_order]
            if self.insufficient_eth_policy == "ERROR":
                raise RuntimeError(f"round {t}: member {winner.join_order} thiếu ETH (cần {req_eff:.4f})")
            for payer in payers:
                payer.cash_balance_usd += C
                payer.contribution_paid_usd -= C
            self.hui_status = "HUI_BREAK"
            self.hui_broken = True
            self.failure_type = "NO_ELIGIBLE_BIDDER_INSUFFICIENT_ETH"
            self.broken_round = t
            self.running = False
            rec = self._empty_rec(t, price, insufficient_eth_members=short,
                                  collateral_usd=coll_eff, collateral_eth_base=req_eff,
                                  buffer=buf, collateral_eth_actual=req_eff)
            return self._finalize_round(t, rec, n0)

        # E) F3, F4, F5
        discount = bid_amount
        payout = mech.winner_payout(pot, discount)
        winner.cash_balance_usd += payout
        self.system_fund_usd += bid_collateral
        self.bid_collateral_total_usd += bid_collateral

        # F) Winner -> RECEIVED; F6-F9, F14; lock ETH
        winner.status = RECEIVED
        winner.received_hui = True
        winner.received_round = t
        winner.bid_rate = bid
        winner.bid_amount = bid_amount
        winner.winner_payout_usd = payout
        winner.bid_collateral_usd = bid_collateral
        winner.collateral_ratio = ratio
        winner.collateral_usd = coll_eff
        winner.required_collateral_usd = coll_eff
        winner.additional_collateral_usd = additional_usd
        winner.collateral_eth_base = req_eff
        winner.collateral_buffer = buf
        winner.lock_price = price
        winner.liquidity_benefit_usd = mech.liquidity_benefit(payout, C, coll_eff)
        winner.available_eth = max(0.0, winner.available_eth - req_eff)
        winner.locked_eth += req_eff
        self._log_event("BID_LOCKED", winner.join_order, bid_collateral,
                "Bid được giữ lại làm bid collateral", round_number=t,
                date=self.current_date, price=price)
        self._log_event("COLLATERAL_LOCKED", winner.join_order, req_eff,
                "ETH bổ sung được khóa làm collateral", round_number=t,
                date=self.current_date, price=price)
        for a in self.members:
            if a.status == RECEIVED:
                a.future_obligation_usd = fut

        # G) Collateral remains locked until completion, default, or margin top-up.
        released = {}
        # H) F11 + margin call (7-step checks) + top-up / price-driven failure
        margin_ratios, margin_calls, topups, due = {}, [], {}, []
        if not self.daily_mode and t % self.margin_check_interval_days == 0:
            for a in self.members:
                if a.status == RECEIVED and a.future_obligation_usd > 0:
                    total_collateral = mech.total_collateral_value(
                        a.bid_collateral_usd, a.locked_eth, price)
                    required = mech.collateral_usd(t, end, C, a.collateral_ratio)
                    mr = total_collateral / required if required > 0 else float("inf")
                    a.margin_ratio = mr
                    a.eth_collateral_value_usd = mech.eth_collateral_value(a.locked_eth, price)
                    a.margin_call_status = "CALL" if total_collateral < required else "OK"
                    margin_ratios[a.join_order] = mr
                    if total_collateral < required:
                        a.margin_call_count += 1
                        a.margin_call_streak += 1
                        margin_calls.append(a.join_order)
                        if a.margin_call_streak > self.grace:
                            due.append(a)
                    else:
                        a.margin_call_streak = 0
        for a in due:
            if not self.running:
                break
            moved = self._handle_margin_call(a, t, price)
            if moved is not None:
                topups[a.join_order] = moved

        # I) Kết thúc ở round cuối (nếu chưa bị kết thúc bởi failure)
        if is_last and self.running:
            for a in self.members:
                if a.status == RECEIVED:
                    a.status = COMPLETED
            self._release_all_locked()
            self._refund_bid_collateral()
            self.hui_status = "COMPLETED"
            self.running = False
            self._log_event("HUI_COMPLETED", None, 0.0, "Hoàn thành toàn bộ các kỳ",
                            round_number=t, date=self.current_date, price=price)

        rec = {
            "round": t, "date": self.current_date, "eth_usd": price, "is_last": is_last,
            "eligible_bidders": [a.join_order for a in eligible], "bids": bids,
            "insufficient_eth_members": short, "tie_break": tie_break,
            "winner": winner.join_order, "bid_rate": bid, "pot": pot,
            "discount": discount, "payout": payout, "discount_shares": {},
            "bid_collateral": bid_collateral, "additional_collateral_usd": additional_usd,
            "future_obligation_usd": fut, "collateral_ratio": ratio,
            "collateral_usd": coll_eff, "collateral_eth_base": req_eff, "buffer": buf,
            "collateral_eth_actual": req_eff,
            "liquidity_benefit_usd": winner.liquidity_benefit_usd,
            "margin_ratios": margin_ratios, "margin_calls": margin_calls, "topups": topups,
            "released": released, "released_total_eth": sum(released.values()),
            "locked_eth_total": sum(a.locked_eth for a in self.members),
            **self._mkt,
        }
        self._finalize_round(t, rec, n0)

    def run(self):
        if self.daily_mode:
            return self._run_daily()
        while self.running:
            self.step()
        grp = sorted(a.join_order for a in self.members if a.collusion_group)
        if grp:
            self.collusion_result = self.collusion_check(grp, self.collusion_suspicion_threshold)
        return self

    def _run_daily(self):
        """Run daily prices while auctions and contributions remain periodic."""
        for day, (price, date) in enumerate(zip(self.prices, self.dates), start=1):
            if not self.running:
                break
            self.current_day = day
            self._daily_price = float(price)
            self._daily_date = date
            self.current_eth_usd = self._daily_price
            self.current_date = self._daily_date
            self._set_market_fields(day, self._daily_price)

            is_round_day = (day - 1) % self.cycle_days == 0
            if self.current_round < self.end_round and is_round_day:
                self.step()
            if self.running and day % self.margin_check_interval_days == 0:
                self._daily_margin_check(day, self._daily_price)

            self.daily_log.append({
                "day": day,
                "date": self._daily_date,
                "eth_usd": self._daily_price,
                "round": self.current_round,
                "margin_calls": list(self._daily_margin_calls),
                "topups": dict(self._daily_topups),
                "locked_eth_total": sum(a.locked_eth for a in self.members),
                "total_collateral_usd": sum(
                    a.bid_collateral_usd + a.locked_eth * self._daily_price
                    for a in self.members
                ),
                "required_collateral_usd": self._required_collateral_snapshot(),
                "coverage_ratio": self._coverage_snapshot(self._daily_price),
                "hui_status": self.hui_status,
                **self._mkt,
            })
        return self

    def _daily_margin_check(self, day, price):
        calls, topups, due, ratios = [], {}, [], {}
        for a in self.members:
            if a.status != RECEIVED or a.future_obligation_usd <= 0:
                continue
            required = a.collateral_ratio * a.future_obligation_usd
            total = mech.total_collateral_value(a.bid_collateral_usd, a.locked_eth, price)
            ratio = total / required if required > 0 else float("inf")
            a.margin_ratio = ratio
            a.eth_collateral_value_usd = mech.eth_collateral_value(a.locked_eth, price)
            a.margin_call_status = "CALL" if total < required else "OK"
            ratios[a.join_order] = ratio
            if total < required:
                a.margin_call_count += 1
                a.margin_call_streak += 1
                calls.append(a.join_order)
                self._log_event("MARGIN_CALL", a.join_order, required - total,
                                "Tổng collateral thấp hơn required collateral",
                                round_number=self.current_round, date=self.current_date,
                                price=price)
                if a.margin_call_streak > self.grace:
                    due.append(a)
            else:
                a.margin_call_streak = 0
        for a in due:
            moved = self._handle_margin_call(a, self.current_round, price)
            if moved is not None:
                topups[a.join_order] = moved
                self._log_event("ETH_TOPUP", a.join_order, moved,
                                "Bổ sung ETH sau margin call", round_number=self.current_round,
                                date=self.current_date, price=price)
        self._daily_margin_calls = calls
        self._daily_topups = topups
        self._daily_margin_ratios = ratios

    # ------------------------------------------------------------------
    # Market / crash
    def _set_market_fields(self, t, price):
        if self.daily_mode:
            ret = None if self.current_day <= 1 else price / self.prices[self.current_day - 2] - 1.0
            history = self.prices[:self.current_day]
        else:
            ret = None if t == 1 else price / self.prices[t - 2] - 1.0
            history = self.prices[:t]
        dd = 1.0 - price / max(history)
        self.max_drawdown = max(self.max_drawdown, dd)
        crash = bool(self.crash_enabled and ret is not None and ret <= -self.crash_threshold)
        if crash:
            self.crash_count += 1
        self._mkt = {"eth_return": ret, "drawdown": dd, "crash_flag": crash}

    # ------------------------------------------------------------------
    # Default detection
    def _collect_defaulters(self, t):
        forced = set(self.forced_defaults.get(t, ()))
        out = []
        for a in self.members:
            u = self.beh_rng.random()                 # luôn rút -> reproducible, common random numbers
            if a.status not in (ACTIVE, RECEIVED):
                continue
            if a.cash_balance_usd + 1e-12 < self.C:
                out.append((a, "INSUFFICIENT_CASH"))
            elif a.join_order in forced:
                out.append((a, "FORCED"))
            elif a.status == RECEIVED and u < a.default_probability:
                out.append((a, "RANDOM_DEFAULT"))
        return out

    # ------------------------------------------------------------------
    # Margin call
    def _handle_margin_call(self, a, t, price):
        """Trả về ETH đã nạp thêm, hoặc None nếu dẫn tới price-driven failure."""
        fut = a.future_obligation_usd if self.daily_mode else (self.end_round - t) * self.C
        required = (a.collateral_ratio * fut if self.daily_mode
                    else mech.collateral_usd(t, self.end_round, self.C, a.collateral_ratio))
        need_usd = mech.additional_collateral(required, a.bid_collateral_usd)
        target = need_usd / price
        need = max(0.0, target - a.locked_eth)
        if self.margin_topup_policy == "AUTO" and a.available_eth + TOL >= need:
            moved = min(need, a.available_eth)
            a.available_eth -= moved
            a.locked_eth += moved
            a.collateral_topup_eth += moved
            a.margin_call_streak = 0
            a.additional_collateral_usd = need_usd
            a.eth_collateral_value_usd = mech.eth_collateral_value(a.locked_eth, price)
            a.margin_ratio = mech.total_collateral_value(
                a.bid_collateral_usd, a.locked_eth, price) / required if required else float("inf")
            return moved
        self._resolve_failure(a, "PRICE_DRIVEN", "ETH_PRICE_CRASH", fut, t, price, t)
        return None

    # ------------------------------------------------------------------
    # Failure (behavioral default hoặc price-driven) + TH1/TH2
    def _resolve_failure(self, a, kind, reason, missed, completed_rounds, price, t):
        eth_sold = a.locked_eth
        eth_liq = mech.eth_collateral_value(eth_sold, price)
        bid_liq = a.bid_collateral_usd
        recoverable = bid_liq + eth_liq
        a.locked_eth = 0.0
        a.total_eth -= eth_sold
        a.collateral_liquidated_eth += eth_sold
        a.bid_collateral_usd = 0.0
        self.liquidation_proceeds_usd += eth_liq
        self.system_fund_usd += eth_liq

        recovery, residual = mech.recovery_and_residual(missed, recoverable)        # F13
        branch = "TH1" if recoverable < missed else "TH2"
        surplus = max(0.0, recoverable - missed)
        system_loss = residual if branch == "TH1" else 0.0

        if kind == "BEHAVIORAL":
            a.behavioral_default_count += 1
            ftype = ("BEHAVIORAL_DEFAULT_INSUFFICIENT_COLLATERAL" if branch == "TH1"
                     else "BEHAVIORAL_DEFAULT_SURPLUS_COLLATERAL")
        else:
            a.price_driven_failure_count += 1
            ftype = "ETH_PRICE_CRASH"
        a.status = DEFAULTED
        a.defaulted_round = t
        a.default_reason = reason
        a.failure_type = ftype
        a.future_obligation_usd = 0.0
        a.recovery_usd += recovery
        a.residual_loss_usd += residual
        self._log_event("DEFAULT", a.join_order, missed, reason, round_number=t,
                        date=self.current_date, price=price)
        self._log_event("RECOVERY", a.join_order, recovery,
                        "Bid collateral + ETH collateral", round_number=t,
                        date=self.current_date, price=price)
        if residual > 0:
            self._log_event("RESIDUAL_LOSS", a.join_order, residual,
                            "Collateral không đủ", round_number=t,
                            date=self.current_date, price=price)
        if surplus > 0:
            self._log_event("SURPLUS", a.join_order, surplus,
                            "Collateral vượt nghĩa vụ; hoàn trả chủ sở hữu",
                            round_number=t, date=self.current_date, price=price)

        remaining = [x for x in self.members if x.status in (ACTIVE, RECEIVED)]
        ev = {
            "round": t, "member": a.join_order, "kind": kind, "reason": reason,
            "price": price, "eth_liquidated": eth_sold, "branch": branch,
            "bid_collateral_usd": bid_liq, "eth_collateral_value_usd": eth_liq,
            "total_recoverable_usd": recoverable,
            "obligation_missed_usd": missed, "liquidation_value_usd": recoverable,
            "recovery_usd": recovery, "residual_loss_usd": residual,
            "system_loss_usd": system_loss, "surplus_liquidation_usd": surplus,
            "post_default_decision": None, "refund_pool_usd": 0.0, "total_refund_usd": 0.0,
            "system_surplus_usd": 0.0, "surplus_per_member": 0.0, "refunds": {},
            "num_remaining_members": len(remaining), "sub_hui_created": False,
            "hui_broken": False, "failure_type": ftype, "released_after": {},
        }
        if self.first_failure_round is None:
            self.first_failure_round = t
        self.default_branch = branch

        if branch == "TH1":
            # Không tạo sub-hui; hụi vỡ; ghi system_loss. recovery nằm lại system_fund.
            self.hui_broken = True
            ev["hui_broken"] = True
            self._end_hui("HUI_BREAK", t, ftype, ev)
        else:
            decision = self.post_default_decision
            ev["post_default_decision"] = decision
            if surplus > 0:
                a.cash_balance_usd += surplus
                self.system_fund_usd -= surplus
                ev["surplus_returned_to_owner"] = surplus
            if decision == "CONTINUE_SUB_HUI" and remaining:
                unreceived = sum(1 for x in remaining if not x.received_hui)
                self.end_round = completed_rounds + unreceived
                self.sub_hui_created = True
                ev["sub_hui_created"] = True
                if unreceived == 0:                      # không còn ai chờ nhận -> hụi hoàn tất
                    for x in remaining:
                        if x.status == RECEIVED:
                            x.status = COMPLETED
                    self._end_hui("COMPLETED", None, None, ev)
                if self.failure_type is None:
                    self.failure_type = ftype
            else:                                        # TERMINATE_AND_REFUND (hoặc không còn ai)
                pool = 0.0
                paid = {x.join_order: x.contribution_paid_usd for x in remaining}
                refunds, total_refund, sys_surplus = mech.refund_distribution(pool, paid)
                for x in remaining:
                    x.cash_balance_usd += refunds.get(x.join_order, 0.0)
                self.system_fund_usd -= total_refund
                ev.update(refund_pool_usd=pool, total_refund_usd=total_refund,
                          system_surplus_usd=sys_surplus, refunds=refunds)
                self.hui_broken = True
                ev["hui_broken"] = True
                self._end_hui("HUI_BREAK", t, ftype, ev)

        self.failure_events.append(ev)
        if self.verbose:
            self._print_failure(ev)

    def _end_hui(self, status, broken_round, ftype, ev):
        self.hui_status = status
        self.running = False
        if status == "HUI_BREAK":
            self.broken_round = broken_round
            self.failure_type = ftype
            self._log_event("HUI_BREAK", ev.get("member"), ev.get("residual_loss_usd", 0.0),
                            ftype or "Collateral không đủ", round_number=broken_round,
                            date=self.current_date, price=self.current_eth_usd)
        ev["released_after"] = self._release_all_locked()

    def _log_event(self, event, agent_id, amount, reason, *, round_number=None,
                   date=None, price=None):
        self.event_log.append({
            "date": date if date is not None else self.current_date,
            "day": self.current_day,
            "round": round_number if round_number is not None else self.current_round,
            "agent_id": agent_id,
            "event": event,
            "value": float(amount or 0.0),
            "amount": float(amount or 0.0),
            "eth_price": price if price is not None else self.current_eth_usd,
            "reason": reason,
        })

    def _required_collateral_snapshot(self):
        return sum(
            (a.collateral_ratio or 0.0) * a.future_obligation_usd
            for a in self.members if a.status == RECEIVED
        )

    def _coverage_snapshot(self, price):
        required = self._required_collateral_snapshot()
        actual = sum(
            a.bid_collateral_usd + a.locked_eth * price
            for a in self.members if a.status == RECEIVED
        )
        return actual / required if required > 0 else float("inf")

    def _refund_bid_collateral(self):
        for a in self.members:
            if a.bid_collateral_usd > 0:
                a.cash_balance_usd += a.bid_collateral_usd
                self.system_fund_usd -= a.bid_collateral_usd
                a.bid_collateral_usd = 0.0

    def _release_all_locked(self):
        out = {}
        for a in self.members:
            if a.locked_eth > 0:
                out[a.join_order] = a.locked_eth
                a.available_eth += a.locked_eth
                a.collateral_released_eth += a.locked_eth
                a.locked_eth = 0.0
        return out

    # ------------------------------------------------------------------
    def _empty_rec(self, t, price, **kw):
        rec = {
            "round": t, "date": self.current_date, "eth_usd": price, "is_last": False,
            "eligible_bidders": [], "bids": {}, "insufficient_eth_members": [],
            "tie_break": None, "winner": None, "bid_rate": 0.0, "pot": 0.0,
            "discount": 0.0, "payout": 0.0, "discount_shares": {},
            "future_obligation_usd": None, "collateral_ratio": self.r,
            "collateral_usd": 0.0, "collateral_eth_base": 0.0, "buffer": 0.0,
            "collateral_eth_actual": 0.0, "liquidity_benefit_usd": None,
            "margin_ratios": {}, "margin_calls": [], "topups": {}, "released": {},
            "released_total_eth": 0.0,
            "locked_eth_total": sum(a.locked_eth for a in self.members),
            **self._mkt,
        }
        rec.update(kw)
        return rec

    def _finalize_round(self, t, rec, n0):
        evs = self.failure_events[n0:]
        rec["failures"] = [(e["member"], e["kind"], e["branch"]) for e in evs]
        rec["hui_status"] = self.hui_status
        n = len(evs)
        self.round_failures[t] = n
        if n:
            self.cascade_count += (n - 1) + (1 if self.round_failures.get(t - 1, 0) > 0 else 0)
        self.check_invariants()
        self.round_log.append(rec)
        self.datacollector.collect(self)
        if self.verbose:
            self._print_round(rec)

    # ------------------------------------------------------------------
    def collusion_check(self, group, threshold=0.05):
        """F15 (nghi ngờ thống kê, không phải bằng chứng) cho một nhóm join_order."""
        group = set(group)
        n = k = 0
        probs = []
        for rec in self.round_log:
            if rec["winner"] is None or rec.get("is_last"):
                continue
            elig = rec["eligible_bidders"]
            g = len(group & set(elig))
            if g == 0 or g == len(elig):
                continue
            n += 1
            probs.append(g / len(elig))
            k += int(rec["winner"] in group)
        p_bar = sum(probs) / n if n else 0.0
        out = mech.collusion_suspicion(n, p_bar, k, threshold)
        out["group"] = sorted(group)
        return out

    def post_default_summary(self):
        evs = self.failure_events

        def S(key):
            return sum(e[key] for e in evs)

        th2 = [e for e in evs if e["branch"] == "TH2"]
        return {
            "num_failures": len(evs),
            "num_behavioral_defaults": sum(e["kind"] == "BEHAVIORAL" for e in evs),
            "num_price_driven_failures": sum(e["kind"] == "PRICE_DRIVEN" for e in evs),
            "default_branch": ("TH1" if any(e["branch"] == "TH1" for e in evs)
                               else ("TH2" if evs else None)),
            "obligation_missed_usd": S("obligation_missed_usd"),
            "liquidation_value_usd": S("liquidation_value_usd"),
            "recovery_usd": S("recovery_usd"),
            "residual_loss_usd": S("residual_loss_usd"),
            "system_loss_usd": S("system_loss_usd"),
            "surplus_liquidation_usd": S("surplus_liquidation_usd"),
            "post_default_decision": th2[-1]["post_default_decision"] if th2 else None,
            "refund_pool_usd": S("refund_pool_usd"),
            "total_refund_usd": S("total_refund_usd"),
            "system_surplus_usd": S("system_surplus_usd"),
            "num_remaining_members": evs[-1]["num_remaining_members"] if evs else None,
            "sub_hui_created": self.sub_hui_created,
            "hui_broken": self.hui_broken,
            "broken_round": self.broken_round,
            "first_failure_round": self.first_failure_round,
            "cascade_count": self.cascade_count,
        }

    def check_invariants(self):
        for a in self.members:
            if a.cash_balance_usd < -TOL:
                raise AssertionError(f"cash âm: member {a.join_order}")
            if a.available_eth < -TOL or a.locked_eth < -TOL:
                raise AssertionError(f"ETH âm: member {a.join_order}")
            if abs(a.available_eth + a.locked_eth - a.total_eth) > TOL:
                raise AssertionError(f"ETH không bảo toàn: member {a.join_order}")
        if self.system_fund_usd < -TOL:
            raise AssertionError("system_fund âm")
        total = sum(a.cash_balance_usd for a in self.members) + self.system_fund_usd
        expected = self.total_initial_cash + self.liquidation_proceeds_usd
        if abs(total - expected) > TOL:
            raise AssertionError(f"tiền bị tạo/mất: {total} != {expected}")

    # ------------------------------------------------------------------
    def _print_failure(self, e):
        print(f"\n--- FAILURE round {e['round']} | member {e['member']} | {e['kind']} ({e['reason']}) ---")
        print(f"ETH price (liquidation): {e['price']:.2f} | ETH liquidated {e['eth_liquidated']:.4f}")
        print(f"obligation missed      : {e['obligation_missed_usd']:.2f}")
        print(f"liquidation value (F10): {e['liquidation_value_usd']:.2f}")
        print(f"recovery / residual    : {e['recovery_usd']:.2f} / {e['residual_loss_usd']:.2f}")
        print(f"post-default branch    : {e['branch']} | surplus {e['surplus_liquidation_usd']:.2f} "
              f"| system loss {e['system_loss_usd']:.2f}")
        print(f"post-default decision  : {e['post_default_decision']}")
        print(f"refund pool / refunded : {e['refund_pool_usd']:.2f} / {e['total_refund_usd']:.2f} "
              f"| system surplus {e['system_surplus_usd']:.2f}")
        print(f"remaining members      : {e['num_remaining_members']} | surplus/member {e['surplus_per_member']:.2f}")
        print(f"sub-hui created        : {e['sub_hui_created']} | hui broken: {e['hui_broken']}")

    def _print_round(self, rec):
        ret = rec["eth_return"]
        ret_s = "-" if ret is None else f"{ret:+.2%}"
        print(f"\n=== ROUND {rec['round']}/{self.end_round} | {rec['date']} | ETH/USD {rec['eth_usd']:.2f} "
              f"| return {ret_s} | drawdown {rec['drawdown']:.2%} | crash {rec['crash_flag']} ===")
        if rec["winner"] is None:
            print(f"không có auction (không ai đủ ETH hoặc hụi đã dừng). status: {rec['hui_status']}")
            return
        bids = {k: round(v, 4) for k, v in rec["bids"].items()}
        shares = {k: round(v, 2) for k, v in rec["discount_shares"].items()}
        mr = {k: round(v, 3) for k, v in rec["margin_ratios"].items()}
        rel = {k: round(v, 4) for k, v in rec["released"].items()}
        print(f"eligible bidders      : {rec['eligible_bidders']} (thiếu ETH: {rec['insufficient_eth_members']})")
        print(f"bids                  : {bids}")
        tb = rec["tie_break"]
        if tb:
            print(f"tie-break             : {tb.get('rule', 'earliest_join_order')} "
                f"-> winner {tb['winner']}")
        print(f"winner (join_order)   : {rec['winner']}  | winning bid {rec['bid_rate']:.4f}")
        print(f"total pot             : {rec['pot']:.2f}")
        print(f"discount / payout     : {rec['discount']:.2f} / {rec['payout']:.2f}")
        print(f"discount shares       : {shares}")
        print(f"future obligation USD : {rec['future_obligation_usd']:.2f}")
        print(f"collateral ratio      : {rec['collateral_ratio']:.3f} (r = {self.r})")
        print(f"collateral USD (F7)   : {rec['collateral_usd']:.2f}")
        print(f"base ETH (F8)         : {rec['collateral_eth_base']:.4f}")
        print(f"buffer (F9)           : {rec['buffer']:.4f}")
        print(f"actual ETH locked     : {rec['collateral_eth_actual']:.4f}")
        print(f"liquidity benefit F14 : {rec['liquidity_benefit_usd']:.2f}")
        print(f"release F16           : {rel} | total {rec['released_total_eth']:.4f} "
              f"| locked sau round {rec['locked_eth_total']:.4f}")
        print(f"margin ratio (F11)    : {mr} | threshold {self.maintenance_threshold}")
        print(f"margin call / top-up  : {rec['margin_calls']} / {({k: round(v, 4) for k, v in rec['topups'].items()})}")
        print(f"failures              : {rec['failures']}")
        print(f"hui status            : {rec['hui_status']}")
