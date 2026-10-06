import argparse

from src.model import HuiChainModel
from src.scenarios import load_path
from src.utils import load_config
from src.volatility import load_sigma_eth


def _f(x, spec):
    return "-" if x is None else format(x, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="PATH_0001")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bid-tick", type=float, default=None)
    ap.add_argument("--release-policy", default=None, choices=["PROGRESSIVE", "AT_END"])
    ap.add_argument("--r", type=float, default=None)
    ap.add_argument("--haircut", type=float, default=None)
    ap.add_argument("--decision", default=None, choices=["CONTINUE_SUB_HUI", "TERMINATE_AND_REFUND"])
    ap.add_argument("--grace", type=int, default=None)
    ap.add_argument("--topup-policy", default=None, choices=["AUTO", "NONE"])
    ap.add_argument("--default-prob", type=float, default=None)
    ap.add_argument("--collusion-prob", type=float, default=None)
    ap.add_argument("--force-default", action="append", default=[], help="ROUND:MEMBER (lặp được)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    seed = args.seed if args.seed is not None else cfg["simulation"]["random_seed"]
    prices, dates = load_path(args.path)
    sigma = load_sigma_eth()
    col, auc, beh = cfg["collateral"], cfg["auction"], cfg["behavior"]
    mar, pdf, cr = cfg["margin"], cfg["post_default"], cfg["crash"]

    r = args.r if args.r is not None else col["r"]
    haircut = args.haircut if args.haircut is not None else col["haircut"]
    decision = args.decision or pdf["th2_default_decision"]
    grace = args.grace if args.grace is not None else mar["margin_call_grace_periods"]
    topup = args.topup_policy or mar.get("margin_topup_policy", "AUTO")
    dprob = args.default_prob if args.default_prob is not None else beh["default_probability"]
    cprob = args.collusion_prob if args.collusion_prob is not None else beh["collusion_probability"]
    tick = args.bid_tick if args.bid_tick is not None else auc.get("bid_tick", 0.0)
    rel_pol = args.release_policy or col.get("release_policy", "PROGRESSIVE")
    forced = {}
    for s in args.force_default:
        rr, mm = s.split(":")
        forced.setdefault(int(rr), []).append(int(mm))

    print("=== CONFIGURATION (baseline, không phải optimal) ===")
    for sec in ("pool", "agent_endowment", "collateral", "auction", "behavior",
                "post_default", "margin", "crash"):
        print(f"{sec}: {cfg[sec]}")
    print(f"DÙNG: r={r} haircut={haircut} decision={decision} grace={grace} topup={topup} "
          f"default_prob={dprob} collusion_prob={cprob} bid_tick={tick} release={rel_pol} forced={forced}")
    print(f"sigma_eth (từ dữ liệu thật, monthly): {sigma:.6f}")
    print(f"historical path: {args.path} | {dates[0]} -> {dates[-1]} | seed {seed}")
    print(f"ETH/USD path: {[round(p, 2) for p in prices]}")

    model = HuiChainModel(
        N=cfg["pool"]["N"], contribution_usd=cfg["pool"]["contribution_usd"],
        r=r, bid_max=auc["bid_max"], max_ratio=auc["max_ratio"],
        increment=auc["increment"], bid_tick=tick, release_policy=rel_pol,
        initial_eth=cfg["agent_endowment"]["initial_eth"],
        initial_eth_source=cfg["agent_endowment"]["initial_eth_source"],
        cash_multiplier=cfg["agent_endowment"]["cash_multiplier"],
        default_probability=dprob, z=col["z"], sigma_eth=sigma, haircut=haircut,
        maintenance_threshold=col["maintenance_threshold"],
        insufficient_eth_policy=col.get("insufficient_eth_policy", "FILTER"),
        post_default_decision=decision, margin_call_grace_periods=grace,
        margin_check_interval_days=mar.get("margin_check_interval_days", 7),
        margin_topup_policy=topup, crash_enabled=cr["enabled"], crash_threshold=cr["threshold"],
        collusion_probability=cprob,
        collusion_suspicion_threshold=beh.get("collusion_suspicion_threshold", 0.05),
        forced_defaults=forced,
        prices=prices, dates=dates, path_id=args.path, seed=seed, verbose=args.verbose,
    )
    model.run()

    print("\n=== ROUND TABLE ===")
    print(f"{'rnd':>3} {'date':>10} {'ETH/USD':>9} {'ret':>7} {'win':>3} {'pot':>6} {'payout':>8} "
          f"{'act_ETH':>8} {'release':>8} {'locked':>8} {'mcall':>7} {'fail':>5} {'status':>9}")
    for x in model.round_log:
        win = "-" if x["winner"] is None else x["winner"]
        ret = "-" if x["eth_return"] is None else f"{x['eth_return']:+.1%}"
        print(f"{x['round']:>3} {x['date']:>10} {x['eth_usd']:>9.2f} {ret:>7} {win:>3} "
              f"{x['pot']:>6.0f} {x['payout']:>8.2f} {x['collateral_eth_actual']:>8.3f} "
              f"{x['released_total_eth']:>8.3f} {x['locked_eth_total']:>8.3f} "
              f"{str(x['margin_calls']):>7} {len(x['failures']):>5} {x['hui_status']:>9}")

    print("\n=== FAILURE EVENTS ===")
    if not model.failure_events:
        print("(không có)")
    for e in model.failure_events:
        print(f"round {e['round']} member {e['member']} {e['kind']} ({e['reason']}) -> {e['branch']} | "
              f"missed {e['obligation_missed_usd']:.2f} | liq {e['liquidation_value_usd']:.2f} | "
              f"recovery {e['recovery_usd']:.2f} | surplus {e['surplus_liquidation_usd']:.2f} | "
              f"sys_loss {e['system_loss_usd']:.2f} | decision {e['post_default_decision']} | "
              f"refund {e['total_refund_usd']:.2f} | sys_surplus {e['system_surplus_usd']:.2f}")

    print("\n=== FINAL AGENTS ===")
    for a in model.members:
        print(f"member {a.join_order} | {a.status} | rcv_round {a.received_round} | "
              f"cash {a.cash_balance_usd:,.2f} | total_ETH {a.total_eth:.3f} | locked {a.locked_eth:.3f} | "
              f"liq_ETH {a.collateral_liquidated_eth:.3f} | topup {a.collateral_topup_eth:.3f} | "
              f"mcalls {a.margin_call_count} | beh_def {a.behavioral_default_count} | "
              f"price_fail {a.price_driven_failure_count} | grp {a.collusion_group}")
    print(f"\nHui status: {model.hui_status} | failure_type: {model.failure_type} | "
          f"sub_hui_created: {model.sub_hui_created} | end_round: {model.end_round}")
    print(f"max_drawdown {model.max_drawdown:.2%} | crash_count {model.crash_count} | "
          f"cascade_count {model.cascade_count} | system_fund {model.system_fund_usd:,.2f}")
    print("post_default_summary:", model.post_default_summary())
    if model.collusion_result:
        print("F15 (nghi ngờ thống kê, không phải bằng chứng):", model.collusion_result)
    total = sum(a.cash_balance_usd for a in model.members) + model.system_fund_usd
    print(f"Σcash + system_fund = {total:,.2f} | tiền đầu + thanh lý = "
          f"{model.total_initial_cash + model.liquidation_proceeds_usd:,.2f}")


if __name__ == "__main__":
    main()
