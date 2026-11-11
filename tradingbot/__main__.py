"""Command-line entry point.

    python -m tradingbot run                 # Scout the watchlist, then research/verify/trade candidates
    python -m tradingbot run --symbols AAPL,NVDA
    python -m tradingbot analyze TSLA        # skip the Scout, run stages 2-6 on one symbol
    python -m tradingbot account             # show broker account / positions
    python -m tradingbot loop --interval 60  # repeat `run` every N minutes, during market hours only
    python -m tradingbot market              # is the market open, and when does it next open/close

run / analyze / loop only start during US market hours; add --anytime to override.
Add --dry-run to run every stage without sending orders.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

from .broker import AlpacaBroker, SimBroker
from .config import Settings
from .data import AlpacaMarketData, SyntheticMarketData
from .journal import Journal
from .llm import ClaudeLLM
from .market_hours import (NY, AlpacaClock, MarketClock, NyseCalendarClock, can_start_run,
                           seconds_until_next_window)
from .pipeline import TradingDesk


def build_desk(settings: Settings, dry_run: bool, live_flag: bool) -> TradingDesk:
    mode = settings.trading_mode
    if mode == "live":
        if not (settings.allow_live_trading and live_flag):
            sys.exit("Refusing LIVE trading: set ALLOW_LIVE_TRADING=yes in .env AND pass --live.")
        print("!!! LIVE TRADING WITH REAL MONEY !!!", file=sys.stderr)

    if mode in {"paper", "live"} and not settings.has_alpaca_keys:
        sys.exit(f"TRADING_MODE={mode} needs ALPACA_API_KEY and ALPACA_SECRET_KEY.")

    if settings.has_alpaca_keys:
        data = AlpacaMarketData(settings.alpaca_api_key, settings.alpaca_secret_key)
    else:
        logging.warning("No Alpaca keys - using SYNTHETIC prices (not real market data).")
        data = SyntheticMarketData()

    if mode == "sim":
        broker = SimBroker(os.path.join(settings.data_dir, "sim_broker.json"), settings.sim_starting_cash)
    else:
        broker = AlpacaBroker(settings.alpaca_api_key, settings.alpaca_secret_key, paper=(mode == "paper"))

    llm = ClaudeLLM(settings.model, settings.enable_refusal_fallback, settings.research_web_search)
    return TradingDesk(settings, llm, data, broker, Journal(settings.log_dir), dry_run=dry_run)


def build_clock(settings: Settings) -> MarketClock:
    if settings.has_alpaca_keys:
        return AlpacaClock(settings.alpaca_api_key, settings.alpaca_secret_key)
    return NyseCalendarClock()


def market_gate(settings: Settings, clock: MarketClock) -> tuple[bool, str]:
    """(allowed, human-readable reason)."""
    now = datetime.now(timezone.utc)
    st = clock.status()
    o, c = settings.market_open_buffer_min, settings.market_close_buffer_min
    if can_start_run(st, now, c, o):
        return True, f"market open, closes {st.next_close.astimezone(NY):%H:%M} ET"
    wait = seconds_until_next_window(st, now, c, o)
    resume = (now + timedelta(seconds=wait)).astimezone(NY)
    why = "market closed" if not st.is_open else "inside the open/close buffer"
    return False, f"{why} - next run window starts {resume:%a %Y-%m-%d %H:%M} ET ({wait / 3600:.1f}h)"


def print_summary(outcomes) -> None:
    print("\n=== Results ===")
    if not outcomes:
        print("No candidates selected.")
    for o in outcomes:
        print(f"{o.symbol:<6} {o.status:<17} rounds={o.rounds}  {o.detail[:200]}")


def print_usage(desk: TradingDesk) -> None:
    usage = getattr(desk.llm, "usage", None)
    if usage and usage.stages:
        print("\n=== Claude usage (this process) ===")
        print(usage.report())
        desk.journal.write("usage", None, {"total_cost_usd": round(usage.total_cost, 4)})


def print_cost_estimate(settings: Settings, candidates: float, model: str | None) -> None:
    from .costs import SCENARIOS, estimate, schedule_table

    model = model or settings.model
    est = estimate(settings, candidates, model)
    toks = est.pop("_prompt_tokens")
    print(f"Model: {model}   watchlist: {len(settings.watchlist)} symbols   "
          f"avg candidates/run: {candidates}   max revision rounds: {settings.max_revision_rounds}")
    print("\nMeasured prompt sizes (input tokens, excl. web search results):")
    for st, v in toks.items():
        print(f"  {st:<13} {v}")
    print("\nCost per candidate by debate outcome (low / mid / high assumptions):")
    for name, prob, _ in SCENARIOS:
        vals = " / ".join(f"${est[c].per_scenario[name]:.2f}" for c in ("low", "mid", "high"))
        print(f"  {name:<30} p={prob:.2f}   {vals}")
    print("\n" + "                      low      mid     high")
    print(f"  Scout call      {est['low'].scout_cost:>8.3f} {est['mid'].scout_cost:>8.3f} {est['high'].scout_cost:>8.3f}")
    print(f"  Per candidate   {est['low'].per_candidate:>8.3f} {est['mid'].per_candidate:>8.3f} "
          f"{est['high'].per_candidate:>8.3f}")
    print(f"  Per full run    {est['low'].per_run:>8.3f} {est['mid'].per_run:>8.3f} {est['high'].per_run:>8.3f}")
    print(f"\n{'Schedule':<34}{'runs/day':>9}{'$/day (mid)':>13}{'$/month low':>13}{'mid':>10}{'high':>10}")
    for (label, rpd, day, month), lo, hi in zip(schedule_table(est["mid"].per_run),
                                                 schedule_table(est["low"].per_run),
                                                 schedule_table(est["high"].per_run)):
        print(f"{label:<34}{rpd:>9.1f}{day:>13,.0f}{lo[3]:>13,.0f}{month:>10,.0f}{hi[3]:>10,.0f}")
    print("\nInput sizes are measured from the real prompts; output/thinking tokens and web-search\n"
          "result sizes are assumptions. Run `--dry-run run` with a real key for measured costs.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tradingbot", description="Six-stage AI trading desk powered by Claude")
    parser.add_argument("--dry-run", action="store_true", help="run all stages but never send orders")
    parser.add_argument("--live", action="store_true", help="required (with ALLOW_LIVE_TRADING=yes) for live mode")
    parser.add_argument("--anytime", action="store_true", help="ignore market hours (testing only)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--symbols", help="comma-separated override for the watchlist")
    p_an = sub.add_parser("analyze")
    p_an.add_argument("symbol")
    sub.add_parser("account")
    sub.add_parser("market", help="show market open/closed status")
    p_cost = sub.add_parser("estimate-cost", help="offline cost projection (no API calls)")
    p_cost.add_argument("--candidates", type=float, default=2.5, help="average candidates the Scout picks per run")
    p_cost.add_argument("--model", help="price a different model, e.g. claude-sonnet-5-5")
    p_loop = sub.add_parser("loop")
    p_loop.add_argument("--interval", type=int, default=60, help="minutes between runs")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = Settings.from_env()
    if args.cmd == "estimate-cost":
        print_cost_estimate(settings, args.candidates, args.model)
        return
    clock = build_clock(settings)
    if args.cmd == "market":
        st = clock.status()
        allowed, reason = market_gate(settings, clock)
        print(f"clock={clock.name} open={st.is_open} next_open={st.next_open.astimezone(NY):%a %Y-%m-%d %H:%M} ET "
              f"next_close={st.next_close.astimezone(NY):%a %Y-%m-%d %H:%M} ET")
        print(("runs allowed: " if allowed else "runs blocked: ") + reason)
        return
    if args.cmd in {"run", "analyze"} and not args.anytime:
        allowed, reason = market_gate(settings, clock)
        if not allowed:
            print(f"Not running: {reason}. Use --anytime to override.")
            return
    desk = build_desk(settings, args.dry_run, args.live)
    logging.info("mode=%s broker=%s data=%s model=%s dry_run=%s", settings.trading_mode, desk.broker.mode,
                 desk.data.name, settings.model, args.dry_run)

    if args.cmd == "account":
        print(json.dumps(desk.broker.account().to_prompt_dict(), indent=2))
    elif args.cmd == "analyze":
        print_summary([desk.analyze(args.symbol)])
        print_usage(desk)
    elif args.cmd == "run":
        symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
        print_summary(desk.run(symbols).outcomes)
        print_usage(desk)
    elif args.cmd == "loop":
        while True:
            allowed, reason = (True, "--anytime") if args.anytime else market_gate(settings, clock)
            if not allowed:
                now = datetime.now(timezone.utc)
                wait = seconds_until_next_window(clock.status(), now, settings.market_close_buffer_min,
                                                 settings.market_open_buffer_min)
                logging.info("%s - sleeping", reason)
                time.sleep(min(wait, 6 * 3600))  # re-check at least every 6h
                continue
            logging.info("starting run (%s)", reason)
            desk._bars.clear()
            print_summary(desk.run().outcomes)
            print_usage(desk)
            time.sleep(args.interval * 60)


if __name__ == "__main__":
    main()
