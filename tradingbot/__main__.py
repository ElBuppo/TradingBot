"""Command-line entry point.

    python -m tradingbot run                 # Scout the watchlist, then research/verify/trade candidates
    python -m tradingbot run --symbols AAPL,NVDA
    python -m tradingbot analyze TSLA        # skip the Scout, run stages 2-6 on one symbol
    python -m tradingbot account             # show broker account / positions
    python -m tradingbot loop --interval 60  # repeat `run` every N minutes

Add --dry-run to run every stage without sending orders.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from .broker import AlpacaBroker, SimBroker
from .config import Settings
from .data import AlpacaMarketData, SyntheticMarketData
from .journal import Journal
from .llm import ClaudeLLM
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


def print_summary(outcomes) -> None:
    print("\n=== Results ===")
    if not outcomes:
        print("No candidates selected.")
    for o in outcomes:
        print(f"{o.symbol:<6} {o.status:<17} rounds={o.rounds}  {o.detail[:200]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tradingbot", description="Six-stage AI trading desk powered by Claude")
    parser.add_argument("--dry-run", action="store_true", help="run all stages but never send orders")
    parser.add_argument("--live", action="store_true", help="required (with ALLOW_LIVE_TRADING=yes) for live mode")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--symbols", help="comma-separated override for the watchlist")
    p_an = sub.add_parser("analyze")
    p_an.add_argument("symbol")
    sub.add_parser("account")
    p_loop = sub.add_parser("loop")
    p_loop.add_argument("--interval", type=int, default=60, help="minutes between runs")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = Settings.from_env()
    desk = build_desk(settings, args.dry_run, args.live)
    logging.info("mode=%s broker=%s data=%s model=%s dry_run=%s", settings.trading_mode, desk.broker.mode,
                 desk.data.name, settings.model, args.dry_run)

    if args.cmd == "account":
        print(json.dumps(desk.broker.account().to_prompt_dict(), indent=2))
    elif args.cmd == "analyze":
        print_summary([desk.analyze(args.symbol)])
    elif args.cmd == "run":
        symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
        print_summary(desk.run(symbols).outcomes)
    elif args.cmd == "loop":
        while True:
            desk._bars.clear()
            if isinstance(desk.broker, AlpacaBroker) and not desk.broker.is_market_open():
                logging.info("market closed - sleeping")
            else:
                print_summary(desk.run().outcomes)
            time.sleep(args.interval * 60)


if __name__ == "__main__":
    main()
