"""Runtime configuration loaded from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass
class RiskLimits:
    """Hard limits enforced in code. No AI stage can override these."""

    max_position_pct: float = 10.0  # max % of equity in a single position
    max_risk_per_trade_pct: float = 1.0  # max % of equity lost if the stop is hit
    max_open_positions: int = 5
    max_trades_per_day: int = 3
    min_reward_risk: float = 1.5
    min_conviction: int = 6
    allow_shorting: bool = False


@dataclass
class Settings:
    model: str = "claude-opus-5-5"
    enable_refusal_fallback: bool = True
    research_web_search: bool = True

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    trading_mode: str = "sim"  # sim | paper | live
    allow_live_trading: bool = False

    watchlist: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "SPY"])
    max_candidates: int = 3
    max_revision_rounds: int = 2
    sim_starting_cash: float = 100_000.0
    # Market-hours gate: no new runs in the first/last N minutes of the regular session.
    market_open_buffer_min: int = 0
    market_close_buffer_min: int = 15

    log_dir: str = "logs"
    data_dir: str = "data"

    risk: RiskLimits = field(default_factory=RiskLimits)

    @property
    def has_alpaca_keys(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        watchlist = [s.strip().upper() for s in os.getenv("WATCHLIST", "").split(",") if s.strip()]
        settings = cls(
            model=os.getenv("CLAUDE_MODEL", "claude-opus-5-5"),
            enable_refusal_fallback=_bool("ENABLE_REFUSAL_FALLBACK", True),
            research_web_search=_bool("RESEARCH_WEB_SEARCH", True),
            alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
            alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            trading_mode=os.getenv("TRADING_MODE", "sim").strip().lower(),
            allow_live_trading=_bool("ALLOW_LIVE_TRADING", False),
            max_candidates=_int("MAX_CANDIDATES", 3),
            max_revision_rounds=_int("MAX_REVISION_ROUNDS", 2),
            sim_starting_cash=_float("SIM_STARTING_CASH", 100_000.0),
            market_open_buffer_min=_int("MARKET_OPEN_BUFFER_MIN", 0),
            market_close_buffer_min=_int("MARKET_CLOSE_BUFFER_MIN", 15),
            log_dir=os.getenv("LOG_DIR", "logs"),
            data_dir=os.getenv("DATA_DIR", "data"),
            risk=RiskLimits(
                max_position_pct=_float("MAX_POSITION_PCT", 10.0),
                max_risk_per_trade_pct=_float("MAX_RISK_PER_TRADE_PCT", 1.0),
                max_open_positions=_int("MAX_OPEN_POSITIONS", 5),
                max_trades_per_day=_int("MAX_TRADES_PER_DAY", 3),
                min_reward_risk=_float("MIN_REWARD_RISK", 1.5),
                min_conviction=_int("MIN_CONVICTION", 6),
            ),
        )
        if watchlist:
            settings.watchlist = watchlist
        if settings.trading_mode not in {"sim", "paper", "live"}:
            raise ValueError(f"TRADING_MODE must be sim, paper or live (got {settings.trading_mode!r})")
        return settings
