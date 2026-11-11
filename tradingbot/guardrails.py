"""Deterministic risk checks. These run in code between AI stages and cannot be
talked around: if they fail, the trade does not execute, whatever the AI says.

Position size is computed HERE from the stop distance - the AI only requests a
percentage, which acts as an upper bound.
"""

from __future__ import annotations

import math

from .broker import AccountInfo
from .config import RiskLimits
from .models import Action, GuardrailResult, TradeProposal


def check(proposal: TradeProposal, account: AccountInfo, last_price: float, limits: RiskLimits,
          size_cap_pct: float | None = None) -> GuardrailResult:
    v: list[str] = []
    p = proposal

    if p.action == Action.HOLD:
        return GuardrailResult(passed=False, violations=["action is HOLD - nothing to execute"])

    if p.action == Action.SELL:
        pos = account.position(p.symbol)
        if not pos:
            if not limits.allow_shorting:
                return GuardrailResult(passed=False, violations=["SELL with no open position and shorting disabled"])
        return GuardrailResult(passed=True, violations=[], computed_qty=int(pos.qty) if pos else 0,
                               notional=round(pos.market_value, 2) if pos else 0.0)

    # ---- BUY checks ----
    entry = p.entry_price
    if entry <= 0 or p.stop_loss <= 0 or p.take_profit <= 0:
        v.append("entry, stop_loss and take_profit must all be positive")
    if not (p.stop_loss < entry < p.take_profit):
        v.append(f"for a long, need stop ({p.stop_loss}) < entry ({entry}) < target ({p.take_profit})")
    if last_price > 0 and abs(entry - last_price) / last_price > 0.03:
        v.append(f"entry {entry} is more than 3% from last price {last_price:.2f} (stale or hallucinated level)")
    if p.conviction < limits.min_conviction:
        v.append(f"conviction {p.conviction} below minimum {limits.min_conviction}")
    if account.trades_today >= limits.max_trades_per_day:
        v.append(f"daily trade limit reached ({account.trades_today}/{limits.max_trades_per_day})")
    if not account.position(p.symbol) and len(account.positions) >= limits.max_open_positions:
        v.append(f"max open positions reached ({limits.max_open_positions})")

    risk_per_share = entry - p.stop_loss
    reward_per_share = p.take_profit - entry
    rr = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0
    if risk_per_share > 0 and rr < limits.min_reward_risk:
        v.append(f"reward:risk {rr:.2f} below minimum {limits.min_reward_risk}")

    # ---- Sizing: the smallest of (requested %, hard cap %, reviewer cap %, risk budget, cash) ----
    qty = 0
    if risk_per_share > 0 and entry > 0 and account.equity > 0:
        pct = min(p.position_size_pct, limits.max_position_pct, size_cap_pct or math.inf)
        existing = account.position(p.symbol)
        existing_value = existing.market_value if existing else 0.0
        by_alloc = (account.equity * pct / 100 - existing_value) / entry
        by_risk = (account.equity * limits.max_risk_per_trade_pct / 100) / risk_per_share
        by_cash = account.buying_power / entry
        qty = max(0, math.floor(min(by_alloc, by_risk, by_cash)))
        if qty == 0:
            v.append("computed position size is 0 shares under current limits")

    return GuardrailResult(
        passed=not v,
        violations=v,
        computed_qty=qty,
        reward_risk=round(rr, 2),
        risk_dollars=round(qty * max(risk_per_share, 0), 2),
        notional=round(qty * entry, 2),
    )
