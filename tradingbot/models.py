"""Pydantic schemas passed between the six AI stages.

Every AI stage (except Research, which writes a free-text dossier with web
citations) returns one of these via Claude structured outputs, so downstream
stages and the code-level guardrails always receive validated data.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------- Stage 1: Market Scout ----------
class Candidate(BaseModel):
    symbol: str = Field(description="Ticker from the provided watchlist")
    priority: int = Field(description="1 (low) to 5 (high) - how worth researching this is right now")
    reason: str = Field(description="Why this symbol stands out in the data provided")


class ScoutReport(BaseModel):
    market_regime: str = Field(description="One-paragraph read of the overall market from the snapshot")
    candidates: list[Candidate]


# ---------- Stage 2: Research Analyst ----------
class ResearchDossier(BaseModel):
    symbol: str
    text: str
    used_web_search: bool = False


# ---------- Stage 3: Technical Analyst ----------
class Bias(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class TechnicalReport(BaseModel):
    symbol: str
    trend: str = Field(description="Primary trend description, e.g. 'uptrend above rising 50-SMA'")
    momentum: str
    support_levels: list[float]
    resistance_levels: list[float]
    bias: Bias
    setup_quality: int = Field(description="0 (no setup) to 10 (textbook setup)")
    notes: str


# ---------- Stage 4: Strategist ----------
class Action(str, Enum):
    BUY = "BUY"  # open / add to a long
    SELL = "SELL"  # close an existing long
    HOLD = "HOLD"  # do nothing


class TradeProposal(BaseModel):
    symbol: str
    action: Action
    entry_price: float = Field(description="Expected entry price (use the current price for market orders)")
    stop_loss: float = Field(description="Price at which the thesis is invalidated; 0 if action is HOLD")
    take_profit: float = Field(description="Profit target price; 0 if action is HOLD")
    position_size_pct: float = Field(description="Requested % of account equity to allocate")
    conviction: int = Field(description="0-10 confidence in the trade")
    time_horizon: str = Field(description="e.g. 'swing: 3-10 trading days'")
    thesis: str
    invalidation: str = Field(description="What would prove the thesis wrong")
    response_to_feedback: str = Field(
        default="", description="How this version addresses prior reviewer critiques (empty on first draft)"
    )


# ---------- Stages 5 & 6: Risk Manager / Verifier ----------
class Decision(str, Enum):
    APPROVE = "APPROVE"
    REVISE = "REVISE"  # send back to the Strategist with feedback
    REJECT = "REJECT"


class RiskReview(BaseModel):
    decision: Decision
    issues: list[str]
    max_position_size_pct: Optional[float] = Field(
        default=None, description="Optional cap on position size (% of equity) if approving smaller"
    )
    feedback_for_strategist: str
    reasoning: str


class FinalVerdict(BaseModel):
    decision: Decision
    confidence: int = Field(description="0-10 confidence in the decision")
    counter_arguments: list[str] = Field(description="Strongest arguments against the trade")
    unsupported_claims: list[str] = Field(
        description="Claims in the thesis that are not backed by the research or technical data"
    )
    feedback_for_strategist: str
    reasoning: str


# ---------- Code-level guardrail result & execution ----------
class GuardrailResult(BaseModel):
    passed: bool
    violations: list[str]
    computed_qty: int = 0
    reward_risk: float = 0.0
    risk_dollars: float = 0.0
    notional: float = 0.0


class ExecutionResult(BaseModel):
    submitted: bool
    mode: str
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    qty: int = 0
    message: str = ""
