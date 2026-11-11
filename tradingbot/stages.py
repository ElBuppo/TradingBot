"""The six AI stages. Each is a role with its own system prompt and output schema.

  1. Market Scout       - scans the watchlist, picks what is worth researching
  2. Research Analyst   - news / catalysts / fundamentals (web search)
  3. Technical Analyst  - price action and indicators
  4. Strategist         - turns research + technicals into a concrete trade plan
  5. Risk Manager       - reviews the plan against portfolio + hard-limit results
  6. Verifier (CIO)     - adversarial final check; tries to break the thesis

Stages 5 and 6 can send the plan back to stage 4 with written feedback; the
Strategist must respond to it. That back-and-forth is the "debate" loop in
pipeline.py.
"""

from __future__ import annotations

import json
from datetime import date

from .llm import LLM
from .models import (
    FinalVerdict,
    GuardrailResult,
    ResearchDossier,
    RiskReview,
    ScoutReport,
    TechnicalReport,
    TradeProposal,
)

_COMMON = (
    "You are one stage in a six-stage AI trading desk (Scout -> Research -> Technical -> Strategist -> "
    "Risk Manager -> Verifier). Other stages will read and challenge your output, so be specific, "
    "state uncertainty honestly, and never invent numbers that are not in the data you were given. "
    f"Today's date is {date.today().isoformat()}."
)


def _j(obj) -> str:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, indent=2, default=str)


class MarketScout:
    name = "1_scout"
    system = _COMMON + (
        "\n\nROLE: Market Scout. You receive an indicator snapshot for every symbol on the watchlist plus the "
        "current portfolio. Pick the symbols most worth a full research pass right now - strong trends, "
        "breakouts, oversold bounces with confirmation, unusual volume, or open positions that may need to be "
        "closed. Only choose symbols that appear in the snapshot. Fewer, better candidates beat many weak ones; "
        "return an empty list if nothing is interesting."
    )

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, snapshot: dict[str, dict], account: dict, max_candidates: int) -> ScoutReport:
        prompt = (
            f"Watchlist snapshot (daily bars):\n{_j(snapshot)}\n\nPortfolio:\n{_j(account)}\n\n"
            f"Return at most {max_candidates} candidates, highest priority first."
        )
        report = self.llm.structured(stage=self.name, system=self.system, prompt=prompt, schema=ScoutReport,
                                     effort="medium")
        valid = [c for c in report.candidates if c.symbol.upper() in snapshot]
        for c in valid:
            c.symbol = c.symbol.upper()
        report.candidates = sorted(valid, key=lambda c: -c.priority)[:max_candidates]
        return report


class ResearchAnalyst:
    name = "2_research"
    system = _COMMON + (
        "\n\nROLE: Research Analyst. Build a concise research dossier on ONE stock. If you have a web search "
        "tool, use it to find the latest news, earnings dates and results, guidance, analyst actions, sector "
        "moves, macro events and any upcoming catalysts in the next two weeks. Cover: (1) recent news & "
        "catalysts with dates, (2) fundamentals snapshot, (3) sentiment, (4) key risks / upcoming events that "
        "could gap the stock, (5) an overall bull vs bear summary. Cite sources inline. Clearly separate facts "
        "you found from your own inference. If you could not verify something, say so."
    )

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, symbol: str, scout_reason: str) -> ResearchDossier:
        prompt = f"Research {symbol}. The Scout flagged it because: {scout_reason}"
        text, used = self.llm.research(stage=self.name, system=self.system, prompt=prompt, effort="high")
        return ResearchDossier(symbol=symbol, text=text, used_web_search=used)


class TechnicalAnalyst:
    name = "3_technical"
    system = _COMMON + (
        "\n\nROLE: Technical Analyst. Analyse ONLY the price/volume data and indicators provided. Identify the "
        "trend, momentum, nearest meaningful support and resistance levels (derived from the bars given), and "
        "whether there is a tradeable setup. Grade setup_quality honestly - most days most stocks are a 3-5."
    )

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, symbol: str, indicators: dict, recent_bars: list[dict]) -> TechnicalReport:
        prompt = (
            f"Symbol: {symbol}\nIndicators:\n{_j(indicators)}\n\n"
            f"Last {len(recent_bars)} daily bars (oldest first):\n{_j(recent_bars)}"
        )
        report = self.llm.structured(stage=self.name, system=self.system, prompt=prompt, schema=TechnicalReport,
                                     effort="medium")
        report.symbol = symbol
        return report


class Strategist:
    name = "4_strategist"

    def __init__(self, llm: LLM, limits) -> None:
        self.llm = llm
        self.system = _COMMON + (
            "\n\nROLE: Strategist / Portfolio Manager. Combine the research dossier and the technical report into "
            "ONE concrete decision for this symbol: BUY (open or add to a long), SELL (close an existing long), "
            "or HOLD. HOLD is a perfectly good answer when the evidence is mixed. For BUY: entry near the current "
            "price, a stop_loss at a level that genuinely invalidates the thesis (typically beyond support or "
            "~1.5-2x ATR), and a take_profit at a realistic level (e.g. next resistance). "
            f"Hard limits enforced in code: reward:risk >= {limits.min_reward_risk}, conviction >= "
            f"{limits.min_conviction} to trade, max {limits.max_position_pct}% of equity per position, max "
            f"{limits.max_risk_per_trade_pct}% of equity at risk per trade, shorting "
            f"{'allowed' if limits.allow_shorting else 'NOT allowed'}. Final share quantity is computed by code. "
            "When reviewers send feedback, address every point explicitly in response_to_feedback - change the "
            "plan where they are right, and defend it with evidence where they are wrong."
        )

    def run(self, symbol: str, last_price: float, research: ResearchDossier, technical: TechnicalReport,
            account: dict, feedback: list[str]) -> TradeProposal:
        prompt = (
            f"Symbol: {symbol}\nCurrent price: {last_price:.2f}\n\n"
            f"=== Research dossier ===\n{research.text}\n\n"
            f"=== Technical report ===\n{_j(technical)}\n\n"
            f"=== Portfolio ===\n{_j(account)}\n"
        )
        if feedback:
            prompt += "\n=== Reviewer feedback on your previous draft(s) - address all of it ===\n"
            prompt += "\n\n".join(feedback)
        proposal = self.llm.structured(stage=self.name, system=self.system, prompt=prompt, schema=TradeProposal,
                                       effort="high")
        proposal.symbol = symbol
        return proposal


class RiskManager:
    name = "5_risk"

    def __init__(self, llm: LLM, limits) -> None:
        self.llm = llm
        self.system = _COMMON + (
            "\n\nROLE: Risk Manager. You are the desk's brake. Review the Strategist's trade plan for: position "
            "sizing vs. conviction, stop placement vs. volatility (ATR) and support, reward:risk, concentration and "
            "correlation with existing positions, event risk (earnings / macro within the holding period), and "
            "liquidity. You also receive the results of the code-level guardrail checks - any violation there is "
            "an automatic REVISE or REJECT. Decide APPROVE, REVISE (fixable - say exactly what to change in "
            "feedback_for_strategist), or REJECT (the idea itself is unsound). You may cap position size via "
            f"max_position_size_pct. Hard limits: {limits.__dict__}."
        )

    def run(self, proposal: TradeProposal, technical: TechnicalReport, research: ResearchDossier,
            guardrails: GuardrailResult, account: dict) -> RiskReview:
        prompt = (
            f"=== Trade plan ===\n{_j(proposal)}\n\n=== Code guardrail results ===\n{_j(guardrails)}\n\n"
            f"=== Technical report ===\n{_j(technical)}\n\n=== Portfolio ===\n{_j(account)}\n\n"
            f"=== Research dossier ===\n{research.text}"
        )
        return self.llm.structured(stage=self.name, system=self.system, prompt=prompt, schema=RiskReview,
                                   effort="high")


class Verifier:
    name = "6_verifier"
    system = _COMMON + (
        "\n\nROLE: Verifier / Chief Investment Officer - the last line of defence before an order is sent. Act as "
        "a sceptical devil's advocate. Independently cross-check the trade plan against the raw research and "
        "technical reports: flag every claim in the thesis that the evidence does not support (unsupported_claims), "
        "check the price levels are consistent with the technical data, and write the strongest case AGAINST the "
        "trade. Only APPROVE if the trade survives that scrutiny. Use REVISE when a specific fix would make it "
        "acceptable (explain it in feedback_for_strategist) and REJECT otherwise. Capital preservation beats "
        "missing a trade."
    )

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, proposal: TradeProposal, research: ResearchDossier, technical: TechnicalReport,
            risk: RiskReview, guardrails: GuardrailResult, debate_log: list[str]) -> FinalVerdict:
        prompt = (
            f"=== Trade plan (final draft) ===\n{_j(proposal)}\n\n=== Risk Manager review ===\n{_j(risk)}\n\n"
            f"=== Code guardrail results ===\n{_j(guardrails)}\n\n=== Technical report ===\n{_j(technical)}\n\n"
            f"=== Research dossier ===\n{research.text}\n\n"
            f"=== Debate history so far ===\n" + ("\n\n".join(debate_log) or "(first draft)")
        )
        return self.llm.structured(stage=self.name, system=self.system, prompt=prompt, schema=FinalVerdict,
                                   effort="high")
