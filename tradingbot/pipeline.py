"""Orchestrates the six stages and the Strategist <-> reviewers debate loop.

Flow per run:

    Scout ──► for each candidate:
                Research ─┐
                Technical ┴─► Strategist ──► code guardrails ──► Risk Manager ──► Verifier ──► Execute
                               ▲                                     │               │
                               └──────── REVISE + feedback ──────────┴───────────────┘
                                          (up to MAX_REVISION_ROUNDS)

A trade executes only if: the Strategist proposes it, the code guardrails pass,
the Risk Manager APPROVES and the Verifier APPROVES - all on the same draft.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import guardrails
from .broker import Broker, SimBroker
from .config import Settings
from .data import MarketData
from .indicators import summarize
from .journal import Journal
from .llm import LLM, StageError
from .models import Action, Decision, ExecutionResult, ScoutReport
from .stages import MarketScout, ResearchAnalyst, RiskManager, Strategist, TechnicalAnalyst, Verifier

log = logging.getLogger(__name__)


@dataclass
class SymbolOutcome:
    symbol: str
    status: str  # EXECUTED | APPROVED_DRY_RUN | HOLD | REJECTED | NO_CONSENSUS | ERROR | SKIPPED
    detail: str = ""
    rounds: int = 0
    execution: ExecutionResult | None = None


@dataclass
class RunSummary:
    scout: ScoutReport | None = None
    outcomes: list[SymbolOutcome] = field(default_factory=list)


class TradingDesk:
    def __init__(self, settings: Settings, llm: LLM, data: MarketData, broker: Broker, journal: Journal,
                 dry_run: bool = False) -> None:
        self.s = settings
        self.data = data
        self.broker = broker
        self.journal = journal
        self.dry_run = dry_run
        self.scout = MarketScout(llm)
        self.research = ResearchAnalyst(llm)
        self.technical = TechnicalAnalyst(llm)
        self.strategist = Strategist(llm, settings.risk)
        self.risk = RiskManager(llm, settings.risk)
        self.verifier = Verifier(llm)
        self._bars: dict[str, list] = {}

    # ---------------- helpers ----------------
    def _load_bars(self, symbol: str):
        if symbol not in self._bars:
            self._bars[symbol] = self.data.daily_bars(symbol, 250)
            if isinstance(self.broker, SimBroker) and self._bars[symbol]:
                self.broker.mark(symbol, self._bars[symbol][-1].close)
        return self._bars[symbol]

    def _emit(self, stage: str, symbol: str | None, payload, headline: str) -> None:
        self.journal.write(stage, symbol, payload)
        log.info("%-13s %-6s %s", stage, symbol or "", headline)

    # ---------------- entry points ----------------
    def run(self, symbols: list[str] | None = None) -> RunSummary:
        """Full pipeline: Scout picks candidates from the watchlist (or ``symbols``)."""
        universe = symbols or self.s.watchlist
        snapshot = {}
        for sym in universe:
            try:
                bars = self._load_bars(sym)
                if len(bars) >= 30:
                    snapshot[sym] = summarize(bars)
            except Exception as e:  # data errors shouldn't kill the run
                log.warning("data error for %s: %s", sym, e)
        summary = RunSummary()
        if not snapshot:
            log.error("no market data available")
            return summary

        account = self.broker.account()
        scout = self.scout.run(snapshot, account.to_prompt_dict(), self.s.max_candidates)
        summary.scout = scout
        self._emit(self.scout.name, None, scout,
                   f"candidates: {[c.symbol for c in scout.candidates] or 'none'}")
        for c in scout.candidates:
            summary.outcomes.append(self.analyze(c.symbol, c.reason))
        return summary

    def analyze(self, symbol: str, reason: str = "requested directly by the user") -> SymbolOutcome:
        """Stages 2-6 for a single symbol."""
        symbol = symbol.upper()
        try:
            return self._analyze(symbol, reason)
        except StageError as e:
            log.error("%s: %s", symbol, e)
            self.journal.write("error", symbol, {"error": str(e)})
            return SymbolOutcome(symbol, "ERROR", str(e))

    def _analyze(self, symbol: str, reason: str) -> SymbolOutcome:
        bars = self._load_bars(symbol)
        if len(bars) < 30:
            return SymbolOutcome(symbol, "SKIPPED", "not enough price history")
        last_price = bars[-1].close
        indicators = summarize(bars)
        recent = [
            {"date": b.timestamp[:10], "o": round(b.open, 2), "h": round(b.high, 2), "l": round(b.low, 2),
             "c": round(b.close, 2), "v": int(b.volume)}
            for b in bars[-40:]
        ]

        research = self.research.run(symbol, reason)
        self._emit(self.research.name, symbol, research,
                   f"dossier {len(research.text)} chars, web_search={research.used_web_search}")

        technical = self.technical.run(symbol, indicators, recent)
        self._emit(self.technical.name, symbol, technical,
                   f"bias={technical.bias.value} setup={technical.setup_quality}/10")

        feedback: list[str] = []
        debate_log: list[str] = []
        max_rounds = self.s.max_revision_rounds + 1

        for rnd in range(1, max_rounds + 1):
            account = self.broker.account()
            acct = account.to_prompt_dict()

            # ---- Stage 4: Strategist ----
            proposal = self.strategist.run(symbol, last_price, research, technical, acct, feedback)
            self._emit(self.strategist.name, symbol, proposal,
                       f"round {rnd}: {proposal.action.value} conv={proposal.conviction} "
                       f"entry={proposal.entry_price} stop={proposal.stop_loss} tp={proposal.take_profit}")
            debate_log.append(f"[Round {rnd}] Strategist: {proposal.action.value} - {proposal.thesis}\n"
                              f"Response to feedback: {proposal.response_to_feedback or '(n/a)'}")
            if proposal.action == Action.HOLD:
                return SymbolOutcome(symbol, "HOLD", proposal.thesis, rnd)

            # ---- Code guardrails ----
            g = guardrails.check(proposal, account, last_price, self.s.risk)
            self._emit("guardrails", symbol, g, f"passed={g.passed} qty={g.computed_qty} rr={g.reward_risk}"
                       + (f" violations={g.violations}" if g.violations else ""))

            # ---- Stage 5: Risk Manager ----
            risk = self.risk.run(proposal, technical, research, g, acct)
            self._emit(self.risk.name, symbol, risk, f"{risk.decision.value}: {'; '.join(risk.issues)[:160]}")
            debate_log.append(f"[Round {rnd}] Risk Manager: {risk.decision.value} - {risk.feedback_for_strategist}")
            if risk.decision == Decision.REJECT:
                return SymbolOutcome(symbol, "REJECTED", f"Risk Manager: {risk.reasoning}", rnd)

            # Re-run guardrails with any size cap the Risk Manager imposed.
            g = guardrails.check(proposal, account, last_price, self.s.risk, risk.max_position_size_pct)
            if risk.decision == Decision.REVISE or not g.passed:
                msg = f"Risk Manager (round {rnd}): {risk.feedback_for_strategist}"
                if g.violations:
                    msg += "\nCode guardrail violations (must be fixed): " + "; ".join(g.violations)
                feedback.append(msg)
                continue

            # ---- Stage 6: Verifier ----
            verdict = self.verifier.run(proposal, research, technical, risk, g, debate_log)
            self._emit(self.verifier.name, symbol, verdict,
                       f"{verdict.decision.value} conf={verdict.confidence} "
                       f"unsupported={len(verdict.unsupported_claims)}")
            debate_log.append(f"[Round {rnd}] Verifier: {verdict.decision.value} - {verdict.feedback_for_strategist}")
            if verdict.decision == Decision.REJECT:
                return SymbolOutcome(symbol, "REJECTED", f"Verifier: {verdict.reasoning}", rnd)
            if verdict.decision == Decision.REVISE:
                feedback.append(
                    f"Verifier (round {rnd}): {verdict.feedback_for_strategist}\n"
                    f"Counter-arguments: {verdict.counter_arguments}\nUnsupported claims: {verdict.unsupported_claims}"
                )
                continue

            # ---- All six stages agree: execute ----
            return self._execute(proposal, g, last_price, rnd)

        return SymbolOutcome(symbol, "NO_CONSENSUS", f"no approval after {max_rounds} rounds", max_rounds)

    def _execute(self, proposal, g, last_price: float, rnd: int) -> SymbolOutcome:
        sym = proposal.symbol
        if self.dry_run:
            res = ExecutionResult(submitted=False, mode=self.broker.mode, symbol=sym, side=proposal.action.value,
                                  qty=g.computed_qty, message="dry run - order not sent")
            self._emit("execution", sym, res, res.message)
            return SymbolOutcome(sym, "APPROVED_DRY_RUN", res.message, rnd, res)
        if proposal.action == Action.BUY:
            res = self.broker.buy_bracket(sym, g.computed_qty, last_price, proposal.stop_loss, proposal.take_profit)
        else:
            res = self.broker.close(sym, last_price)
        self._emit("execution", sym, res, f"[{res.mode}] {res.message}")
        return SymbolOutcome(sym, "EXECUTED" if res.submitted else "ERROR", res.message, rnd, res)
