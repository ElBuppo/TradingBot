"""Cost tracking (real API usage) and an offline cost estimator.

* ``UsageTracker`` records the real ``usage`` block of every Claude response,
  so live/dry runs report what they actually cost.
* ``estimate`` runs the real pipeline with a measuring fake LLM: the prompt
  sizes are MEASURED from the prompts the stages really build; output/thinking
  tokens and web-search result sizes are ASSUMPTIONS (see OUTPUT_TOKENS /
  WEB_SEARCH_*), because those only exist once the real model runs.
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from dataclasses import dataclass, field

# USD per million tokens (Claude API list prices). Update if pricing changes.
PRICING = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.10},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
}
WEB_SEARCH_PER_1000 = 10.00


def _price(model: str) -> dict:
    return PRICING.get(model, PRICING["claude-opus-5-5"])


@dataclass
class StageUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    web_searches: int = 0

    def cost(self, model: str) -> float:
        p = _price(model)
        return (
            self.input_tokens * p["input"]
            + self.output_tokens * p["output"]
            + self.cache_write_tokens * p["cache_write"]
            + self.cache_read_tokens * p["cache_read"]
        ) / 1e6 + self.web_searches * WEB_SEARCH_PER_1000 / 1000


@dataclass
class UsageTracker:
    model: str
    stages: dict[str, StageUsage] = field(default_factory=lambda: defaultdict(StageUsage))

    def record(self, stage: str, usage) -> None:
        if usage is None:
            return
        s = self.stages[stage]
        s.calls += 1
        s.input_tokens += getattr(usage, "input_tokens", 0) or 0
        s.output_tokens += getattr(usage, "output_tokens", 0) or 0
        s.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        s.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        stu = getattr(usage, "server_tool_use", None)
        s.web_searches += (getattr(stu, "web_search_requests", 0) or 0) if stu else 0

    @property
    def total_cost(self) -> float:
        return sum(s.cost(self.model) for s in self.stages.values())

    def report(self) -> str:
        lines = [f"{'stage':<14}{'calls':>6}{'in tok':>10}{'out tok':>10}{'search':>8}{'cost $':>9}"]
        for name in sorted(self.stages):
            s = self.stages[name]
            lines.append(f"{name:<14}{s.calls:>6}{s.input_tokens:>10}{s.output_tokens:>10}"
                         f"{s.web_searches:>8}{s.cost(self.model):>9.3f}")
        lines.append(f"{'TOTAL':<14}{'':>34}{self.total_cost:>9.3f}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Offline estimator
# ----------------------------------------------------------------------------

CHARS_PER_TOKEN = 3.2  # conservative for JSON-heavy prompts on the current tokenizer

# Assumed output tokens per call, INCLUDING adaptive thinking (billed as output).
OUTPUT_TOKENS = {
    "low": {"1_scout": 1800, "2_research": 3500, "3_technical": 1800, "4_strategist": 2800, "5_risk": 2000,
            "6_verifier": 2800},
    "mid": {"1_scout": 2500, "2_research": 5000, "3_technical": 2500, "4_strategist": 4000, "5_risk": 3000,
            "6_verifier": 4000},
    "high": {"1_scout": 4000, "2_research": 8000, "3_technical": 4000, "4_strategist": 6500, "5_risk": 5000,
             "6_verifier": 6500},
}
# Web searches per research call, and input tokens each search's results add.
WEB_SEARCHES = {"low": 2, "mid": 4, "high": 6}
WEB_SEARCH_RESULT_TOKENS = {"low": 5000, "mid": 8000, "high": 12000}
RESEARCH_DOSSIER_CHARS = 6000  # length of the dossier text fed to later stages

# How a candidate's debate typically ends, and its probability. Each path is
# actually executed through the pipeline so prompt growth across rounds is measured.
SCENARIOS = [
    ("HOLD in round 1", 0.40, ["HOLD"]),
    ("Rejected in round 1", 0.25, ["REJECT"]),
    ("Approved in round 1", 0.20, ["APPROVE"]),
    ("Revised once, then approved", 0.10, ["REVISE", "APPROVE"]),
    ("Revised twice, no consensus", 0.05, ["REVISE", "REVISE", "REVISE"]),
]


class _MeasuringLLM:
    """Fake LLM that follows a scripted debate path and records prompt sizes."""

    def __init__(self, path: list[str], symbols: list[str], price: float) -> None:
        self.path = path
        self.symbols = symbols
        self.price = price
        self.round = 0
        self.prompt_chars: dict[str, list[int]] = defaultdict(list)

    def research(self, *, stage, system, prompt, effort="high"):
        self.prompt_chars[stage].append(len(system) + len(prompt))
        return "x" * RESEARCH_DOSSIER_CHARS, True

    def structured(self, *, stage, system, prompt, schema, effort="high"):
        from .models import (Action, Bias, Candidate, Decision, FinalVerdict, RiskReview, ScoutReport,
                             TechnicalReport, TradeProposal)

        self.prompt_chars[stage].append(len(system) + len(prompt))
        p = self.price
        filler = "Realistic-length reasoning text. " * 12
        if schema is ScoutReport:
            return ScoutReport(market_regime=filler, candidates=[
                Candidate(symbol=s, priority=5, reason=filler) for s in self.symbols])
        if schema is TechnicalReport:
            return TechnicalReport(symbol="X", trend=filler, momentum=filler, support_levels=[p * .95, p * .9],
                                   resistance_levels=[p * 1.1, p * 1.2], bias=Bias.LONG, setup_quality=7,
                                   notes=filler)
        if schema is TradeProposal:
            self.round += 1
            step = self.path[min(self.round - 1, len(self.path) - 1)]
            return TradeProposal(symbol="X", action=Action.HOLD if step == "HOLD" else Action.BUY,
                                 entry_price=p, stop_loss=p * .95, take_profit=p * 1.12, position_size_pct=5,
                                 conviction=7, time_horizon="swing", thesis=filler * 2, invalidation=filler,
                                 response_to_feedback=filler if self.round > 1 else "")
        step = self.path[min(self.round - 1, len(self.path) - 1)]
        if schema is RiskReview:
            d = Decision.REJECT if step == "REJECT" else Decision.REVISE if step == "REVISE" else Decision.APPROVE
            return RiskReview(decision=d, issues=[filler], feedback_for_strategist=filler, reasoning=filler)
        if schema is FinalVerdict:
            return FinalVerdict(decision=Decision.APPROVE, confidence=7, counter_arguments=[filler],
                                unsupported_claims=[], feedback_for_strategist=filler, reasoning=filler)
        raise TypeError(schema)


def _tokens(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN)


def _cost_from_prompts(prompt_chars: dict[str, list[int]], model: str, case: str) -> float:
    p = _price(model)
    total = 0.0
    for stage, sizes in prompt_chars.items():
        for chars in sizes:
            inp = _tokens(chars)
            out = OUTPUT_TOKENS[case][stage]
            if stage == "2_research":
                n = WEB_SEARCHES[case]
                inp += n * WEB_SEARCH_RESULT_TOKENS[case]
                total += n * WEB_SEARCH_PER_1000 / 1000
            total += (inp * p["input"] + out * p["output"]) / 1e6
    return total


@dataclass
class Estimate:
    case: str
    scout_cost: float
    per_candidate: float  # probability-weighted
    per_scenario: dict[str, float]
    per_run: float


def estimate(settings, candidates_per_run: float = 2.5, model: str | None = None) -> dict[str, Estimate]:
    from .broker import SimBroker
    from .data import SyntheticMarketData
    from .journal import Journal
    from .pipeline import TradingDesk

    import logging

    model = model or settings.model
    logging.getLogger("tradingbot.pipeline").setLevel(logging.WARNING)
    data = SyntheticMarketData()
    syms = settings.watchlist
    price = data.daily_bars(syms[0])[-1].close
    results: dict[str, Estimate] = {}

    measured: dict[str, dict[str, list[int]]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, _, path in SCENARIOS:
            llm = _MeasuringLLM(path, syms[: settings.max_candidates], price)
            desk = TradingDesk(settings, llm, data, SimBroker(f"{tmp}/{name}.json"), Journal(f"{tmp}/logs"),
                               dry_run=True)
            desk.analyze(syms[0])
            measured[name] = dict(llm.prompt_chars)
        from .indicators import summarize

        scout_llm = _MeasuringLLM(["HOLD"], syms[: settings.max_candidates], price)
        desk = TradingDesk(settings, scout_llm, data, SimBroker(f"{tmp}/scout.json"), Journal(f"{tmp}/logs"),
                           dry_run=True)
        snap = {s: summarize(data.daily_bars(s)) for s in syms}
        desk.scout.run(snap, desk.broker.account().to_prompt_dict(), settings.max_candidates)
        scout_chars = dict(scout_llm.prompt_chars)

    for case in ("low", "mid", "high"):
        scout_cost = _cost_from_prompts(scout_chars, model, case)
        per_scn = {name: _cost_from_prompts(measured[name], model, case) for name, _, _ in SCENARIOS}
        per_cand = sum(prob * per_scn[name] for name, prob, _ in SCENARIOS)
        results[case] = Estimate(case, scout_cost, per_cand, per_scn, scout_cost + candidates_per_run * per_cand)
    prompt_tokens: dict[str, list[int]] = {"1_scout": [_tokens(scout_chars["1_scout"][0])]}
    for name in ("Revised twice, no consensus", "Revised once, then approved"):
        for st, v in measured[name].items():
            prompt_tokens.setdefault(st, [_tokens(c) for c in v])
    results["_prompt_tokens"] = dict(sorted(prompt_tokens.items()))  # type: ignore[assignment]
    return results


def schedule_table(per_run: float) -> list[tuple[str, float, float, float]]:
    """(label, runs/day, $/day, $/month) for common schedules."""
    rows = []
    for interval in (15, 30, 60, 240):
        r247 = 24 * 60 / interval
        rows.append((f"24/7, every {interval} min", r247, r247 * per_run, r247 * per_run * 30))
    for interval in (15, 30, 60):
        rmkt = 390 / interval  # 6.5h regular session
        rows.append((f"market hours, every {interval} min", rmkt, rmkt * per_run, rmkt * per_run * 21))
    rows.append(("once per trading day", 1, per_run, per_run * 21))
    return rows
