"""Offline tests: a scripted fake LLM drives the full six-stage pipeline."""

from __future__ import annotations

from collections import defaultdict

import pytest

from tradingbot import guardrails
from tradingbot.broker import AccountInfo, PositionInfo, SimBroker
from tradingbot.config import RiskLimits, Settings
from tradingbot.data import SyntheticMarketData
from tradingbot.indicators import rsi, summarize
from tradingbot.journal import Journal
from tradingbot.models import (
    Action, Bias, Candidate, Decision, FinalVerdict, RiskReview, ScoutReport, TechnicalReport, TradeProposal,
)
from tradingbot.pipeline import TradingDesk


class FakeLLM:
    """Returns queued responses per stage; the last one repeats."""

    def __init__(self, script: dict[str, list]) -> None:
        self.script = script
        self.calls: dict[str, int] = defaultdict(int)
        self.prompts: dict[str, list[str]] = defaultdict(list)

    def _next(self, stage):
        i = self.calls[stage]
        self.calls[stage] += 1
        seq = self.script[stage]
        return seq[min(i, len(seq) - 1)]

    def structured(self, *, stage, system, prompt, schema, effort="high"):
        self.prompts[stage].append(prompt)
        out = self._next(stage)
        assert isinstance(out, schema)
        return out.model_copy(deep=True)

    def research(self, *, stage, system, prompt, effort="high"):
        self.prompts[stage].append(prompt)
        return "Dossier: solid earnings, no events next 2 weeks.", True


DATA = SyntheticMarketData()
SYM = "AAPL"
PRICE = DATA.daily_bars(SYM)[-1].close


def proposal(**kw) -> TradeProposal:
    base = dict(symbol=SYM, action=Action.BUY, entry_price=PRICE, stop_loss=PRICE * 0.95,
                take_profit=PRICE * 1.12, position_size_pct=8, conviction=7, time_horizon="swing",
                thesis="trend continuation", invalidation="close below support")
    base.update(kw)
    return TradeProposal(**base)


TECH = TechnicalReport(symbol=SYM, trend="up", momentum="positive", support_levels=[PRICE * 0.95],
                       resistance_levels=[PRICE * 1.12], bias=Bias.LONG, setup_quality=7, notes="")
SCOUT = ScoutReport(market_regime="risk-on",
                    candidates=[Candidate(symbol=SYM, priority=5, reason="breakout"),
                                Candidate(symbol="NOTREAL", priority=4, reason="hallucinated")])


def risk(decision=Decision.APPROVE, cap=None) -> RiskReview:
    return RiskReview(decision=decision, issues=[], max_position_size_pct=cap,
                      feedback_for_strategist="tighten stop", reasoning="ok")


def verdict(decision=Decision.APPROVE) -> FinalVerdict:
    return FinalVerdict(decision=decision, confidence=7, counter_arguments=["valuation"], unsupported_claims=[],
                        feedback_for_strategist="cite catalyst", reasoning="survives scrutiny")


def make_desk(tmp_path, script, dry_run=False):
    s = Settings(watchlist=[SYM, "MSFT"], log_dir=str(tmp_path / "logs"), max_revision_rounds=2)
    broker = SimBroker(str(tmp_path / "sim.json"), 100_000)
    llm = FakeLLM(script)
    return TradingDesk(s, llm, DATA, broker, Journal(s.log_dir), dry_run=dry_run), llm, broker


def base_script(**over):
    script = {"1_scout": [SCOUT], "3_technical": [TECH], "4_strategist": [proposal()],
              "5_risk": [risk()], "6_verifier": [verdict()]}
    script.update(over)
    return script


def test_all_stages_approve_executes_on_sim_broker(tmp_path):
    desk, llm, broker = make_desk(tmp_path, base_script())
    summary = desk.run()
    assert [o.symbol for o in summary.outcomes] == [SYM]  # hallucinated symbol filtered out
    out = summary.outcomes[0]
    assert out.status == "EXECUTED", out.detail
    pos = broker.account().position(SYM)
    assert pos and pos.qty > 0
    # 1% risk budget on 100k with 5% stop => <= 200 shares-worth of risk; 8% alloc cap also applies
    assert pos.qty * PRICE <= 8_000 + PRICE
    assert all(llm.calls[s] == 1 for s in ["1_scout", "3_technical", "4_strategist", "5_risk", "6_verifier"])


def test_dry_run_sends_nothing(tmp_path):
    desk, _, broker = make_desk(tmp_path, base_script(), dry_run=True)
    out = desk.analyze(SYM)
    assert out.status == "APPROVED_DRY_RUN"
    assert broker.account().positions == []


def test_risk_revise_feeds_back_to_strategist(tmp_path):
    script = base_script(**{"5_risk": [risk(Decision.REVISE), risk()]})
    desk, llm, _ = make_desk(tmp_path, script)
    out = desk.analyze(SYM)
    assert out.status == "EXECUTED" and out.rounds == 2
    assert "tighten stop" in llm.prompts["4_strategist"][1]


def test_verifier_reject_blocks_trade(tmp_path):
    desk, _, broker = make_desk(tmp_path, base_script(**{"6_verifier": [verdict(Decision.REJECT)]}))
    assert desk.analyze(SYM).status == "REJECTED"
    assert broker.account().positions == []


def test_guardrails_override_ai_approval(tmp_path):
    # Bad reward:risk - AIs both "approve" but code refuses every round.
    bad = proposal(take_profit=PRICE * 1.02)
    desk, llm, broker = make_desk(tmp_path, base_script(**{"4_strategist": [bad]}))
    out = desk.analyze(SYM)
    assert out.status == "NO_CONSENSUS"
    assert llm.calls["6_verifier"] == 0
    assert "reward:risk" in llm.prompts["4_strategist"][1]
    assert broker.account().positions == []


def test_hold_short_circuits(tmp_path):
    desk, llm, _ = make_desk(tmp_path, base_script(**{"4_strategist": [proposal(action=Action.HOLD)]}))
    assert desk.analyze(SYM).status == "HOLD"
    assert llm.calls["5_risk"] == 0


def test_guardrail_sizing_and_side_checks():
    acct = AccountInfo(equity=100_000, cash=100_000, buying_power=100_000, positions=[])
    lim = RiskLimits()
    g = guardrails.check(proposal(entry_price=100, stop_loss=95, take_profit=110, position_size_pct=50),
                         acct, 100, lim)
    assert g.passed
    assert g.computed_qty == 100  # 10% cap => $10k / $100; risk 100*5 = $500 <= $1000
    g = guardrails.check(proposal(entry_price=100, stop_loss=105, take_profit=110), acct, 100, lim)
    assert not g.passed
    g = guardrails.check(proposal(entry_price=100, stop_loss=95, take_profit=110), acct, 100, lim, size_cap_pct=2)
    assert g.computed_qty == 20
    g = guardrails.check(proposal(action=Action.SELL), acct, 100, lim)
    assert not g.passed  # no position, no shorting
    acct.positions.append(PositionInfo(SYM, 10, 90, 100))
    assert guardrails.check(proposal(action=Action.SELL), acct, 100, lim).computed_qty == 10


def test_indicators():
    assert rsi([float(i) for i in range(1, 30)]) == 100.0
    snap = summarize(DATA.daily_bars("MSFT"))
    assert snap["sma200"] is not None and 0 <= snap["rsi14"] <= 100


@pytest.mark.parametrize("mode", ["bogus"])
def test_bad_mode_rejected(monkeypatch, mode):
    monkeypatch.setenv("TRADING_MODE", mode)
    with pytest.raises(ValueError):
        Settings.from_env()
