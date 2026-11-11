# TradingBot — a six-stage AI trading desk

Six Claude-powered agents research, debate and verify every trade before an order goes out.
Hard risk limits are enforced **in code** between the AI stages, so no amount of AI confidence can bypass them.

> ⚠️ **Not financial advice.** LLMs can be confidently wrong. The bot defaults to a local simulator;
> use Alpaca **paper** trading for a long time before even considering live mode.

## The six stages

| # | Stage | Job | Output |
|---|-------|-----|--------|
| 1 | **Market Scout** | Scans indicator snapshots for the whole watchlist + portfolio, picks what's worth researching | `ScoutReport` |
| 2 | **Research Analyst** | Uses Claude web search for news, earnings, catalysts, sentiment and risks | cited text dossier |
| 3 | **Technical Analyst** | Reads price bars + SMA/RSI/ATR/MACD, finds trend, levels and setup quality | `TechnicalReport` |
| 4 | **Strategist** | Combines 2 + 3 into BUY / SELL / HOLD with entry, stop, target, size, thesis | `TradeProposal` |
| 5 | **Risk Manager** | Checks sizing, stop vs volatility, concentration, event risk, and the code guardrail results | `RiskReview` |
| 6 | **Verifier (CIO)** | Devil's advocate: cross-checks the thesis against the raw evidence, flags unsupported claims | `FinalVerdict` |

### How they talk to each other

```
Scout ─► per candidate:
          Research ─┐
          Technical ┴─► Strategist ─► code guardrails ─► Risk Manager ─► Verifier ─► execute
                          ▲                                   │              │
                          └──── REVISE + written feedback ────┴──────────────┘
                                   (up to MAX_REVISION_ROUNDS)
```

* The Risk Manager and the Verifier can **APPROVE**, **REJECT**, or send the plan back with **REVISE** plus specific feedback.
* The Strategist must answer every critique (`response_to_feedback`) in its next draft, and the Verifier sees the whole debate history.
* A trade executes only when the Strategist proposes it, the **code guardrails pass**, and **both** reviewers approve the same draft.

### Code guardrails (`tradingbot/guardrails.py`)
- stop < entry < target; entry within 3% of the real last price (catches made-up levels)
- reward:risk ≥ `MIN_REWARD_RISK`, conviction ≥ `MIN_CONVICTION`
- max `MAX_POSITION_PCT` % of equity per position, max `MAX_RISK_PER_TRADE_PCT` % of equity lost at the stop
- max open positions, max trades per day, no shorting
- **Share quantity is computed by code** from the stop distance. The AI's requested size only acts as an upper bound.

Every stage's output is appended to `logs/journal-YYYY-MM-DD.jsonl` for auditing.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add ANTHROPIC_API_KEY (+ Alpaca paper keys for real market data)
```

Without Alpaca keys the bot uses **synthetic** prices so you can try the pipeline end to end.
Get free paper keys at <https://app.alpaca.markets> (Paper Trading → API Keys).

## Usage

```bash
python -m tradingbot account                 # balances and positions
python -m tradingbot --dry-run run           # full 6-stage run, no orders sent
python -m tradingbot run --symbols NVDA,AMD  # scout a custom list
python -m tradingbot analyze TSLA            # stages 2-6 on one symbol
python -m tradingbot loop --interval 60      # repeat every 60 minutes (skips when the market is closed)
```

| `TRADING_MODE` | Broker | Notes |
|---|---|---|
| `sim` (default) | local simulator (`data/sim_broker.json`) | instant fills, nothing leaves your machine |
| `paper` | Alpaca paper account | bracket orders with stop-loss + take-profit attached |
| `live` | Alpaca live account | **real money**: needs `ALLOW_LIVE_TRADING=yes` **and** the `--live` flag |

## Claude configuration
- Model: `claude-opus-5-5` (change with `CLAUDE_MODEL`). Effort is `medium` for Scout/Technical and `high` for Research/Strategist/Risk/Verifier.
- Structured outputs (Pydantic schemas) for every stage except Research.
- Server-side refusal fallback is on by default (`ENABLE_REFUSAL_FALLBACK=false` turns it off).
- Web search for Research: `RESEARCH_WEB_SEARCH=false` turns it off. Web search must also be enabled for your org in the Claude Console.
- Cost: one symbol takes about 5 to 7 Claude calls (more if there are revision rounds), and Research with web search is the most expensive call.

## Tests

```bash
pytest -q
```
The tests use a scripted fake LLM, so they cost nothing. They cover the approve, revise, reject and HOLD paths, guardrails overriding AI approval, sizing maths and the indicators.

## Layout
```
tradingbot/
  stages.py      six agent roles and their prompts
  pipeline.py    orchestration + debate loop
  guardrails.py  deterministic risk limits & position sizing
  llm.py         Anthropic SDK wrapper (structured outputs, web search, refusal handling)
  models.py      Pydantic schemas exchanged between stages
  broker.py      SimBroker + AlpacaBroker
  data.py        Alpaca + synthetic market data
  indicators.py  SMA / EMA / RSI / ATR / MACD
  journal.py     JSONL audit log
```
