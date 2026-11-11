"""Brokers: a local simulator and Alpaca (paper or live)."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from typing import Protocol

from .models import ExecutionResult


@dataclass
class PositionInfo:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buying_power: float
    positions: list[PositionInfo]
    trades_today: int = 0

    def position(self, symbol: str) -> PositionInfo | None:
        return next((p for p in self.positions if p.symbol == symbol), None)

    def to_prompt_dict(self) -> dict:
        return {
            "equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "buying_power": round(self.buying_power, 2),
            "open_positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_entry_price": round(p.avg_entry_price, 2),
                    "current_price": round(p.current_price, 2),
                    "pct_of_equity": round(100 * p.market_value / self.equity, 2) if self.equity else 0,
                }
                for p in self.positions
            ],
            "trades_today": self.trades_today,
        }


class Broker(Protocol):
    mode: str

    def account(self) -> AccountInfo: ...

    def buy_bracket(self, symbol: str, qty: int, price: float, stop: float, target: float) -> ExecutionResult: ...

    def close(self, symbol: str, price: float) -> ExecutionResult: ...


class SimBroker:
    """In-process paper broker persisted to a JSON file. Fills instantly at the given price."""

    mode = "sim"

    def __init__(self, state_path: str, starting_cash: float = 100_000.0) -> None:
        self.state_path = state_path
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
        else:
            state = {"cash": starting_cash, "positions": {}, "orders": []}
        self.state = state

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def account(self) -> AccountInfo:
        positions = [PositionInfo(**p) for p in self.state["positions"].values()]
        equity = self.state["cash"] + sum(p.market_value for p in positions)
        from datetime import date

        today = date.today().isoformat()
        trades_today = sum(1 for o in self.state["orders"] if o["date"] == today)
        return AccountInfo(equity, self.state["cash"], self.state["cash"], positions, trades_today)

    def mark(self, symbol: str, price: float) -> None:
        if symbol in self.state["positions"]:
            self.state["positions"][symbol]["current_price"] = price

    def _record(self, **order) -> str:
        from datetime import date

        order_id = f"sim-{uuid.uuid4().hex[:10]}"
        self.state["orders"].append({"id": order_id, "date": date.today().isoformat(), **order})
        self._save()
        return order_id

    def buy_bracket(self, symbol: str, qty: int, price: float, stop: float, target: float) -> ExecutionResult:
        cost = qty * price
        if cost > self.state["cash"]:
            return ExecutionResult(submitted=False, mode=self.mode, symbol=symbol, message="insufficient cash")
        self.state["cash"] -= cost
        pos = self.state["positions"].get(symbol)
        if pos:
            total = pos["qty"] + qty
            pos["avg_entry_price"] = (pos["avg_entry_price"] * pos["qty"] + cost) / total
            pos["qty"] = total
            pos["current_price"] = price
        else:
            self.state["positions"][symbol] = asdict(PositionInfo(symbol, qty, price, price))
        oid = self._record(symbol=symbol, side="buy", qty=qty, price=price, stop=stop, target=target)
        return ExecutionResult(submitted=True, mode=self.mode, order_id=oid, symbol=symbol, side="buy", qty=qty,
                               message=f"simulated fill {qty} @ {price:.2f} (stop {stop:.2f}, target {target:.2f})")

    def close(self, symbol: str, price: float) -> ExecutionResult:
        pos = self.state["positions"].pop(symbol, None)
        if not pos:
            return ExecutionResult(submitted=False, mode=self.mode, symbol=symbol, message="no position to close")
        self.state["cash"] += pos["qty"] * price
        oid = self._record(symbol=symbol, side="sell", qty=pos["qty"], price=price)
        return ExecutionResult(submitted=True, mode=self.mode, order_id=oid, symbol=symbol, side="sell",
                               qty=int(pos["qty"]), message=f"simulated close @ {price:.2f}")


class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        from alpaca.trading.client import TradingClient

        self.mode = "paper" if paper else "live"
        self._client = TradingClient(api_key, secret_key, paper=paper)

    def account(self) -> AccountInfo:
        from datetime import datetime, timezone

        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        acct = self._client.get_account()
        positions = [
            PositionInfo(p.symbol, float(p.qty), float(p.avg_entry_price), float(p.current_price or 0))
            for p in self._client.get_all_positions()
        ]
        midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        orders = self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=midnight, limit=500))
        return AccountInfo(float(acct.equity), float(acct.cash), float(acct.buying_power), positions, len(orders))

    def buy_bracket(self, symbol: str, qty: int, price: float, stop: float, target: float) -> ExecutionResult:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

        order = self._client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(target, 2)),
                stop_loss=StopLossRequest(stop_price=round(stop, 2)),
            )
        )
        return ExecutionResult(submitted=True, mode=self.mode, order_id=str(order.id), symbol=symbol, side="buy",
                               qty=qty, message=f"bracket order submitted (status {order.status})")

    def close(self, symbol: str, price: float) -> ExecutionResult:
        order = self._client.close_position(symbol)
        return ExecutionResult(submitted=True, mode=self.mode, order_id=str(getattr(order, "id", "")),
                               symbol=symbol, side="sell", message="close position submitted")
