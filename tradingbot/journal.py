"""Append-only JSONL audit log of every stage's output, one file per day."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


class Journal:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    def write(self, stage: str, symbol: str | None, payload) -> None:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "stage": stage,
            "symbol": symbol,
            "data": payload,
        }
        path = os.path.join(self.log_dir, f"journal-{record['ts'][:10]}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
