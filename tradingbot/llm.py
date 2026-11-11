"""Thin wrapper around the Anthropic SDK used by every AI stage.

Two call shapes:
  * ``structured`` - returns a validated Pydantic object (Claude structured outputs).
  * ``research``   - free-text answer, optionally using Claude's server-side web search.

Both opt into the server-side refusal fallback by default, and both treat
``stop_reason == "refusal"`` as a hard stop for that stage.
"""

from __future__ import annotations

import logging
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from .costs import UsageTracker

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class StageError(RuntimeError):
    """Raised when a stage cannot produce a usable answer."""


class LLM(Protocol):
    def structured(self, *, stage: str, system: str, prompt: str, schema: type[T], effort: str = "high") -> T: ...

    def research(self, *, stage: str, system: str, prompt: str, effort: str = "high") -> tuple[str, bool]: ...


class ClaudeLLM:
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        enable_fallback: bool = True,
        web_search: bool = True,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.client = client or anthropic.Anthropic(max_retries=4)
        self.model = model
        self.enable_fallback = enable_fallback
        self.web_search = web_search
        self.usage = UsageTracker(model)

    def _fallback_kwargs(self) -> dict:
        if not self.enable_fallback:
            return {}
        return {"betas": [FALLBACK_BETA], "fallbacks": "default"}

    def structured(self, *, stage: str, system: str, prompt: str, schema: type[T], effort: str = "high") -> T:
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort},
            output_format=schema,
            **self._fallback_kwargs(),
        )
        self.usage.record(stage, response.usage)
        self._check_stop(stage, response)
        parsed = response.parsed_output
        if parsed is None:
            raise StageError(f"[{stage}] model returned no parseable {schema.__name__}")
        return parsed

    def research(self, *, stage: str, system: str, prompt: str, effort: str = "high") -> tuple[str, bool]:
        tools = []
        if self.web_search:
            tools.append({"type": "web_search_20260209", "name": "web_search", "max_uses": 6})

        messages: list[dict] = [{"role": "user", "content": prompt}]
        used_search = False
        text_parts: list[str] = []

        for _ in range(4):  # resume server-tool turns that hit the sampling-loop limit
            kwargs = dict(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=messages,
                output_config={"effort": effort},
                **self._fallback_kwargs(),
            )
            if tools:
                kwargs["tools"] = tools
            response = self.client.beta.messages.create(**kwargs)
            self.usage.record(stage, response.usage)
            self._check_stop(stage, response)

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "server_tool_use":
                    used_search = True

            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue
            break

        text = "".join(text_parts).strip()
        if not text:
            raise StageError(f"[{stage}] research returned no text")
        return text, used_search

    @staticmethod
    def _check_stop(stage: str, response) -> None:
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise StageError(f"[{stage}] model declined the request (category={category})")
        if response.stop_reason == "max_tokens":
            log.warning("[%s] hit max_tokens; output may be truncated", stage)
