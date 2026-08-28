"""
LLM client — the single wrapper every model call goes through.

Nothing else in the system calls the Anthropic SDK directly. That is the
whole point: it means tracing, model pinning, cost accounting and structured
output handling are properties of the system rather than things each caller
remembers to do.

Tracing is optional and degrades quietly. If Langfuse keys are absent the
calls still work; you simply lose the spans.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

from agent.db import setting

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=setting("ANTHROPIC_API_KEY"))
    return _client


def model() -> str:
    return setting("MODEL_VERSION", "claude-sonnet-5")


@dataclass
class Call:
    """One model call and what came back.

    Kept as an object rather than a bare string so the caller has the raw
    text, the parsed structure, the token counts and the trace reference
    without a second round trip.
    """

    text: str
    parsed: dict | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    stop_reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelOutputError(ValueError):
    """The model returned something that could not be used.

    Raised rather than guessed at. A check that cannot read the model's
    answer must record inconclusive, not invent a result.
    """


# --------------------------------------------------------------------------
# Tracing — optional
# --------------------------------------------------------------------------

_langfuse = None
_tracing_checked = False


def _tracer():
    """Return a Langfuse client, or None if tracing is not configured."""
    global _langfuse, _tracing_checked
    if _tracing_checked:
        return _langfuse
    _tracing_checked = True

    import os

    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        return None
    try:
        from langfuse import Langfuse

        _langfuse = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception:
        _langfuse = None
    return _langfuse


# --------------------------------------------------------------------------
# JSON handling
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response.

    Models sometimes wrap JSON in code fences or add a sentence either side
    despite instructions. Handle the common cases, and raise rather than
    guess when the text genuinely is not JSON.
    """
    candidate = text.strip()

    m = _FENCE.search(candidate)
    if m:
        candidate = m.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost braces.
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ModelOutputError(f"Could not read JSON from model output: {text[:300]}")


# --------------------------------------------------------------------------
# The call
# --------------------------------------------------------------------------


def complete(
    *,
    system: str,
    user: str,
    max_tokens: int = 1000,
    temperature: float | None = None,
    expect_json: bool = True,
    trace_name: str = "llm_call",
    trace_tags: dict[str, Any] | None = None,
) -> Call:
    """Send one message and return the result.

    temperature is not sent unless explicitly given. Newer models reject the
    parameter outright, and the assessment tasks here do not want sampling
    variance anyway — the same claim against the same policy should produce
    the same finding.
    """
    tags = trace_tags or {}
    tracer = _tracer()
    span = None

    if tracer is not None:
        try:
            span = tracer.trace(
                name=trace_name,
                input={"system": system, "user": user},
                metadata=tags,
                tags=[str(v) for v in tags.values() if isinstance(v, (str, int))],
            )
        except Exception:
            span = None

    kwargs: dict[str, Any] = {
        "model": model(),
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        response = client().messages.create(**kwargs)
    except Exception as exc:
        if span is not None:
            try:
                span.update(output={"error": str(exc)}, level="ERROR")
            except Exception:
                pass
        raise

    text = "".join(block.text for block in response.content if block.type == "text")

    call = Call(
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model=response.model,
        stop_reason=response.stop_reason or "",
        meta=dict(tags),
    )

    if expect_json:
        call.parsed = extract_json(text)

    if span is not None:
        try:
            span.update(
                output=call.parsed if call.parsed is not None else text,
                metadata={
                    **tags,
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                    "model": call.model,
                },
            )
            tracer.flush()
        except Exception:
            pass

    return call
