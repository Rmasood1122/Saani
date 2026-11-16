"""Model access: one neutral message format, several providers.

* AnthropicClient     - Claude via the Messages API
* OpenAICompatClient  - OpenAI and anything that speaks its chat-completions
                        dialect (Ollama, LM Studio, OpenRouter, Groq, Gemini's
                        compatible endpoint, ...)

Only the standard library is used, so there is nothing to install.
"""
from __future__ import annotations

import base64
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Msg:
    role: str                      # user | assistant | tool
    content: str = ""
    tool_calls: list = field(default_factory=list)
    tool_call_id: str = ""
    is_error: bool = False
    images: list = field(default_factory=list)   # file paths
    raw: Any = None                # provider-native assistant content, replayed verbatim


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None


class LLMError(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


_sleep = time.sleep
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def post_json(url: str, headers: dict, payload: dict, timeout: float, max_retries: int = 4) -> dict:
    data = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"content-type": "application/json", **headers})
        delay = min(30.0, 1.6 ** attempt + random.random())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:1000]
            last = LLMError(f"HTTP {e.code} from {url}: {body}")
            last.status = e.code  # type: ignore[attr-defined]
            if e.code not in RETRYABLE:
                raise last
            try:
                delay = max(delay, min(60.0, float(e.headers.get("retry-after", 0))))
            except (TypeError, ValueError):
                pass
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = LLMError(f"network error calling {url}: {e}")
        except ValueError as e:
            raise LLMError(f"invalid JSON from {url}: {e}")
        if attempt < max_retries:
            _sleep(delay)
    raise last or LLMError("request failed")


def _image_b64(path: str) -> tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/png")
    with open(path, "rb") as f:
        return media, base64.b64encode(f.read()).decode("ascii")


def extract_json(text: str) -> dict | None:
    """Best-effort: pull the first JSON object out of free text."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [m.group(1)] if m else []
    start = text.find("{")
    if start >= 0:
        candidates.append(text[start: text.rfind("}") + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None


class LLMClient:
    name = "base"

    def complete(self, *, system: str, messages: list[Msg], tools: list[dict] | None = None,
                 model: str = "", max_tokens: int = 4096, force_tool: str | None = None) -> LLMResponse:
        raise NotImplementedError


# ------------------------------------------------------------------ Anthropic
class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, api_key: str, base_url: str = "", timeout: float = 300, max_retries: int = 4):
        self.api_key = api_key
        self.base = (base_url or "https://api.anthropic.com").rstrip("/")
        self.timeout, self.max_retries = timeout, max_retries

    @staticmethod
    def convert(messages: list[Msg]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m.role == "assistant":
                role = "assistant"
                if isinstance(m.raw, dict) and m.raw.get("provider") == "anthropic":
                    blocks = list(m.raw["content"])
                else:
                    blocks = [{"type": "text", "text": m.content}] if m.content else []
                    blocks += [{"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args}
                               for tc in m.tool_calls]
                if not blocks:
                    blocks = [{"type": "text", "text": "(no content)"}]
            else:
                role = "user"
                imgs = []
                for p in m.images:
                    try:
                        media, data = _image_b64(p)
                        imgs.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
                    except OSError:
                        pass
                if m.role == "tool":
                    block = {"type": "tool_result", "tool_use_id": m.tool_call_id,
                             "content": [{"type": "text", "text": m.content or "(empty)"}] + imgs}
                    if m.is_error:
                        block["is_error"] = True
                    blocks = [block]
                else:
                    blocks = [{"type": "text", "text": m.content or "(empty)"}] + imgs
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})
        return out

    def _payload(self, system, messages, tools, model, max_tokens, force_tool):
        p: dict = {"model": model, "max_tokens": max_tokens, "messages": self.convert(messages)}
        if system:
            p["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if tools:
            p["tools"] = [{"name": t["name"], "description": t.get("description", ""),
                           "input_schema": t.get("input_schema") or {"type": "object", "properties": {}}}
                          for t in tools]
            if force_tool == "none":
                p["tool_choice"] = {"type": "none"}
            elif force_tool:
                p["tool_choice"] = {"type": "tool", "name": force_tool}
        return p

    def complete(self, *, system, messages, tools=None, model="", max_tokens=4096, force_tool=None):
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        url = self.base + "/v1/messages"
        payload = self._payload(system, messages, tools, model, max_tokens, force_tool)
        try:
            data = post_json(url, headers, payload, self.timeout, self.max_retries)
        except LLMError as e:
            # Some models reject a forced tool choice (e.g. when thinking is always on).
            # Fall back to an instruction instead of a constraint.
            if getattr(e, "status", 0) != 400 or not force_tool or "tool_choice" not in payload:
                raise
            payload.pop("tool_choice", None)
            hint = ("Do not call any tool; answer in plain text." if force_tool == "none"
                    else f"You must respond by calling the `{force_tool}` tool exactly once.")
            payload["system"] = (payload.get("system") or []) + [{"type": "text", "text": hint}]
            data = post_json(url, headers, payload, self.timeout, self.max_retries)
        text, calls = [], []
        for b in data.get("content", []) or []:
            if b.get("type") == "text":
                text.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                calls.append(ToolCall(b.get("id", ""), b.get("name", ""), b.get("input") or {}))
        usage = data.get("usage", {}) or {}
        return LLMResponse("".join(text).strip(), calls, data.get("stop_reason", ""),
                           int(usage.get("input_tokens", 0) or 0) + int(usage.get("cache_read_input_tokens", 0) or 0)
                           + int(usage.get("cache_creation_input_tokens", 0) or 0),
                           int(usage.get("output_tokens", 0) or 0),
                           raw={"provider": "anthropic", "content": data.get("content", [])})


# --------------------------------------------------------- OpenAI-compatible
class OpenAICompatClient(LLMClient):
    name = "openai"

    def __init__(self, api_key: str = "", base_url: str = "", timeout: float = 300, max_retries: int = 4):
        self.api_key = api_key
        self.base = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout, self.max_retries = timeout, max_retries

    @staticmethod
    def convert(system: str, messages: list[Msg]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}] if system else []
        pending_images: list[str] = []

        def flush():
            if pending_images:
                parts: list[dict] = [{"type": "text", "text": "Images returned by the tools above:"}]
                for p in pending_images:
                    try:
                        media, data = _image_b64(p)
                        parts.append({"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}})
                    except OSError:
                        pass
                out.append({"role": "user", "content": parts})
                pending_images.clear()

        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id,
                            "content": ("ERROR: " if m.is_error else "") + (m.content or "(empty)")})
                pending_images.extend(m.images)
                continue
            flush()
            if m.role == "assistant":
                d: dict = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    d["tool_calls"] = [{"id": tc.id, "type": "function",
                                        "function": {"name": tc.name, "arguments": json.dumps(tc.args)}}
                                       for tc in m.tool_calls]
                elif not m.content:
                    d["content"] = "(no content)"
                out.append(d)
            else:
                if m.images:
                    parts = [{"type": "text", "text": m.content or "(empty)"}]
                    for p in m.images:
                        try:
                            media, data = _image_b64(p)
                            parts.append({"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}})
                        except OSError:
                            pass
                    out.append({"role": "user", "content": parts})
                else:
                    out.append({"role": "user", "content": m.content or "(empty)"})
        flush()
        return out

    def complete(self, *, system, messages, tools=None, model="", max_tokens=4096, force_tool=None):
        payload: dict = {"model": model, "messages": self.convert(system, messages)}
        payload["max_completion_tokens" if "api.openai.com" in self.base else "max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}} for t in tools]
            if force_tool == "none":
                payload["tool_choice"] = "none"
            elif force_tool:
                payload["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = post_json(self.base + "/chat/completions", headers, payload, self.timeout, self.max_retries)
        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response shape: {str(data)[:300]}")
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
            except ValueError:
                args = {"_raw": raw_args}
            calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args if isinstance(args, dict) else {}))
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        usage = data.get("usage", {}) or {}
        return LLMResponse(content.strip(), calls, choice.get("finish_reason", "") or "",
                           int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0))


# --------------------------------------------------------------- router etc.
class Budget:
    """Per-task ceiling on model calls, plus usage accounting."""

    def __init__(self, max_calls: int = 200):
        self.max_calls = max_calls
        self.calls = self.input_tokens = self.output_tokens = 0
        self._lock = threading.Lock()

    def charge(self) -> None:
        with self._lock:
            if self.calls >= self.max_calls:
                raise BudgetExceeded(f"task exceeded its budget of {self.max_calls} model calls")
            self.calls += 1

    def add(self, resp: LLMResponse) -> None:
        with self._lock:
            self.input_tokens += resp.input_tokens
            self.output_tokens += resp.output_tokens


class TraceRecorder:
    """Appends every model call (after the secret scrub) to a JSONL file, one file per day.

    These traces are the training data for your own model (see the saanimind project):
    what the system prompt was, what the conversation looked like, which tools were
    offered and what the model answered. Off unless `llm.record_traces` is true.
    """

    def __init__(self, folder):
        self.folder = folder
        self._lock = threading.Lock()

    def record(self, *, tier: str, model: str, system: str, messages: list[Msg],
               tools: list[dict] | None, resp: LLMResponse) -> None:
        rec = {
            "ts": time.time(), "tier": tier, "model": model, "system": system,
            "messages": [{"role": m.role, "content": m.content,
                          "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in m.tool_calls],
                          "tool_call_id": m.tool_call_id, "is_error": m.is_error} for m in messages],
            "tools": [{"name": t["name"], "description": t.get("description", ""),
                       "input_schema": t.get("input_schema") or {}} for t in (tools or [])],
            "response": {"text": resp.text, "stop_reason": resp.stop_reason,
                         "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in resp.tool_calls]},
        }
        line = json.dumps(rec, ensure_ascii=False, default=str)
        path = os.path.join(str(self.folder), time.strftime("%Y-%m-%d") + ".jsonl")
        with self._lock:
            os.makedirs(str(self.folder), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


class LLMRouter:
    """Maps a tier (planner / worker / fast) to a client and model, applies the last-line
    secret scrub, and optionally records the exchange for training your own model."""

    def __init__(self, client: LLMClient, models: dict | None = None, max_tokens: int = 16000, redactor=None,
                 tier_clients: dict[str, tuple[LLMClient, str]] | None = None, recorder: TraceRecorder | None = None):
        self.client = client
        self.models = models or {}
        self.max_tokens = max_tokens
        self.redactor = redactor
        self.tier_clients = tier_clients or {}     # tier -> (client, model name) overrides
        self.recorder = recorder

    def model_for(self, tier: str) -> str:
        if tier in self.tier_clients:
            return self.tier_clients[tier][1]
        return self.models.get(tier) or self.models.get("worker") or ""

    def client_for(self, tier: str) -> LLMClient:
        return self.tier_clients[tier][0] if tier in self.tier_clients else self.client

    def complete(self, *, tier: str = "worker", budget: Budget | None = None, system: str,
                 messages: list[Msg], tools: list[dict] | None = None,
                 force_tool: str | None = None) -> LLMResponse:
        if budget:
            budget.charge()
        if self.redactor:
            system = self.redactor.redact(system)
            messages = [replace(m, content=self.redactor.redact(m.content)) if m.content else m for m in messages]
        model = self.model_for(tier)
        resp = self.client_for(tier).complete(system=system, messages=messages, tools=tools,
                                              model=model, max_tokens=self.max_tokens,
                                              force_tool=force_tool)
        if budget:
            budget.add(resp)
        if self.recorder is not None:
            try:
                out = resp if not self.redactor else replace(resp, text=self.redactor.redact(resp.text))
                self.recorder.record(tier=tier, model=model, system=system, messages=messages, tools=tools, resp=out)
            except OSError:
                pass
        return resp

    def structured(self, *, tool: dict, tier: str, budget: Budget | None, system: str, messages: list[Msg]) -> dict | None:
        """Ask for a single forced tool call and return its arguments (JSON-in-text as fallback)."""
        resp = self.complete(tier=tier, budget=budget, system=system, messages=messages,
                             tools=[tool], force_tool=tool["name"])
        for tc in resp.tool_calls:
            if tc.name == tool["name"] and isinstance(tc.args, dict):
                return tc.args
        return extract_json(resp.text)


def make_client(config, vault=None) -> LLMClient:
    provider = config.get("llm.provider", "anthropic")
    timeout = config.get("llm.timeout_s", 300)
    retries = config.get("llm.max_retries", 4)
    key_env = config.get("llm.api_key_env", "")
    key = os.environ.get(key_env, "") if key_env else ""
    if not key and vault is not None and key_env:
        key = vault.get(key_env) or ""
    if vault is not None and getattr(vault, "redactor", None) and key:
        vault.redactor.register(key_env or "LLM_API_KEY", key)
    if provider == "demo":
        from .simllm import demo_client
        return demo_client()
    if provider == "anthropic":
        if not key:
            raise LLMError(f"No API key. Run `saani init`, or set the {key_env} environment variable.")
        return AnthropicClient(key, config.get("llm.base_url", ""), timeout, retries)
    if provider == "openai":
        base = config.get("llm.base_url", "")
        if not key and (not base or "api.openai.com" in base):
            raise LLMError(f"No API key. Run `saani init`, or set the {key_env} environment variable.")
        return OpenAICompatClient(key, base, timeout, retries)
    if provider == "own":
        return own_model_client(config)
    raise LLMError(f"unknown llm.provider '{provider}' (use anthropic, openai, own or demo)")


def own_model_client(config) -> LLMClient:
    """Your own model, served by the saanimind project on this machine (OpenAI chat dialect, no key)."""
    base = config.get("llm.own.base_url", "http://127.0.0.1:8900/v1")
    return OpenAICompatClient("", base, config.get("llm.timeout_s", 300), config.get("llm.max_retries", 4))


def own_model_tiers(config) -> dict[str, tuple[LLMClient, str]]:
    """Tier overrides that send some tiers to your own model while another provider handles the rest."""
    tiers = config.get("llm.own.tiers", []) or []
    if config.get("llm.provider") == "own" or not tiers:
        return {}
    client = own_model_client(config)
    model = config.get("llm.own.model", "saanimind")
    return {t: (client, model) for t in tiers if t in ("planner", "worker", "fast")}
