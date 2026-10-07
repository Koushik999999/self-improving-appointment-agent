"""LLM access for every role (agent, sim, judge, improver) via OpenAI-compatible endpoints.

Configuration is per role, from the environment (.env is loaded, real env vars win):
    {ROLE}_PROVIDER   groq | gemini | openrouter | ollama   (fills base URL + key var)
    {ROLE}_MODEL      model id at that provider
    {ROLE}_BASE_URL / {ROLE}_API_KEY   optional explicit overrides
    {ROLE}_RPM / _TPM / _RPD / _TPD    client-side limits (0 = unknown/unlimited)
    {ROLE}_REASONING_EFFORT            optional (low|medium|high) for reasoning models
IMPROVER falls back to JUDGE settings when unset.

What a call goes through, in order:
  1. Disk cache, keyed by (model, messages, tools, params, sample index). The sample index
     keeps N runs of a scenario independent while making reruns reproducible and free.
  2. Daily budget (RPD/TPD, tracked in a per-day ledger): raises QuotaExhausted instead of
     burning retries, so eval checkpoints can resume tomorrow.
  3. Per-minute limiter (RPM/TPM), shared by every role on the same provider+model.
  4. The request, retried with exponential backoff on 429 / 5xx / timeouts. Other 4xx errors
     are raised: they fail the same way every time.
Assistant messages are kept verbatim (message.model_dump), never rebuilt: Gemini rejects
tool-call history without its thought signatures (400), and Groq accepts either.
"""
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import openai
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
ROLES = ("agent", "sim", "judge", "improver")

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


# ---------------------------------------------------------------- configuration

def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def env_int(name: str, default: int) -> int:
    load_dotenv()
    return int(os.environ.get(name) or default)


@dataclass(frozen=True)
class RoleConfig:
    role: str
    provider: str
    base_url: str
    api_key: str = field(repr=False)
    model: str
    rpm: int
    tpm: int
    rpd: int
    tpd: int
    reasoning_effort: str = ""

    @property
    def bucket(self) -> str:
        """Quota identity: providers count per (endpoint, model), not per role."""
        return f"{self.model}@{self.base_url}"

    def problems(self) -> list[str]:
        p = self.role.upper()
        return [msg for ok, msg in [
            (self.base_url, f"{p}_PROVIDER (or {p}_BASE_URL) is not set"),
            (self.api_key, f"API key for {p} is missing (set the provider key or {p}_API_KEY)"),
            (self.model, f"{p}_MODEL is not set"),
        ] if not ok]


def role_config(role: str) -> RoleConfig:
    load_dotenv()
    p = role.upper()

    def get(key, default=""):
        value = os.environ.get(f"{p}_{key}")
        if not value and p == "IMPROVER":
            value = os.environ.get(f"JUDGE_{key}")
        return value or default

    provider = get("PROVIDER").lower()
    url, key_var = PROVIDERS.get(provider, ("", None))
    api_key = get("API_KEY") or (os.environ.get(key_var, "") if key_var else ("ollama" if provider == "ollama" else ""))
    return RoleConfig(role=role, provider=provider or "custom", base_url=get("BASE_URL") or url,
                      api_key=api_key, model=get("MODEL"),
                      rpm=int(get("RPM", 0)), tpm=int(get("TPM", 0)),
                      rpd=int(get("RPD", 0)), tpd=int(get("TPD", 0)),
                      reasoning_effort=get("REASONING_EFFORT"))


def estimate_tokens(obj) -> int:
    """Rough count (~4 chars/token) used to reserve TPM budget before a call; settled with real usage after."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return max(1, len(text) // 4)


# ---------------------------------------------------------------- errors

class QuotaExhausted(Exception):
    """Daily quota is used up (local ledger or provider's per-day 429). Stop and resume later."""


class MalformedToolCall(Exception):
    """Provider rejected the model's own tool-call output (e.g. Groq 400 tool_use_failed)."""


DAILY_LIMIT_HINTS = ("per day", "perday", "(tpd)", "(rpd)", "daily")


def is_daily_limit(error: Exception) -> bool:
    return any(h in str(error).lower() for h in DAILY_LIMIT_HINTS)


def is_tool_use_failure(error: Exception) -> bool:
    text = str(error).lower()
    return "tool_use_failed" in text or "failed to call a function" in text


# ---------------------------------------------------------------- limits

class RateLimiter:
    """Sliding 60s window over requests and tokens for one provider+model bucket."""

    def __init__(self, rpm: int, tpm: int, clock=time.monotonic, sleep=time.sleep):
        self.rpm, self.tpm = rpm, tpm
        self.clock, self.sleep = clock, sleep
        self.events: deque = deque()  # [timestamp, tokens]
        self.lock = threading.Lock()

    def acquire(self, est_tokens: int) -> list:
        while True:
            with self.lock:
                now = self.clock()
                while self.events and now - self.events[0][0] >= 60:
                    self.events.popleft()
                used = sum(e[1] for e in self.events)
                rpm_ok = not self.rpm or len(self.events) < self.rpm
                # An oversized single request is allowed into an empty window rather than blocking forever.
                tpm_ok = not self.tpm or used + est_tokens <= self.tpm or not self.events
                if rpm_ok and tpm_ok:
                    entry = [now, est_tokens]
                    self.events.append(entry)
                    return entry
                wait = 60 - (now - self.events[0][0]) + 0.05
            self.sleep(max(wait, 0.05))

    def settle(self, entry: list, actual_tokens: int) -> None:
        with self.lock:
            entry[1] = actual_tokens


class DailyLedger:
    """Per-day request/token counts per bucket, persisted so separate runs share one budget.

    Approximate guard (UTC day; providers reset on their own schedules). The provider's
    per-day 429 is the real authority and is also turned into QuotaExhausted.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.lock = threading.Lock()

    def _path(self) -> Path:
        return self.directory / f"usage-{datetime.now(timezone.utc):%Y-%m-%d}.json"

    def _read(self) -> dict:
        path = self._path()
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def usage(self, bucket: str) -> dict:
        with self.lock:
            return self._read().get(bucket, {"requests": 0, "tokens": 0})

    def check(self, cfg: RoleConfig, est_tokens: int) -> None:
        used = self.usage(cfg.bucket)
        if cfg.rpd and used["requests"] + 1 > cfg.rpd:
            raise QuotaExhausted(f"{cfg.role}: daily request budget reached ({used['requests']}/{cfg.rpd}) for {cfg.model}")
        if cfg.tpd and used["tokens"] + est_tokens > cfg.tpd:
            raise QuotaExhausted(f"{cfg.role}: daily token budget reached ({used['tokens']}/{cfg.tpd}) for {cfg.model}")

    def record(self, bucket: str, tokens: int) -> None:
        with self.lock:
            data = self._read()
            entry = data.setdefault(bucket, {"requests": 0, "tokens": 0})
            entry["requests"] += 1
            entry["tokens"] += tokens
            self.directory.mkdir(parents=True, exist_ok=True)
            self._path().write_text(json.dumps(data, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- shared process state

_limiters: dict[str, RateLimiter] = {}
_limiters_lock = threading.Lock()
STATS: dict[str, dict] = defaultdict(lambda: defaultdict(int))  # role -> counters
_stats_lock = threading.Lock()


def bump(role: str, key: str, n: int = 1) -> None:
    with _stats_lock:
        STATS[role][key] += n


def stats_snapshot() -> dict:
    with _stats_lock:
        return {role: dict(counters) for role, counters in STATS.items()}


def limiter_for(cfg: RoleConfig) -> RateLimiter:
    """One limiter per provider+model. If two roles share a model, the stricter limits win."""
    with _limiters_lock:
        lim = _limiters.get(cfg.bucket)
        if lim is None:
            lim = _limiters[cfg.bucket] = RateLimiter(cfg.rpm, cfg.tpm)
        else:
            lim.rpm = min(x for x in (lim.rpm, cfg.rpm) if x) if (lim.rpm or cfg.rpm) else 0
            lim.tpm = min(x for x in (lim.tpm, cfg.tpm) if x) if (lim.tpm or cfg.tpm) else 0
        return lim


def cache_dir() -> Path:
    load_dotenv()
    return ROOT / os.environ.get("LLM_CACHE_DIR", ".llm_cache")


# ---------------------------------------------------------------- JSON helpers

FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_content(text: str | None):
    """Parse model output as JSON. Reads only message content (never a reasoning field);
    tolerates ```json fences and leading/trailing prose around one object."""
    if not text:
        raise ValueError("empty content")
    cleaned = FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


# ---------------------------------------------------------------- client

@dataclass
class ChatResult:
    message: dict          # verbatim assistant message; append this to history as-is
    usage: dict
    cached: bool
    latency: float
    finish_reason: str | None = None

    @property
    def content(self) -> str:
        return self.message.get("content") or ""

    @property
    def tool_calls(self) -> list[dict]:
        return self.message.get("tool_calls") or []


class LLM:
    def __init__(self, role: str, use_cache: bool = True):
        self.cfg = role_config(role)
        problems = self.cfg.problems()
        if problems:
            raise RuntimeError("; ".join(problems))
        self.role = role
        self.use_cache = use_cache
        self.client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key,
                             max_retries=0, timeout=env_int("LLM_TIMEOUT", 60))
        self.max_retries = env_int("LLM_MAX_RETRIES", 6)
        self.limiter = limiter_for(self.cfg)
        self.ledger = DailyLedger(cache_dir() / "usage")

    def _cache_path(self, request: dict, sample: int) -> Path:
        key = json.dumps({"model": self.cfg.model, "base_url": self.cfg.base_url,
                          "request": request, "sample": sample}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return cache_dir() / "responses" / digest[:2] / f"{digest}.json"

    def chat(self, messages: list[dict], tools: list[dict] | None = None, *, sample: int = 0,
             temperature: float | None = None, response_format: dict | None = None,
             max_tokens: int | None = None) -> ChatResult:
        request: dict = {"messages": messages}
        if tools:
            request["tools"] = [{"type": "function", "function": t} for t in tools]
        if temperature is not None:
            request["temperature"] = temperature
        if response_format:
            request["response_format"] = response_format
        if max_tokens:
            request["max_tokens"] = max_tokens
        if self.cfg.reasoning_effort:
            request["reasoning_effort"] = self.cfg.reasoning_effort

        path = self._cache_path(request, sample)
        if self.use_cache and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            bump(self.role, "cache_hits")
            return ChatResult(data["message"], data["usage"], True, 0.0, data.get("finish_reason"))

        est = estimate_tokens(request) + (max_tokens or 1024)
        self.ledger.check(self.cfg, est)
        resp, latency = self._call_with_retries(request, est)
        choice = resp.choices[0]
        message = choice.message.model_dump(exclude_none=True)
        usage = resp.usage.model_dump(exclude_none=True) if resp.usage else {}
        result = ChatResult(message, usage, False, latency, choice.finish_reason)
        if self.use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"message": message, "usage": usage,
                                        "finish_reason": choice.finish_reason}, ensure_ascii=False),
                            encoding="utf-8")
        return result

    def _call_with_retries(self, request: dict, est: int):
        for attempt in range(self.max_retries + 1):
            entry = self.limiter.acquire(est)
            started = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(model=self.cfg.model, **request)
            except openai.RateLimitError as e:
                self.limiter.settle(entry, est)
                bump(self.role, "http_429")
                if is_daily_limit(e):
                    raise QuotaExhausted(f"{self.role}: provider daily limit hit for {self.cfg.model}: {str(e)[:200]}")
                delay = self._retry_after(e) or self._backoff(attempt)
            except (openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError) as e:
                self.limiter.settle(entry, 0)
                bump(self.role, "http_5xx_or_timeout")
                delay = self._backoff(attempt)
                if attempt == self.max_retries:
                    raise
            except openai.BadRequestError as e:
                self.limiter.settle(entry, 0)
                bump(self.role, "http_400")
                if is_tool_use_failure(e):
                    raise MalformedToolCall(str(e)[:500]) from e
                raise
            else:
                latency = time.perf_counter() - started
                tokens = resp.usage.total_tokens if resp.usage else est
                self.limiter.settle(entry, tokens)
                self.ledger.record(self.cfg.bucket, tokens)
                bump(self.role, "calls")
                bump(self.role, "prompt_tokens", resp.usage.prompt_tokens if resp.usage else 0)
                bump(self.role, "completion_tokens", resp.usage.completion_tokens if resp.usage else 0)
                return resp, latency
            if attempt == self.max_retries:
                raise RuntimeError(f"{self.role}: still rate limited after {self.max_retries} retries")
            bump(self.role, "retries")
            time.sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(60.0, 2.0 * 2 ** attempt) * (0.5 + random.random() / 2)

    @staticmethod
    def _retry_after(error) -> float | None:
        try:
            value = error.response.headers.get("retry-after")
            return min(float(value), 120.0) if value else None
        except (AttributeError, ValueError):
            return None

    def chat_json(self, messages: list[dict], schema: dict, name: str, *, sample: int = 0,
                  temperature: float = 0.0, max_tokens: int | None = None) -> dict:
        """Structured output: strict json_schema first; json_object mode if the endpoint rejects schemas."""
        formats = [{"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
                   {"type": "json_object"}]
        last_error: Exception | None = None
        for fmt in formats:
            try:
                result = self.chat(messages, sample=sample, temperature=temperature,
                                   response_format=fmt, max_tokens=max_tokens)
            except openai.BadRequestError as e:
                last_error = e
                continue
            try:
                return parse_json_content(result.content)
            except ValueError as e:  # JSONDecodeError is a ValueError
                bump(self.role, "json_parse_failures")
                last_error = e
        raise ValueError(f"{self.role}: no valid JSON from {self.cfg.model}: {last_error}")
