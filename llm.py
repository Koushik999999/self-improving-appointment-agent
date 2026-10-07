"""LLM access for every role (agent, sim, judge, improver) via OpenAI-compatible endpoints.

Configuration is per role, from the environment (.env is loaded, real env vars win):
    {ROLE}_PROVIDER   groq | gemini | openrouter | ollama   (fills base URL + key var)
    {ROLE}_MODEL      model id at that provider
    {ROLE}_BASE_URL / {ROLE}_API_KEY   optional explicit overrides
    {ROLE}_RPM / _TPM / _RPD / _TPD    client-side limits (0 = unknown/unlimited)
IMPROVER falls back to JUDGE settings when unset.
"""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROLES = ("agent", "sim", "judge", "improver")

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


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


@dataclass(frozen=True)
class RoleConfig:
    role: str
    provider: str
    base_url: str
    api_key: str
    model: str
    rpm: int
    tpm: int
    rpd: int
    tpd: int

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
                      rpd=int(get("RPD", 0)), tpd=int(get("TPD", 0)))
