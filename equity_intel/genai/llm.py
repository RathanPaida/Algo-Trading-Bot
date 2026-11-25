"""Language-model backends behind one tiny interface:  llm.complete(system, prompt) -> str

    ClaudeLLM   - Anthropic's Claude via the official `anthropic` SDK (needs credentials)
    OllamaLLM   - a free model running locally with Ollama (https://ollama.com)
    OfflineLLM  - no model at all; callers fall back to templates, so the app always works

Choose with config.yaml -> genai.provider: auto | anthropic | ollama | offline.
`auto` uses Claude if ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN is set, else Ollama if it
is running on this machine, else offline.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The model could not produce an answer (network, auth, refusal ...)."""


class OfflineLLM:
    name = "offline (no language model)"
    available = False

    def complete(self, system: str, prompt: str, max_tokens: int = 4000) -> str:
        raise LLMError("No language model configured")


class ClaudeLLM:
    """Claude through the official Anthropic Python SDK."""

    available = True

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium"):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / ant profile
        self.model, self.effort = model, effort
        self.name = f"Claude ({model})"

    def complete(self, system: str, prompt: str, max_tokens: int = 16000) -> str:
        a = self._anthropic
        try:
            # Server-side fallback: if a safety classifier declines, the API retries the
            # same request on a suitable fallback model inside the same call.
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except a.AuthenticationError as e:
            raise LLMError("Claude: invalid or missing API credentials") from e
        except a.RateLimitError as e:
            raise LLMError("Claude: rate limited - try again shortly") from e
        except a.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMError("Claude: network error") from e

        if response.stop_reason == "refusal":
            raise LLMError("Claude declined to answer this request")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise LLMError("Claude returned an empty answer")
        return text


class OllamaLLM:
    """A local open-source model served by Ollama's REST API."""

    available = True

    def __init__(self, model: str = "llama3.1", url: str = "http://localhost:11434"):
        self.model, self.url = model, url.rstrip("/")
        self.name = f"Ollama ({model}, local)"

    def complete(self, system: str, prompt: str, max_tokens: int = 4000) -> str:
        import httpx

        try:
            r = httpx.post(f"{self.url}/api/chat", timeout=180, json={
                "model": self.model, "stream": False, "options": {"num_predict": max_tokens},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            })
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        return r.json()["message"]["content"].strip()

    @staticmethod
    def is_running(url: str) -> bool:
        import httpx

        try:
            return httpx.get(f"{url.rstrip('/')}/api/tags", timeout=1.5).status_code == 200
        except httpx.HTTPError:
            return False


def get_llm(cfg: dict):
    g = cfg.get("genai", {})
    provider = g.get("provider", "auto")
    if provider == "auto":
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            provider = "anthropic"
        elif OllamaLLM.is_running(g.get("ollama_url", "http://localhost:11434")):
            provider = "ollama"
        else:
            provider = "offline"
    try:
        if provider == "anthropic":
            return ClaudeLLM(g.get("anthropic_model", "claude-opus-5-5"), g.get("effort", "medium"))
        if provider == "ollama":
            return OllamaLLM(g.get("ollama_model", "llama3.1"), g.get("ollama_url", "http://localhost:11434"))
    except ImportError:
        log.warning("Package for provider '%s' not installed - using offline mode", provider)
    return OfflineLLM()
