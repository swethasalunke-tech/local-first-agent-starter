"""LLM client interface and implementations for local_agent.

HONESTY NOTE (matches the caveat pattern used in the `weekly-ai-tutor`
repo for Piper TTS): `OllamaClient` below is real, complete request/response
handling code written against Ollama's documented `/api/generate` HTTP API.
It has NOT been exercised against a live Ollama server in this build
sandbox — a reachability check (`curl -m 3 http://localhost:11434/api/tags`)
was run during development and got a connection error (no listener on
that port in this sandbox), so no live Ollama instance was available to
test against. `OllamaClient`'s request-building and response-parsing logic
is instead covered by unit tests in `tests/test_llm_client.py` that mock
`requests.post`. Anyone running this on a machine with `ollama serve`
actually running will be exercising real, untested-live code paths for
the first time — they are expected to work based on Ollama's published
API docs, but "expected to work" is not the same claim as "verified live."
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import requests


class LLMClientError(RuntimeError):
    """Raised when an LLM backend request fails or returns something unusable."""


@runtime_checkable
class LocalLLMClient(Protocol):
    """Interface all local LLM clients must implement."""

    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt and return it as text."""
        ...


class OllamaClient:
    """Client for a local Ollama server's `/api/generate` endpoint.

    See: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion

    This class has real request-building and response-parsing logic. It has
    only been tested with a mocked `requests.post` (see
    tests/test_llm_client.py), not against a live Ollama server, because no
    Ollama instance was reachable in the build sandbox used to create this
    repo. See the module docstring above for details.
    """

    def __init__(
        self,
        model_name: str,
        backend_url: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.backend_url = backend_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """Send `prompt` to the Ollama `/api/generate` endpoint and return the text.

        Raises:
            LLMClientError: If the HTTP request fails, the server returns a
                non-2xx status, the response is not valid JSON, or the
                response JSON does not contain the expected 'response' field.
        """
        url = f"{self.backend_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
        except requests.exceptions.RequestException as e:
            raise LLMClientError(
                f"Failed to reach Ollama backend at {url}: {e}"
            ) from e

        if response.status_code != 200:
            raise LLMClientError(
                f"Ollama backend returned HTTP {response.status_code}: "
                f"{response.text}"
            )

        try:
            data = response.json()
        except ValueError as e:
            raise LLMClientError(
                f"Ollama backend returned non-JSON response: {e}"
            ) from e

        if "response" not in data:
            raise LLMClientError(
                f"Ollama backend response missing 'response' field: {data!r}"
            )

        return data["response"]


class FakeLLMClient:
    """A canned-response client for tests and for `--fake` CLI runs.

    Does not perform any network I/O. Useful for verifying the rest of the
    pipeline (config loading, CLI wiring) genuinely runs end-to-end without
    requiring a local LLM backend to be installed.
    """

    def __init__(self, canned_response: str = "This is a fake response.") -> None:
        self.canned_response = canned_response
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.canned_response
