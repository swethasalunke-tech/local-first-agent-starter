# DESIGN.md

## Goal

A starter template for AI agents that run fully local — a local LLM backend
(e.g. Ollama) instead of a hosted API — so anyone can clone the repo and try
it without an API key.

## Day 1 scope (this commit)

- `local_agent/config.py`: `AgentConfig` dataclass + `load_config(path)`
  that reads a YAML file and validates it (required fields present,
  `temperature` in `[0, 2]`, `max_tokens` a positive int, `backend_url`
  well-formed). Ships with `config.example.yaml` (Ollama defaults).
- `local_agent/llm_client.py`: a `LocalLLMClient` Protocol (`generate(prompt) -> str`),
  a real `OllamaClient` implementation that POSTs to Ollama's documented
  `/api/generate` endpoint, and a `FakeLLMClient` for tests/demos.
- `local_agent/cli.py`: a real argparse-based CLI (`python3 -m local_agent.cli`)
  that loads config, builds a client, and prints a generation result. A
  `--fake` flag swaps in `FakeLLMClient` so the whole pipeline can be run
  and demonstrated with zero network dependencies.
- Test suite (`tests/`) covering config validation, `FakeLLMClient`, the
  CLI end-to-end (subprocess-level), and `OllamaClient`'s request-building
  and response-parsing logic against a mocked `requests.post`.

## What was NOT live-tested, and why

No Ollama server was reachable in the sandbox this repo was built in.
The check run was:

```
curl -s -m 3 http://localhost:11434/api/tags
```

Result: curl exit code 7 (connection refused / could not connect), no
response body. This means `OllamaClient` has never made a real HTTP call
to a real Ollama server as part of building this repo. Its correctness is
supported by:

1. The request it builds matching Ollama's published `/api/generate` API
   (model, prompt, stream, options.temperature, options.num_predict).
2. Unit tests (`tests/test_llm_client.py`) that mock `requests.post` and
   assert on the exact payload/headers sent, and on parsing of a mocked
   200 response and several mocked error conditions (connection error,
   non-200 status, invalid JSON, missing `response` field).

This is the same honesty pattern used in the `weekly-ai-tutor` repo for
Piper TTS: real code, tested against a mock because the real backend
wasn't available in the build sandbox, documented as such rather than
claimed as live-verified.

## Explicitly deferred (not in this commit)

- **Streaming responses** — Ollama supports `"stream": true` with
  newline-delimited JSON chunks; `OllamaClient` currently always sends
  `"stream": false` and returns the full text in one call. Planned for
  day 2 (see `BUILD-SCHEDULE.md`).
- **llama.cpp backend** — an alternative `LocalLLMClient` implementation
  targeting a local llama.cpp server, as a second backend option that
  doesn't depend on Ollama. Planned for day 3.
- **A genuine live end-to-end run against a real Ollama server** —
  deferred until this is tested on a machine that actually has `ollama
  serve` running and a model pulled. Until that happens, any claim of
  "it works with Ollama" should be understood as "the request/response
  code matches the documented API and passes mocked tests," not "it has
  been run live."
- Conversation/history management, tool-calling, retries/backoff, and
  multi-turn agent loops — none of that exists yet; day 1 is intentionally
  just config + a single-shot `generate()` call.
