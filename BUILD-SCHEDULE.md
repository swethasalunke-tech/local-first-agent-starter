# BUILD-SCHEDULE.md

## Day 1 (this commit) — config + client interface + Ollama implementation

- `AgentConfig` dataclass + `load_config()` with real YAML loading and
  validation.
- `LocalLLMClient` Protocol, `OllamaClient` (real request/response code,
  tested via mocked `requests.post` — no live Ollama server was reachable
  in the build sandbox), `FakeLLMClient` for offline testing/demos.
- Runnable CLI (`local_agent/cli.py`) with a `--fake` flag so the full
  pipeline can be exercised with zero external dependencies.
- Full test suite passing (`pytest tests/`).
- `DESIGN.md`, `README.md`, this file.

## Day 2 (planned) — streaming support

- Add `stream: true` handling to `OllamaClient.generate()`, or a new
  `generate_stream()` method that yields chunks as Ollama sends them
  (newline-delimited JSON over the response body).
- Update the CLI to print tokens as they arrive instead of waiting for
  the full response.
- New tests mocking a streamed response (iterable of chunked JSON lines)
  to verify chunk parsing and reassembly, still without requiring a live
  Ollama server.
- If a real Ollama server becomes reachable in whatever environment day 2
  is built in, attempt one real live call and honestly report the actual
  result (success or failure) rather than assuming it will work.

## Day 3 (planned) — llama.cpp backend as an alternative

- Add a `LlamaCppClient` implementing the same `LocalLLMClient` Protocol,
  targeting a local llama.cpp server's HTTP completion endpoint, as a
  second no-API-key backend option for users who prefer llama.cpp over
  Ollama.
- Extend `AgentConfig`/`config.example.yaml` to select a backend
  (`backend: ollama` vs `backend: llamacpp`) and route the CLI to the
  right client.
- Tests mirroring the `OllamaClient` test structure: mocked HTTP calls
  verifying request-building and response-parsing.
