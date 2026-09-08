# local-first-agent-starter

A starter template for AI agents that run fully local — no API key required
to try it out. Backend: [Ollama](https://ollama.com) (local LLM server).

Day 1 of this repo. See `DESIGN.md` for scope and `BUILD-SCHEDULE.md` for
what's planned next.

## What's implemented (day 1)

- `local_agent/config.py` — `AgentConfig` dataclass and `load_config(path)`,
  loading and validating a YAML config file (required fields, `temperature`
  in `[0, 2]`, positive `max_tokens`, well-formed `backend_url`). Ships with
  `config.example.yaml` (Ollama defaults: `http://localhost:11434`).
- `local_agent/llm_client.py` — a `LocalLLMClient` Protocol, a real
  `OllamaClient` that POSTs to Ollama's `/api/generate` endpoint, and a
  `FakeLLMClient` for offline testing/demos.
- `local_agent/cli.py` — a runnable CLI: `python3 -m local_agent.cli --prompt "..." [--fake | --config path.yaml]`.
- Test suite: config validation, `FakeLLMClient`, CLI end-to-end (including
  a real subprocess invocation), and `OllamaClient`'s request/response
  handling tested against a mocked `requests.post`.

## Honest status: was Ollama reachable in the build sandbox?

**No.** During development the following check was run:

```
curl -s -m 3 http://localhost:11434/api/tags
```

Result: **curl exit code 7** (connection refused / could not connect), HTTP
status `000`, no response body. No process was listening on port 11434 in
the build sandbox.

Because of that, `OllamaClient` has **not** been exercised against a real,
running Ollama server as part of building this repo. What *has* happened:

- `OllamaClient`'s request-building and response-parsing code was written
  against Ollama's documented `/api/generate` API.
- `tests/test_llm_client.py` mocks `requests.post` (via `unittest.mock`)
  and asserts the exact JSON payload/headers `OllamaClient` sends, and that
  it correctly parses a mocked 200 response and several mocked error cases.
- The CLI's `--fake` path (using `FakeLLMClient`, no network I/O) was run
  for real, including as an actual subprocess (`python3 -m local_agent.cli --prompt "hello" --fake`),
  and genuinely passes.

This mirrors the honesty pattern used in the `weekly-ai-tutor` repo for
Piper TTS on this same account: real code, real mock-based test coverage,
explicitly *not* claimed as live-verified where it wasn't.

## How to run tests

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
```

Actual result from this build (29 tests, all passing):

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.1.1, pluggy-1.6.0
collected 29 items

tests/test_cli.py::test_cli_run_with_fake_returns_zero_and_prints_output PASSED
tests/test_cli.py::test_cli_run_without_config_and_without_fake_reports_config_error PASSED
tests/test_cli.py::test_cli_as_subprocess_with_fake_flag PASSED
tests/test_config.py (16 tests) PASSED
tests/test_llm_client.py (10 tests) PASSED

============================== 29 passed in 0.13s ==============================
```

## How to try it right now, without Ollama (no API key, no local model)

```bash
python3 -m local_agent.cli --prompt "hello" --fake
```

This uses `FakeLLMClient` and prints a canned response. It proves the
config → client → CLI wiring genuinely runs end to end.

## How to run it for real, with Ollama installed locally

1. Install Ollama: https://ollama.com/download
2. Pull a model: `ollama pull llama3.2`
3. Make sure Ollama is running (`ollama serve`, or it may already be
   running as a background service after install).
4. Copy the example config and adjust if needed:
   ```bash
   cp config.example.yaml config.yaml
   ```
5. Run:
   ```bash
   python3 -m local_agent.cli --prompt "Explain local-first software in one sentence." --config config.yaml
   ```

This path (real Ollama call, no `--fake`) has **not** been live-tested by
the repo author as of this commit, for the reachability reason above. If
you try it and hit an issue, please open an issue — real bug reports
against real code are exactly what day-2/day-3 work should be informed by.

## Requirements

See `requirements.txt`: `pytest`, `requests`, `pyyaml`.

## Project structure

```
local_agent/
  __init__.py
  config.py       # AgentConfig + load_config()
  llm_client.py   # LocalLLMClient Protocol, OllamaClient, FakeLLMClient
  cli.py          # CLI entrypoint
tests/
  test_config.py
  test_llm_client.py
  test_cli.py
config.example.yaml
DESIGN.md
BUILD-SCHEDULE.md
requirements.txt
```
