"""Command-line entrypoint for local_agent.

Usage:
    python3 -m local_agent.cli --prompt "hello" --fake
    python3 -m local_agent.cli --prompt "hello" --config config.example.yaml

With --fake, no network I/O happens and no config file is required unless
you also pass --config (a FakeLLMClient is used regardless of backend
settings). Without --fake, a config file is loaded and an OllamaClient is
built from it — this path requires a real local Ollama server.
"""

from __future__ import annotations

import argparse
import sys

from local_agent.config import ConfigError, load_config
from local_agent.llm_client import FakeLLMClient, LLMClientError, OllamaClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local_agent",
        description="Run a prompt through a local-first LLM agent.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="The prompt text to send to the model.",
    )
    parser.add_argument(
        "--config",
        default="config.example.yaml",
        help="Path to a YAML config file (default: config.example.yaml).",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help=(
            "Use FakeLLMClient instead of a real backend. Does not require "
            "a running Ollama server or a valid config file's backend_url "
            "to be reachable. Intended for local testing/demo."
        ),
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.fake:
        client = FakeLLMClient()
    else:
        try:
            config = load_config(args.config)
        except ConfigError as e:
            print(f"Config error: {e}", file=sys.stderr)
            return 1
        client = OllamaClient(
            model_name=config.model_name,
            backend_url=config.backend_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    try:
        result = client.generate(args.prompt)
    except LLMClientError as e:
        print(f"Generation failed: {e}", file=sys.stderr)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(run())
