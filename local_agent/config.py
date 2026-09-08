"""Configuration loading for local_agent.

Loads a YAML file into a validated `AgentConfig` dataclass. No network
calls happen here — this module is pure config parsing + validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

REQUIRED_FIELDS = ("model_name", "backend_url", "temperature", "max_tokens")


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or invalid."""


@dataclass
class AgentConfig:
    """Validated configuration for a local agent run.

    Attributes:
        model_name: Name of the model to request from the backend
            (e.g. "llama3.2" for Ollama).
        backend_url: Base URL of the local LLM backend
            (e.g. "http://localhost:11434" for Ollama).
        temperature: Sampling temperature, must be in [0, 2].
        max_tokens: Maximum number of tokens to generate, must be > 0.
    """

    model_name: str
    backend_url: str
    temperature: float
    max_tokens: int

    def __post_init__(self) -> None:
        _validate_model_name(self.model_name)
        _validate_backend_url(self.backend_url)
        _validate_temperature(self.temperature)
        _validate_max_tokens(self.max_tokens)


def _validate_model_name(model_name: object) -> None:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ConfigError(
            f"'model_name' must be a non-empty string, got: {model_name!r}"
        )


def _validate_backend_url(backend_url: object) -> None:
    if not isinstance(backend_url, str) or not backend_url.strip():
        raise ConfigError(
            f"'backend_url' must be a non-empty string, got: {backend_url!r}"
        )
    if not (backend_url.startswith("http://") or backend_url.startswith("https://")):
        raise ConfigError(
            "'backend_url' must start with 'http://' or 'https://', "
            f"got: {backend_url!r}"
        )


def _validate_temperature(temperature: object) -> None:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ConfigError(
            f"'temperature' must be a number, got: {temperature!r}"
        )
    if not (0 <= temperature <= 2):
        raise ConfigError(
            f"'temperature' must be within [0, 2], got: {temperature!r}"
        )


def _validate_max_tokens(max_tokens: object) -> None:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ConfigError(
            f"'max_tokens' must be an integer, got: {max_tokens!r}"
        )
    if max_tokens <= 0:
        raise ConfigError(
            f"'max_tokens' must be a positive integer, got: {max_tokens!r}"
        )


def load_config(path: str) -> AgentConfig:
    """Load and validate an `AgentConfig` from a YAML file.

    Args:
        path: Path to a YAML config file.

    Returns:
        A validated AgentConfig instance.

    Raises:
        ConfigError: If the file does not exist, is not valid YAML, is
            missing required fields, or contains out-of-range values.
    """
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Config file is not valid YAML: {e}") from e

    if raw is None:
        raise ConfigError(f"Config file is empty: {path}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config file must contain a YAML mapping, got: {type(raw).__name__}"
        )

    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ConfigError(
            f"Config file is missing required field(s): {', '.join(missing)}"
        )

    # AgentConfig.__post_init__ performs type/range validation.
    return AgentConfig(
        model_name=raw["model_name"],
        backend_url=raw["backend_url"],
        temperature=raw["temperature"],
        max_tokens=raw["max_tokens"],
    )
