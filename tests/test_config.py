"""Tests for local_agent.config."""

import os
import tempfile

import pytest

from local_agent.config import AgentConfig, ConfigError, load_config


def _write_yaml(tmp_path, content):
    path = os.path.join(tmp_path, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def test_valid_config_loads_correctly(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        model_name: llama3.2
        backend_url: http://localhost:11434
        temperature: 0.7
        max_tokens: 512
        """,
    )
    config = load_config(path)
    assert isinstance(config, AgentConfig)
    assert config.model_name == "llama3.2"
    assert config.backend_url == "http://localhost:11434"
    assert config.temperature == 0.7
    assert config.max_tokens == 512


def test_bundled_example_config_loads(tmp_path):
    # Sanity check against the actual shipped config.example.yaml
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_path = os.path.join(repo_root, "config.example.yaml")
    config = load_config(example_path)
    assert config.backend_url.startswith("http://localhost")


def test_missing_field_raises(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        model_name: llama3.2
        backend_url: http://localhost:11434
        temperature: 0.7
        """,
    )
    with pytest.raises(ConfigError, match="max_tokens"):
        load_config(path)


def test_missing_multiple_fields_raises(tmp_path):
    path = _write_yaml(tmp_path, "model_name: llama3.2\n")
    with pytest.raises(ConfigError) as exc_info:
        load_config(path)
    assert "backend_url" in str(exc_info.value)
    assert "temperature" in str(exc_info.value)
    assert "max_tokens" in str(exc_info.value)


def test_temperature_out_of_range_high_raises(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        model_name: llama3.2
        backend_url: http://localhost:11434
        temperature: 2.5
        max_tokens: 512
        """,
    )
    with pytest.raises(ConfigError, match="temperature"):
        load_config(path)


def test_temperature_out_of_range_negative_raises(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        model_name: llama3.2
        backend_url: http://localhost:11434
        temperature: -0.1
        max_tokens: 512
        """,
    )
    with pytest.raises(ConfigError, match="temperature"):
        load_config(path)


def test_temperature_boundary_values_are_valid(tmp_path):
    path_low = _write_yaml(
        tmp_path,
        "model_name: m\nbackend_url: http://localhost:11434\ntemperature: 0\nmax_tokens: 1\n",
    )
    config_low = load_config(path_low)
    assert config_low.temperature == 0

    path_high = _write_yaml(
        tmp_path,
        "model_name: m\nbackend_url: http://localhost:11434\ntemperature: 2\nmax_tokens: 1\n",
    )
    config_high = load_config(path_high)
    assert config_high.temperature == 2


def test_negative_max_tokens_raises(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        model_name: llama3.2
        backend_url: http://localhost:11434
        temperature: 0.7
        max_tokens: -5
        """,
    )
    with pytest.raises(ConfigError, match="max_tokens"):
        load_config(path)


def test_zero_max_tokens_raises(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        model_name: llama3.2
        backend_url: http://localhost:11434
        temperature: 0.7
        max_tokens: 0
        """,
    )
    with pytest.raises(ConfigError, match="max_tokens"):
        load_config(path)


def test_empty_model_name_raises(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        model_name: ""
        backend_url: http://localhost:11434
        temperature: 0.7
        max_tokens: 512
        """,
    )
    with pytest.raises(ConfigError, match="model_name"):
        load_config(path)


def test_invalid_backend_url_scheme_raises(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        model_name: llama3.2
        backend_url: ftp://localhost:11434
        temperature: 0.7
        max_tokens: 512
        """,
    )
    with pytest.raises(ConfigError, match="backend_url"):
        load_config(path)


def test_nonexistent_file_raises(tmp_path):
    missing_path = os.path.join(tmp_path, "does_not_exist.yaml")
    with pytest.raises(ConfigError, match="not found"):
        load_config(missing_path)


def test_empty_file_raises(tmp_path):
    path = _write_yaml(tmp_path, "")
    with pytest.raises(ConfigError, match="empty"):
        load_config(path)


def test_non_mapping_yaml_raises(tmp_path):
    path = _write_yaml(tmp_path, "- a\n- b\n- c\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_invalid_yaml_syntax_raises(tmp_path):
    path = _write_yaml(tmp_path, "model_name: [unclosed\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_agentconfig_direct_construction_validates():
    with pytest.raises(ConfigError, match="temperature"):
        AgentConfig(
            model_name="m",
            backend_url="http://localhost:11434",
            temperature=99,
            max_tokens=10,
        )
