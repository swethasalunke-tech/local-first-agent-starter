"""Tests for local_agent.cli.

Exercises the real CLI entrypoint end-to-end with --fake, which requires
no network I/O and no local Ollama server, so it runs the same way in CI,
this build sandbox, and on a contributor's machine.
"""

import subprocess
import sys

from local_agent.cli import run


def test_cli_run_with_fake_returns_zero_and_prints_output(capsys):
    exit_code = run(["--prompt", "hello world", "--fake"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "fake response" in captured.out.lower()


def test_cli_run_without_config_and_without_fake_reports_config_error(capsys, tmp_path):
    missing_config = str(tmp_path / "nope.yaml")
    exit_code = run(["--prompt", "hi", "--config", missing_config])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Config error" in captured.err


def test_cli_as_subprocess_with_fake_flag():
    # Genuine end-to-end run: invoke the module as `python3 -m local_agent.cli`
    # exactly as a user would, in a real subprocess.
    result = subprocess.run(
        [sys.executable, "-m", "local_agent.cli", "--prompt", "hello", "--fake"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "fake response" in result.stdout.lower()
