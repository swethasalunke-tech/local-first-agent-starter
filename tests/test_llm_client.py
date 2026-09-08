"""Tests for local_agent.llm_client.

FakeLLMClient tests exercise real code with no mocking needed. OllamaClient
tests mock `requests.post` (via unittest.mock) to verify the real
request-building and response-parsing logic without requiring a live
Ollama server — no local Ollama instance was reachable in the build
sandbox this repo was created in (see README.md and llm_client.py's
module docstring for the exact reachability check that was run).
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from local_agent.llm_client import FakeLLMClient, LLMClientError, OllamaClient


def test_fake_llm_client_returns_canned_output():
    client = FakeLLMClient(canned_response="hello from fake")
    result = client.generate("any prompt")
    assert result == "hello from fake"


def test_fake_llm_client_default_response():
    client = FakeLLMClient()
    result = client.generate("any prompt")
    assert isinstance(result, str)
    assert len(result) > 0


def test_fake_llm_client_records_calls():
    client = FakeLLMClient()
    client.generate("first")
    client.generate("second")
    assert client.calls == ["first", "second"]


@patch("local_agent.llm_client.requests.post")
def test_ollama_client_sends_correct_payload(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "mocked completion text"}
    mock_post.return_value = mock_response

    client = OllamaClient(
        model_name="llama3.2",
        backend_url="http://localhost:11434",
        temperature=0.5,
        max_tokens=256,
    )
    result = client.generate("What is 2+2?")

    assert result == "mocked completion text"

    # Verify the request was built correctly.
    assert mock_post.call_count == 1
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == "http://localhost:11434/api/generate"
    assert call_kwargs["json"] == {
        "model": "llama3.2",
        "prompt": "What is 2+2?",
        "stream": False,
        "options": {
            "temperature": 0.5,
            "num_predict": 256,
        },
    }
    assert call_kwargs["headers"] == {"Content-Type": "application/json"}
    assert call_kwargs["timeout"] == 60.0


@patch("local_agent.llm_client.requests.post")
def test_ollama_client_strips_trailing_slash_from_backend_url(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "ok"}
    mock_post.return_value = mock_response

    client = OllamaClient(
        model_name="llama3.2",
        backend_url="http://localhost:11434/",
    )
    client.generate("hi")

    call_args, _ = mock_post.call_args
    assert call_args[0] == "http://localhost:11434/api/generate"


@patch("local_agent.llm_client.requests.post")
def test_ollama_client_raises_on_connection_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("refused")

    client = OllamaClient(model_name="llama3.2", backend_url="http://localhost:11434")
    with pytest.raises(LLMClientError, match="Failed to reach Ollama backend"):
        client.generate("hi")


@patch("local_agent.llm_client.requests.post")
def test_ollama_client_raises_on_non_200_status(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "internal server error"
    mock_post.return_value = mock_response

    client = OllamaClient(model_name="llama3.2", backend_url="http://localhost:11434")
    with pytest.raises(LLMClientError, match="HTTP 500"):
        client.generate("hi")


@patch("local_agent.llm_client.requests.post")
def test_ollama_client_raises_on_invalid_json(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("not json")
    mock_post.return_value = mock_response

    client = OllamaClient(model_name="llama3.2", backend_url="http://localhost:11434")
    with pytest.raises(LLMClientError, match="non-JSON response"):
        client.generate("hi")


@patch("local_agent.llm_client.requests.post")
def test_ollama_client_raises_on_missing_response_field(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"unexpected": "shape"}
    mock_post.return_value = mock_response

    client = OllamaClient(model_name="llama3.2", backend_url="http://localhost:11434")
    with pytest.raises(LLMClientError, match="missing 'response' field"):
        client.generate("hi")


def test_ollama_client_is_a_local_llm_client():
    from local_agent.llm_client import LocalLLMClient

    client = OllamaClient(model_name="llama3.2", backend_url="http://localhost:11434")
    assert isinstance(client, LocalLLMClient)
    fake = FakeLLMClient()
    assert isinstance(fake, LocalLLMClient)
