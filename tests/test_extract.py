import json
import pytest
from unittest.mock import MagicMock


class _FakeListResponse:
    def __init__(self, destinations: list) -> None:
        self.text = json.dumps(destinations)


@pytest.fixture
def mock_extract(monkeypatch):
    model = MagicMock()
    model.generate_content.return_value = _FakeListResponse(["Tokyo", "Kyoto", "Osaka"])
    monkeypatch.setattr("backend.main._get_extract_model", lambda: model)
    return model


def test_valid_text_returns_destinations(client, mock_extract):
    resp = client.post(
        "/api/v1/extract-destinations",
        json={"text": "I want to visit Tokyo and then go to Kyoto for the temples."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "destinations" in data
    assert isinstance(data["destinations"], list)
    assert len(data["destinations"]) > 0


def test_empty_text_returns_empty_list(client):
    resp = client.post("/api/v1/extract-destinations", json={"text": "   "})
    assert resp.status_code == 200
    assert resp.json() == {"destinations": []}


def test_whitespace_only_text_returns_empty_list(client):
    resp = client.post("/api/v1/extract-destinations", json={"text": "\n\t\n"})
    assert resp.status_code == 200
    assert resp.json()["destinations"] == []


def test_returns_at_most_five_destinations(client, monkeypatch):
    model = MagicMock()
    model.generate_content.return_value = _FakeListResponse(
        ["Rome", "Paris", "London", "Berlin", "Madrid", "Vienna", "Lisbon"]
    )
    monkeypatch.setattr("backend.main._get_extract_model", lambda: model)
    resp = client.post(
        "/api/v1/extract-destinations",
        json={"text": "I'd love to see Rome, Paris, London, Berlin, Madrid, Vienna, and Lisbon."},
    )
    assert resp.status_code == 200
    assert len(resp.json()["destinations"]) <= 5


def test_gemini_failure_returns_empty_list_gracefully(client, monkeypatch):
    model = MagicMock()
    model.generate_content.side_effect = Exception("Gemini unavailable")
    monkeypatch.setattr("backend.main._get_extract_model", lambda: model)
    resp = client.post(
        "/api/v1/extract-destinations",
        json={"text": "Plan a trip to Rome."},
    )
    assert resp.status_code == 200
    assert resp.json()["destinations"] == []


def test_text_too_long_returns_422(client):
    resp = client.post(
        "/api/v1/extract-destinations",
        json={"text": "x" * 4_001},
    )
    assert resp.status_code == 422
