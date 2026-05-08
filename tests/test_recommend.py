import asyncio
from unittest.mock import patch


def test_valid_recommendation_request(client, mock_recommend):
    resp = client.post(
        "/api/v1/recommend",
        json={"destination": "Paris", "budget": "mid_range"},
    )
    assert resp.status_code == 200
    assert resp.json()["destination"] == "Paris"


def test_empty_destination_returns_422(client):
    resp = client.post(
        "/api/v1/recommend",
        json={"destination": "", "budget": "mid_range"},
    )
    assert resp.status_code == 422


def test_invalid_budget_enum_returns_422(client):
    resp = client.post(
        "/api/v1/recommend",
        json={"destination": "Tokyo", "budget": "ultra_luxury"},
    )
    assert resp.status_code == 422


def test_destination_too_long_returns_422(client):
    resp = client.post(
        "/api/v1/recommend",
        json={"destination": "A" * 101, "budget": "budget"},
    )
    assert resp.status_code == 422


def test_gemini_timeout_returns_503(client):
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        resp = client.post(
            "/api/v1/recommend",
            json={"destination": "Paris", "budget": "luxury"},
        )
    assert resp.status_code == 503


def test_response_contains_required_fields(client, mock_recommend):
    resp = client.post(
        "/api/v1/recommend",
        json={"destination": "Paris", "budget": "mid_range"},
    )
    assert resp.status_code == 200
    data = resp.json()
    for field in ("destination", "highlights", "estimated_cost", "best_time", "tips"):
        assert field in data, f"Missing field: {field}"
    assert isinstance(data["highlights"], list) and len(data["highlights"]) > 0
    assert isinstance(data["tips"], list) and len(data["tips"]) > 0
