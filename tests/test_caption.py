from backend.main import Mood


def test_valid_caption_request(client, mock_caption):
    resp = client.post(
        "/api/v1/caption",
        json={"destination": "Santorini", "mood": "romance"},
    )
    assert resp.status_code == 200


def test_all_mood_types_accepted(client, mock_caption):
    for mood in Mood:
        resp = client.post(
            "/api/v1/caption",
            json={"destination": "Paris", "mood": mood.value},
        )
        assert resp.status_code == 200, f"Mood '{mood.value}' returned {resp.status_code}"


def test_returns_non_empty_caption(client, mock_caption):
    resp = client.post(
        "/api/v1/caption",
        json={"destination": "Santorini", "mood": "romance"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "caption" in data
    assert isinstance(data["caption"], str)
    assert len(data["caption"]) > 0


def test_response_includes_hashtags(client, mock_caption):
    resp = client.post(
        "/api/v1/caption",
        json={"destination": "Tokyo", "mood": "adventure"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "hashtags" in data
    assert len(data["hashtags"]) > 0
    assert all(tag.startswith("#") for tag in data["hashtags"])


def test_photo_description_is_optional(client, mock_caption):
    resp_without = client.post(
        "/api/v1/caption",
        json={"destination": "Bali", "mood": "relaxation"},
    )
    resp_with = client.post(
        "/api/v1/caption",
        json={
            "destination": "Bali",
            "mood": "relaxation",
            "photo_description": "rice terraces at golden hour",
        },
    )
    assert resp_without.status_code == 200
    assert resp_with.status_code == 200
