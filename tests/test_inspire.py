def test_valid_aesthetic_input(client, mock_inspire):
    resp = client.post("/api/v1/inspire", json={"aesthetic": "dark academia"})
    assert resp.status_code == 200


def test_empty_aesthetic_returns_422(client):
    resp = client.post("/api/v1/inspire", json={"aesthetic": ""})
    assert resp.status_code == 422


def test_returns_three_destinations(client, mock_inspire):
    resp = client.post("/api/v1/inspire", json={"aesthetic": "cottagecore"})
    assert resp.status_code == 200
    data = resp.json()
    assert "destinations" in data
    assert len(data["destinations"]) == 3


def test_includes_hashtags(client, mock_inspire):
    resp = client.post("/api/v1/inspire", json={"aesthetic": "dark academia"})
    assert resp.status_code == 200
    for dest in resp.json()["destinations"]:
        assert "hashtags" in dest
        assert len(dest["hashtags"]) > 0
        assert all(tag.startswith("#") for tag in dest["hashtags"])


def test_includes_photo_spots(client, mock_inspire):
    resp = client.post("/api/v1/inspire", json={"aesthetic": "dark academia"})
    assert resp.status_code == 200
    for dest in resp.json()["destinations"]:
        assert "photo_spots" in dest
        assert len(dest["photo_spots"]) > 0
