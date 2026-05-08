def test_valid_month_returns_destinations(client, mock_offseason):
    resp = client.get("/api/v1/offseason?month=6")
    assert resp.status_code == 200
    data = resp.json()
    assert "destinations" in data
    assert len(data["destinations"]) > 0


def test_invalid_month_returns_422(client):
    resp = client.get("/api/v1/offseason?month=13")
    assert resp.status_code == 422


def test_budget_filter_low_returns_cheap_options(client, mock_offseason):
    resp = client.get("/api/v1/offseason?month=1&budget=budget")
    assert resp.status_code == 200
    data = resp.json()
    assert "destinations" in data
    assert len(data["destinations"]) > 0


def test_response_has_price_comparison(client, mock_offseason):
    resp = client.get("/api/v1/offseason?month=1")
    assert resp.status_code == 200
    dest = resp.json()["destinations"][0]
    assert "typical_price" in dest
    assert "offseason_price" in dest
    assert "savings_percent" in dest
    assert isinstance(dest["savings_percent"], int)


def test_surprise_me_endpoint(client, mock_offseason):
    resp = client.get("/api/v1/offseason/surprise")
    assert resp.status_code == 200
    data = resp.json()
    assert "destinations" in data
    assert len(data["destinations"]) > 0
