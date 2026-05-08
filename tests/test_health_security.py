# Tests for the /health endpoint, security response headers, and cross-cutting
# input validation that doesn't belong to a single feature endpoint.


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200


def test_health_status_is_ok(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"


def test_health_model_is_gemini_flash(client):
    assert client.get("/api/v1/health").json()["model"] == "gemini-2.5-flash"


def test_health_firestore_and_cache_are_booleans(client):
    data = client.get("/api/v1/health").json()
    assert isinstance(data["firestore"], bool)
    assert isinstance(data["cache"], bool)


# ── Security headers ──────────────────────────────────────────────────────────

def test_security_header_nosniff(client):
    resp = client.get("/api/v1/health")
    assert resp.headers.get("x-content-type-options") == "nosniff"


def test_security_header_deny_framing(client):
    resp = client.get("/api/v1/health")
    assert resp.headers.get("x-frame-options") == "DENY"


def test_security_header_referrer_policy(client):
    resp = client.get("/api/v1/health")
    assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"


def test_security_header_request_id_injected(client):
    resp = client.get("/api/v1/health")
    rid = resp.headers.get("x-request-id", "")
    assert len(rid) > 0


def test_security_header_csp_blocks_framing(client):
    csp = client.get("/api/v1/health").headers.get("content-security-policy", "")
    assert "default-src" in csp
    assert "frame-ancestors 'none'" in csp


# ── Input validation (cross-cutting) ─────────────────────────────────────────

def test_html_injection_in_destination_is_sanitized(client, mock_recommend):
    # <b>Paris</b> → stripped to "Paris" by the field_validator; request succeeds.
    resp = client.post(
        "/api/v1/recommend",
        json={"destination": "<b>Paris</b>", "budget": "mid_range"},
    )
    assert resp.status_code == 200


def test_oversized_chat_message_returns_422(client):
    # Message.content has max_length=20_000; one extra char must fail validation.
    resp = client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "user", "content": "x" * 20_001}]},
    )
    assert resp.status_code == 422


def test_invalid_session_id_format_returns_422(client):
    # Pattern ^[\w\-]{1,64}$ rejects path-traversal characters.
    resp = client.get("/api/v1/recent-searches?session_id=../../etc/passwd")
    assert resp.status_code == 422
