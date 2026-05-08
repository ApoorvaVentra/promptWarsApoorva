import json
import os
import sys

# Must be set before the app module is imported so that _resolve_secret()
# falls back to this value and the startup check passes.
os.environ.setdefault("GEMINI_API_KEY", "pytest-fake-key")
# Clear auth key so the middleware does not block unauthenticated test requests.
os.environ.pop("VOYAGER_API_KEY", None)

# Project root → allows `import backend.main` as a namespace package.
# Backend dir → satisfies `from exceptions import …` inside main.py.
_ROOT = os.path.join(os.path.dirname(__file__), "..")
_BACKEND = os.path.join(_ROOT, "backend")
for _p in (_ROOT, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from backend.main import app  # noqa: E402 — path must be set first


# ---------------------------------------------------------------------------
# Fake Gemini response payloads
# ---------------------------------------------------------------------------

FAKE_RECOMMEND = {
    "destination": "Paris",
    "highlights": ["Eiffel Tower", "Louvre Museum", "Seine River cruise"],
    "estimated_cost": "$2,000-$3,000",
    "best_time": "April-June",
    "tips": ["Book museums in advance", "Get a Metro day pass"],
}

FAKE_OFFSEASON = {
    "month": 1,
    "destinations": [
        {
            "name": "Lisbon",
            "typical_price": "$2,500",
            "offseason_price": "$1,200",
            "savings_percent": 52,
            "why_visit": "Mild winters, far fewer crowds",
        },
        {
            "name": "Seville",
            "typical_price": "$2,000",
            "offseason_price": "$900",
            "savings_percent": 55,
            "why_visit": "Pleasant temperatures, cheap flights",
        },
    ],
}

FAKE_INSPIRE = {
    "destinations": [
        {
            "name": "Prague",
            "description": "Gothic spires and candlelit libraries",
            "hashtags": ["#DarkAcademia", "#Prague", "#Travel"],
            "photo_spots": ["Charles Bridge at dawn", "National Library reading room"],
        },
        {
            "name": "Edinburgh",
            "description": "Brooding castles and misty cobblestones",
            "hashtags": ["#Edinburgh", "#DarkAcademia", "#Scotland"],
            "photo_spots": ["Greyfriars Kirkyard", "Victoria Street"],
        },
        {
            "name": "Bruges",
            "description": "Medieval towers and candlelit canals",
            "hashtags": ["#Bruges", "#Medieval", "#DarkAcademia"],
            "photo_spots": ["Belfry Tower", "Begijnhof gardens"],
        },
    ]
}

FAKE_CAPTION = {
    "caption": "Golden hour over Santorini. Some places you visit once and carry forever.",
    "hashtags": ["#Santorini", "#Greece", "#Wanderlust", "#Romance"],
    "alt_captions": [
        "Where the sunsets write poetry. #Santorini",
        "Infinite blue meets infinite sky. #Greece",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeGeminiResponse:
    """Minimal stand-in for a Gemini GenerateContentResponse."""

    def __init__(self, payload: dict) -> None:
        self.text = json.dumps(payload)


def _model_mock(payload: dict) -> MagicMock:
    model = MagicMock()
    model.generate_content.return_value = _FakeGeminiResponse(payload)
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def mock_recommend(monkeypatch):
    model = _model_mock(FAKE_RECOMMEND)
    monkeypatch.setattr("backend.main._get_extract_model", lambda: model)
    return model


@pytest.fixture
def mock_offseason(monkeypatch):
    model = _model_mock(FAKE_OFFSEASON)
    monkeypatch.setattr("backend.main._get_extract_model", lambda: model)
    return model


@pytest.fixture
def mock_inspire(monkeypatch):
    model = _model_mock(FAKE_INSPIRE)
    monkeypatch.setattr("backend.main._get_extract_model", lambda: model)
    return model


@pytest.fixture
def mock_caption(monkeypatch):
    model = _model_mock(FAKE_CAPTION)
    monkeypatch.setattr("backend.main._get_extract_model", lambda: model)
    return model
