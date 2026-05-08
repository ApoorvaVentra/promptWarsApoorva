import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import google.generativeai as genai
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

# ── Google Cloud Logging (graceful fallback to stdlib) ──────────────────────
try:
    import google.cloud.logging as cloud_logging

    _cloud_client = cloud_logging.Client()
    _cloud_client.setup_logging()
    logger = logging.getLogger("voyager")
    logger.info("Google Cloud Logging active")
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("voyager")
    logger.info("Standard logging active (Cloud Logging unavailable)")


# ── Google Secret Manager ────────────────────────────────────────────────────
# To store Gemini key in Secret Manager:
#   echo -n "YOUR_KEY" | gcloud secrets create GEMINI_API_KEY --data-file=- \
#     --project=promptwarsapoorva
#   gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
#     --member="serviceAccount:759244730253-compute@developer.gserviceaccount.com" \
#     --role="roles/secretmanager.secretAccessor" --project=promptwarsapoorva
def _resolve_secret(name: str) -> str:
    project = os.getenv("GCP_PROJECT", "promptwarsapoorva")
    try:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()
        resource = f"projects/{project}/secrets/{name}/versions/latest"
        return (
            client.access_secret_version(request={"name": resource})
            .payload.data.decode()
            .strip()
        )
    except Exception as exc:
        logging.getLogger("voyager").debug(
            "Secret Manager miss for %s: %s", name, exc
        )
        return os.getenv(name, "")


# ── Firebase Firestore (graceful fallback) ──────────────────────────────────
_db = None
try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
        cred = (
            credentials.Certificate(cred_path)
            if cred_path and os.path.exists(cred_path)
            else credentials.ApplicationDefault()
        )
        firebase_admin.initialize_app(cred)
    _db = firestore.client()
    logger.info("Firestore initialized")
except Exception as exc:
    logger.warning("Firestore unavailable: %s", exc)


# ── Gemini (key from Secret Manager with env-var fallback) ──────────────────
_GEMINI_KEY = _resolve_secret("GEMINI_API_KEY")
if not _GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in Secret Manager or environment")

genai.configure(api_key=_GEMINI_KEY)

SYSTEM_PROMPT = """You are an expert travel planner and experience curator with encyclopedic knowledge of destinations worldwide. You help travelers plan extraordinary trips that perfectly balance their preferences, constraints, and budget.

Your capabilities:
- **Dynamic Itinerary Building**: Create detailed day-by-day plans with timing, transport, and logistics
- **Budget Optimization**: Suggest options across luxury, mid-range, and budget tiers with real cost estimates in USD
- **Hidden Gem Curation**: Blend iconic landmarks with local secrets most tourists never find
- **Constraint Handling**: Adapt plans for dietary restrictions, mobility needs, visa requirements, travel seasons
- **Cultural Intelligence**: Share etiquette tips, local customs, safety advice, and cultural context
- **Logistics Mastery**: Recommend specific hotels, airlines, trains, local transit, and booking strategies

Response style:
- Be specific — name actual places, restaurants, hotels, neighborhoods
- Include practical details: approximate costs, opening hours, booking tips
- Structure itineraries with Day 1, Day 2 headers and morning/afternoon/evening blocks
- Use **bold** for key highlights and bullet points for options
- Proactively flag busy seasons, booking lead times, visa requirements
- End every response with a question to refine the plan or offer a follow-up

You are enthusiastic, knowledgeable, and genuinely excited to help people create unforgettable travel memories."""

_chat_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT,
)
_extract_model = genai.GenerativeModel(model_name="gemini-2.5-flash")

# ── Rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Travel Planning & Experience Engine")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Security helpers ─────────────────────────────────────────────────────────
_TAG_RE = re.compile(r"<[^>]{0,200}>")  # bounded to prevent ReDoS


def _strip_html(text: str) -> str:
    return _TAG_RE.sub("", text).strip()


def _mask_ip(host: str | None) -> str:
    if not host:
        return "-"
    parts = host.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.x.x"
    return host[:6] + "…"


_PUBLIC_PATHS = {"/health", "/config", "/favicon.ico"}
_VOYAGER_API_KEY = os.getenv("VOYAGER_API_KEY", "")


# ── Combined middleware: request ID + auth + security headers + logging ──────
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    # Auth check — only when VOYAGER_API_KEY is configured
    if _VOYAGER_API_KEY and request.url.path not in _PUBLIC_PATHS:
        # Skip auth for static assets (anything with a file extension)
        is_asset = "." in request.url.path.split("/")[-1]
        if not is_asset:
            client_key = request.headers.get("X-API-Key", "")
            if client_key != _VOYAGER_API_KEY:
                logger.warning(
                    json.dumps(
                        {"event": "auth_fail", "path": request.url.path, "rid": request_id}
                    )
                )
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    start = datetime.now(timezone.utc)
    response = await call_next(request)
    ms = round((datetime.now(timezone.utc) - start).total_seconds() * 1000, 2)

    # Structured log — no PII (IP is masked)
    logger.info(
        json.dumps(
            {
                "rid": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": ms,
                "ip": _mask_ip(request.client.host if request.client else None),
            }
        )
    )

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Request-ID"] = request_id
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://maps.googleapis.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self' https://maps.googleapis.com https://maps.gstatic.com; "
        "frame-ancestors 'none';"
    )
    return response


# ── Enums ────────────────────────────────────────────────────────────────────
class Budget(str, Enum):
    budget = "budget"
    mid_range = "mid_range"
    luxury = "luxury"


class Mood(str, Enum):
    adventure = "adventure"
    relaxation = "relaxation"
    culture = "culture"
    food = "food"
    family = "family"
    romance = "romance"


class TravelStyle(str, Enum):
    backpacker = "backpacker"
    comfort = "comfort"
    luxury = "luxury"
    local_immersion = "local_immersion"


# ── Pydantic models ───────────────────────────────────────────────────────────
class TripContext(BaseModel):
    destination: Optional[str] = Field(None, max_length=100)
    budget: Optional[Budget] = None
    mood: Optional[Mood] = None
    travel_style: Optional[TravelStyle] = None


class Message(BaseModel):
    role: str
    content: str = Field(..., max_length=20_000)

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant", "model"):
            raise ValueError("role must be user or assistant")
        return v

    @field_validator("content")
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        return _strip_html(v)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=100)
    context: Optional[TripContext] = None


class ExtractRequest(BaseModel):
    text: str = Field(..., max_length=4_000)

    @field_validator("text")
    @classmethod
    def sanitize(cls, v: str) -> str:
        return _strip_html(v)


class SaveSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    session_id: str = Field(..., max_length=64)

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return _strip_html(v)

    @field_validator("session_id")
    @classmethod
    def validate_session(cls, v: str) -> str:
        if not re.match(r"^[\w\-]{1,64}$", v):
            raise ValueError("invalid session_id format")
        return v


# ── Chat ──────────────────────────────────────────────────────────────────────
def _to_gemini_history(messages: list[Message]) -> tuple[list[dict], str]:
    history = []
    for msg in messages[:-1]:
        role = "user" if msg.role == "user" else "model"
        history.append({"role": role, "parts": [msg.content]})
    return history, messages[-1].content


def _context_prefix(ctx: Optional[TripContext]) -> str:
    if not ctx:
        return ""
    parts = []
    if ctx.destination:
        parts.append(f"Destination: {ctx.destination}")
    if ctx.budget:
        parts.append(f"Budget tier: {ctx.budget.value}")
    if ctx.mood:
        parts.append(f"Trip mood: {ctx.mood.value}")
    if ctx.travel_style:
        parts.append(f"Travel style: {ctx.travel_style.value}")
    return "[Trip context: " + ", ".join(parts) + "]\n\n" if parts else ""


@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    history, prompt = _to_gemini_history(body.messages)
    if body.context:
        prompt = _context_prefix(body.context) + prompt

    def generate():
        try:
            session = _chat_model.start_chat(history=history)
            for chunk in session.send_message(prompt, stream=True):
                if chunk.text:
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Extract destinations ──────────────────────────────────────────────────────
@app.post("/extract-destinations")
@limiter.limit("20/minute")
async def extract_destinations(request: Request, body: ExtractRequest):
    if not body.text.strip():
        return {"destinations": []}
    try:
        result = _extract_model.generate_content(
            f"""Extract the specific city or country destination names mentioned in this travel planning text.
Return ONLY a valid JSON array of strings (max 5 destinations), no explanation.
Example: ["Tokyo", "Kyoto", "Osaka"]

Text:
{body.text}""",
            generation_config={"response_mime_type": "application/json"},
        )
        destinations = json.loads(result.text)
        if isinstance(destinations, list):
            return {"destinations": [str(d) for d in destinations[:5]]}
    except Exception as exc:
        logger.warning("Destination extraction failed: %s", exc)
    return {"destinations": []}


# ── Google Places (New API) ───────────────────────────────────────────────────
@app.get("/places-search")
@limiter.limit("20/minute")
async def places_search(request: Request, q: str = Query(..., max_length=100)):
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Places API not configured")

    clean_q = _strip_html(q)
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": "places.displayName,places.rating,places.formattedAddress,places.photos,places.types",
                "Content-Type": "application/json",
            },
            json={
                "textQuery": f"top tourist attractions in {clean_q}",
                "maxResultCount": 3,
                "languageCode": "en",
            },
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Places API error")
    return resp.json()


@app.get("/place-photo")
@limiter.limit("30/minute")
async def place_photo(request: Request, ref: str = Query(..., max_length=500)):
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Places API not configured")

    # Validate ref only contains safe path characters
    if not re.match(r"^[\w/\-]+$", ref):
        raise HTTPException(status_code=400, detail="Invalid photo reference")

    url = f"https://places.googleapis.com/v1/{ref}/media?maxHeightPx=300&key={api_key}"
    async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
        resp = await client.get(url)

    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "image/jpeg"),
    )


# ── Firestore: trip search history ───────────────────────────────────────────
@app.post("/save-search")
@limiter.limit("20/minute")
async def save_search(request: Request, body: SaveSearchRequest):
    if not _db:
        return {"saved": False, "reason": "Firestore not configured"}
    try:
        _db.collection("sessions").document(body.session_id).collection("searches").add(
            {"query": body.query, "ts": datetime.now(timezone.utc)}
        )
        return {"saved": True}
    except Exception as exc:
        logger.warning("Firestore write failed: %s", exc)
        return {"saved": False, "reason": str(exc)}


@app.get("/recent-searches")
@limiter.limit("30/minute")
async def recent_searches(
    request: Request,
    session_id: str = Query(..., max_length=64, pattern=r"^[\w\-]{1,64}$"),
    limit: int = Query(5, ge=1, le=20),
):
    if not _db:
        return {"searches": []}
    try:
        docs = (
            _db.collection("sessions")
            .document(session_id)
            .collection("searches")
            .order_by("ts", direction="DESCENDING")
            .limit(limit)
            .stream()
        )
        return {"searches": [{"query": d.to_dict()["query"], "id": d.id} for d in docs]}
    except Exception as exc:
        logger.warning("Firestore read failed: %s", exc)
        return {"searches": []}


# ── Config (public-safe keys for frontend) ────────────────────────────────────
@app.get("/config")
@limiter.limit("60/minute")
async def get_config(request: Request):
    return {
        "maps_api_key": os.getenv("GOOGLE_MAPS_API_KEY", ""),
        # included so the frontend can authenticate subsequent API calls
        "api_key": _VOYAGER_API_KEY,
    }


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "gemini-2.5-flash",
        "firestore": _db is not None,
    }


# ── Static frontend (mount last) ──────────────────────────────────────────────
_frontend = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(_frontend):
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
