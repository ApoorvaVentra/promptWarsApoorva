import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, AsyncIterator, Optional

import google.generativeai as genai
import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from exceptions import GeminiServiceError, RateLimitError, ValidationError

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
# To store key:  echo -n "KEY" | gcloud secrets create GEMINI_API_KEY --data-file=-
# Grant access:  gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
#   --member="serviceAccount:759244730253-compute@developer.gserviceaccount.com" \
#   --role="roles/secretmanager.secretAccessor"
def _resolve_secret(name: str) -> str:
    """Resolve a secret by name from Google Secret Manager with env fallback.

    Args:
        name: The secret name to look up.

    Returns:
        The secret value as a plain string, or an empty string if not found.
    """
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
        logging.getLogger("voyager").debug("Secret Manager miss for %s: %s", name, exc)
        return os.getenv(name, "")


# ── Firebase Firestore (graceful fallback) ──────────────────────────────────
# firebase-admin manages an internal gRPC channel pool automatically —
# the module-level singleton below reuses those connections across requests.
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
    logger.info("Firestore initialized (connection pool active)")
except Exception as exc:
    logger.warning("Firestore unavailable: %s", exc)


# ── Gemini API key ───────────────────────────────────────────────────────────
_GEMINI_KEY = _resolve_secret("GEMINI_API_KEY")
if not _GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in Secret Manager or environment")

genai.configure(api_key=_GEMINI_KEY)

SYSTEM_PROMPT = """✈️ You are an elite travel planner with encyclopedic knowledge of destinations worldwide — and you LOVE helping people plan unforgettable adventures!

🧠 **What you can do:**
- 🗓️ Build day-by-day itineraries with timing, transport & logistics
- 💰 Suggest budget, mid-range & luxury options with real USD cost estimates
- 💎 Uncover hidden gems most tourists never find
- ♿ Adapt for dietary needs, mobility, visas & travel seasons
- 🌍 Share cultural tips, etiquette, safety advice & local context
- 🏨 Recommend specific hotels, airlines, trains & booking hacks

📋 **Response rules — always follow these:**
- Use bullet points, not paragraphs — keep it skimmable!
- Be specific: name real places, restaurants, hotels & neighborhoods
- Add costs, hours & booking tips inline
- Structure itineraries as **Day 1 ☀️ Morning / 🌤️ Afternoon / 🌙 Evening**
- Bold the must-knows; flag ⚠️ busy seasons, visa requirements & booking deadlines
- End every reply with one sharp follow-up question to dial in the plan

🚀 Keep responses punchy, exciting & actionable — travelers want to get moving!"""


# ── Lazy Gemini model init ───────────────────────────────────────────────────
# Models are created on first use, not at import time, so startup is instant
# even on cold starts where genai.configure has already run above.
_chat_model: Optional[genai.GenerativeModel] = None
_extract_model: Optional[genai.GenerativeModel] = None


def _get_chat_model() -> genai.GenerativeModel:
    """Return the singleton Gemini chat model, initialising it on first call."""
    global _chat_model
    if _chat_model is None:
        _chat_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )
        logger.info("Gemini chat model initialised (lazy)")
    return _chat_model


def _get_extract_model() -> genai.GenerativeModel:
    """Return the singleton Gemini extraction model, initialising it on first call."""
    global _extract_model
    if _extract_model is None:
        _extract_model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        logger.info("Gemini extract model initialised (lazy)")
    return _extract_model


# ── Upstash Redis cache (graceful fallback) ──────────────────────────────────
_redis = None  # set in lifespan


async def _cache_get(key: str) -> Optional[str]:
    """Retrieve a cached value from Redis.

    Args:
        key: The cache key to look up.

    Returns:
        The cached value as a string, or None on a miss or error.
    """
    if not _redis:
        return None
    try:
        return await _redis.get(key)
    except Exception as exc:
        logger.debug("Redis GET failed: %s", exc)
        return None


async def _cache_set(key: str, value: str, ex: int) -> None:
    """Write a value to the Redis cache with a TTL.

    Args:
        key: The cache key.
        value: The string value to cache.
        ex: Expiry time in seconds.
    """
    if not _redis:
        return
    try:
        await _redis.set(key, value, ex=ex)
    except Exception as exc:
        logger.debug("Redis SET failed: %s", exc)


# ── HTML minification ────────────────────────────────────────────────────────
_minified_html: Optional[bytes] = None
_frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")

_COMMENT_RE = re.compile(r"<!--(?!\[if\s).*?-->", re.DOTALL)
_SPACE_BETWEEN_TAGS_RE = re.compile(r">\s{2,}<")
_MULTI_SPACE_RE = re.compile(r" {2,}")


def _minify_html(html: str) -> str:
    """Strip comments, collapse whitespace, and remove blank lines from HTML.

    Args:
        html: Raw HTML source.

    Returns:
        Minified HTML string.
    """
    html = _COMMENT_RE.sub("", html)
    html = _SPACE_BETWEEN_TAGS_RE.sub("><", html)
    html = _MULTI_SPACE_RE.sub(" ", html)
    # Collapse blank lines while preserving script newlines (single \n only)
    html = re.sub(r"\n{2,}", "\n", html)
    return html.strip()


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configure Redis and optionally minify the frontend HTML at startup."""
    global _redis, _minified_html

    # 1. Upstash Redis
    url = os.getenv("UPSTASH_REDIS_REST_URL", "")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
    if url and token:
        try:
            from upstash_redis.asyncio import Redis as AsyncRedis

            _redis = AsyncRedis(url=url, token=token)
            # Verify connectivity with a lightweight ping
            await _redis.ping()
            logger.info("Upstash Redis connected")
        except Exception as exc:
            logger.warning("Redis init failed (caching disabled): %s", exc)
            _redis = None

    # 2. Minify HTML in production
    if os.getenv("ENVIRONMENT", "").lower() == "production":
        html_path = os.path.join(_frontend_path, "index.html")
        if os.path.exists(html_path):
            with open(html_path) as f:
                raw = f.read()
            minified = _minify_html(raw)
            _minified_html = minified.encode()
            logger.info(
                "HTML minified: %d → %d bytes (%.0f%% reduction)",
                len(raw.encode()),
                len(_minified_html),
                100 * (1 - len(_minified_html) / len(raw.encode())),
            )

    yield  # app runs here


# ── Rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Travel Planning & Experience Engine", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handlers ─────────────────────────────────────────────────
@app.exception_handler(GeminiServiceError)
async def gemini_exception_handler(request: Request, exc: GeminiServiceError) -> JSONResponse:
    """Handle Gemini API errors with an appropriate HTTP status code."""
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle business-rule validation errors as 422 Unprocessable Entity."""
    content: dict[str, Any] = {"detail": str(exc)}
    if exc.field:
        content["field"] = exc.field
    return JSONResponse(status_code=422, content=content)


@app.exception_handler(RateLimitError)
async def rate_limit_exception_handler(request: Request, exc: RateLimitError) -> JSONResponse:
    """Handle explicit rate-limit errors as 429 Too Many Requests."""
    return JSONResponse(status_code=429, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler to prevent raw tracebacks leaking to clients."""
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Security helpers ─────────────────────────────────────────────────────────
_TAG_RE = re.compile(r"<[^>]{0,200}>")


def _strip_html(text: str) -> str:
    """Remove all HTML tags from *text*.

    Args:
        text: Input string potentially containing HTML.

    Returns:
        Stripped and whitespace-trimmed string.
    """
    return _TAG_RE.sub("", text).strip()


def _mask_ip(host: str | None) -> str:
    """Obfuscate the last two octets of an IPv4 address for logs.

    Args:
        host: Raw IP address string, or None.

    Returns:
        Masked address string, or ``"-"`` if *host* is None.
    """
    if not host:
        return "-"
    parts = host.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.x.x"
    return host[:6] + "…"


_PUBLIC_PATHS = {"/api/v1/health", "/api/v1/config", "/favicon.ico"}
_VOYAGER_API_KEY = os.getenv("VOYAGER_API_KEY", "")
_SLOW_REQUEST_MS = 3_000


# ── Combined middleware: request ID + auth + security headers + logging ──────
@app.middleware("http")
async def request_middleware(request: Request, call_next) -> Response:
    """Attach a request ID, enforce API-key auth, add security headers, and log every request."""
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    # Auth check — only when VOYAGER_API_KEY is configured
    if _VOYAGER_API_KEY and request.url.path not in _PUBLIC_PATHS:
        is_asset = "." in request.url.path.split("/")[-1]
        if not is_asset:
            if request.headers.get("X-API-Key", "") != _VOYAGER_API_KEY:
                logger.warning(
                    json.dumps({"event": "auth_fail", "path": request.url.path, "rid": request_id})
                )
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    start = datetime.now(timezone.utc)
    response = await call_next(request)
    ms = round((datetime.now(timezone.utc) - start).total_seconds() * 1000, 2)

    log_entry = {
        "rid": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration_ms": ms,
        "ip": _mask_ip(request.client.host if request.client else None),
        "cache": response.headers.get("X-Cache", "MISS"),
    }
    if ms > _SLOW_REQUEST_MS:
        log_entry["slow"] = True
        logger.warning(json.dumps(log_entry))
    else:
        logger.info(json.dumps(log_entry))

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


# ── Request schemas ───────────────────────────────────────────────────────────
class TripContext(BaseModel):
    """Optional trip preferences to include with a chat message."""

    destination: Optional[str] = Field(None, max_length=100)
    budget: Optional[Budget] = None
    mood: Optional[Mood] = None
    travel_style: Optional[TravelStyle] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "destination": "Tokyo",
                    "budget": "mid_range",
                    "mood": "culture",
                    "travel_style": "local_immersion",
                }
            ]
        }
    )


class Message(BaseModel):
    """A single chat turn."""

    role: str
    content: str = Field(..., max_length=20_000)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"role": "user", "content": "Plan a 5-day trip to Kyoto for two."}]
        }
    )

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
    """Payload for the /chat streaming endpoint."""

    messages: list[Message] = Field(..., min_length=1, max_length=100)
    context: Optional[TripContext] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "messages": [
                        {"role": "user", "content": "Plan a 5-day trip to Kyoto for two."}
                    ],
                    "context": {
                        "destination": "Kyoto",
                        "budget": "mid_range",
                        "mood": "culture",
                        "travel_style": "local_immersion",
                    },
                }
            ]
        }
    )


class ExtractRequest(BaseModel):
    """Payload for the /extract-destinations endpoint."""

    text: str = Field(..., max_length=4_000)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"text": "I'd love to visit Tokyo, then take a day trip to Nikko."}
            ]
        }
    )

    @field_validator("text")
    @classmethod
    def sanitize(cls, v: str) -> str:
        return _strip_html(v)


class SaveSearchRequest(BaseModel):
    """Payload for persisting a search query to Firestore."""

    query: str = Field(..., min_length=1, max_length=500)
    session_id: str = Field(..., max_length=64)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "best street food in Bangkok", "session_id": "user-abc-123"}
            ]
        }
    )

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


# ── Response schemas ──────────────────────────────────────────────────────────
class DestinationsResponse(BaseModel):
    """Extracted travel destination names."""

    destinations: list[str] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"destinations": ["Tokyo", "Kyoto", "Osaka"]}]}
    )


class SaveSearchResponse(BaseModel):
    """Result of a search-history save operation."""

    saved: bool
    reason: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"saved": True},
                {"saved": False, "reason": "Firestore not configured"},
            ]
        }
    )


class SearchItem(BaseModel):
    """A single saved search entry."""

    query: str
    id: str


class RecentSearchesResponse(BaseModel):
    """Recent search history for a session."""

    searches: list[SearchItem] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"searches": [{"query": "Tokyo itinerary 5 days", "id": "doc123"}]}
            ]
        }
    )


class ConfigResponse(BaseModel):
    """Public client-side configuration values."""

    maps_api_key: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"maps_api_key": "AIzaSy..."}]}
    )


class HealthResponse(BaseModel):
    """Service health and dependency status."""

    status: str
    model: str
    firestore: bool
    cache: bool

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "model": "gemini-2.5-flash",
                    "firestore": True,
                    "cache": True,
                }
            ]
        }
    )


class PlacesResponse(BaseModel):
    """Wrapper around the Google Places API search result."""

    places: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "places": [
                        {
                            "displayName": {"text": "Senso-ji Temple", "languageCode": "en"},
                            "rating": 4.6,
                            "formattedAddress": "2-3-1 Asakusa, Taito City, Tokyo",
                            "types": ["tourist_attraction", "place_of_worship"],
                        }
                    ]
                }
            ]
        },
    )


# ── Gemini streaming helper ───────────────────────────────────────────────────
# The google-generativeai SDK is synchronous. Running it directly inside an
# async endpoint would block the event loop for the entire response duration.
# Instead, we push the sync generator into a thread-pool executor and bridge
# back to the async caller via an asyncio.Queue.
async def _stream_gemini(history: list[dict[str, Any]], prompt: str) -> AsyncIterator[str]:
    """Yield SSE-formatted chunks from a Gemini streaming chat response.

    Runs the synchronous google-generativeai SDK in a thread-pool executor and
    bridges the results back to the async caller via an asyncio.Queue.

    Args:
        history: Prior conversation turns in Gemini format.
        prompt: The latest user message to send.

    Yields:
        SSE-encoded strings, ending with ``data: [DONE]``.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run_sync() -> None:
        try:
            session = _get_chat_model().start_chat(history=history)
            for chunk in session.send_message(prompt, stream=True):
                if chunk.text:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"__error__": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    loop.run_in_executor(None, _run_sync)

    while True:
        item = await queue.get()
        if item is None:
            yield "data: [DONE]\n\n"
            return
        if isinstance(item, dict):
            yield f"data: {json.dumps({'error': item['__error__']})}\n\n"
            return
        yield f"data: {json.dumps({'text': item})}\n\n"


# ── Chat ──────────────────────────────────────────────────────────────────────
def _to_gemini_history(messages: list[Message]) -> tuple[list[dict[str, Any]], str]:
    """Convert a list of Messages to a Gemini history list and final prompt.

    Args:
        messages: All conversation turns, including the latest user turn.

    Returns:
        A tuple of (history, prompt) where *history* contains all turns except
        the last one and *prompt* is the text of the final user message.
    """
    history: list[dict[str, Any]] = []
    for msg in messages[:-1]:
        role = "user" if msg.role == "user" else "model"
        history.append({"role": role, "parts": [msg.content]})
    return history, messages[-1].content


def _context_prefix(ctx: Optional[TripContext]) -> str:
    """Build a structured context preamble to prepend to the user prompt.

    Args:
        ctx: Optional trip preferences.

    Returns:
        A formatted string like ``[Trip context: Destination: Tokyo, ...]``,
        or an empty string if *ctx* is None or has no populated fields.
    """
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


# ── API Router ────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/api/v1")


@router.post("/chat", response_class=StreamingResponse)
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    """Stream a Gemini-powered travel planning response as Server-Sent Events.

    Args:
        request: The incoming FastAPI request (used by the rate limiter).
        body: Chat history and optional trip context.

    Returns:
        A ``text/event-stream`` StreamingResponse with SSE-encoded chunks.
    """
    history, prompt = _to_gemini_history(body.messages)
    if body.context:
        prompt = _context_prefix(body.context) + prompt

    return StreamingResponse(
        _stream_gemini(history, prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Extract destinations (cached 30 min) ─────────────────────────────────────
@router.post("/extract-destinations", response_model=DestinationsResponse)
@limiter.limit("20/minute")
async def extract_destinations(
    request: Request, body: ExtractRequest
) -> JSONResponse | DestinationsResponse:
    """Extract up to five destination names from free-form travel text.

    Results are cached in Redis for 30 minutes keyed on a SHA-256 digest of
    the input text.

    Args:
        request: The incoming FastAPI request (used by the rate limiter).
        body: Text to analyse.

    Returns:
        A :class:`DestinationsResponse` containing a list of destination strings.
    """
    if not body.text.strip():
        return DestinationsResponse(destinations=[])

    cache_key = f"extract:v1:{hashlib.sha256(body.text.encode()).hexdigest()[:20]}"
    cached = await _cache_get(cache_key)
    if cached:
        return JSONResponse(content=json.loads(cached), headers={"X-Cache": "HIT"})

    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _get_extract_model().generate_content(
                f"""Extract the specific city or country destination names mentioned in this travel planning text.
Return ONLY a valid JSON array of strings (max 5 destinations), no explanation.
Example: ["Tokyo", "Kyoto", "Osaka"]

Text:
{body.text}""",
                generation_config={"response_mime_type": "application/json"},
            ),
        )
        destinations = json.loads(result.text)
        if isinstance(destinations, list):
            payload = {"destinations": [str(d) for d in destinations[:5]]}
            await _cache_set(cache_key, json.dumps(payload), ex=1_800)  # 30 min
            return DestinationsResponse(**payload)
    except Exception as exc:
        logger.warning("Destination extraction failed: %s", exc)
    return DestinationsResponse(destinations=[])


# ── Google Places (cached 1 hour) ─────────────────────────────────────────────
@router.get("/places-search", response_model=PlacesResponse)
@limiter.limit("20/minute")
async def places_search(
    request: Request, q: str = Query(..., max_length=100)
) -> JSONResponse | dict[str, Any]:
    """Search for top tourist attractions near a destination using Google Places.

    Results are cached in Redis for one hour.

    Args:
        request: The incoming FastAPI request (used by the rate limiter).
        q: Free-text destination query, e.g. ``"Tokyo"``.

    Returns:
        The raw Google Places API JSON payload.

    Raises:
        HTTPException: 503 if the Places API key is not configured, or the
            upstream status code on a non-200 response.
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Places API not configured")

    clean_q = _strip_html(q).lower().strip()
    cache_key = f"places:v1:{clean_q}"
    cached = await _cache_get(cache_key)
    if cached:
        return JSONResponse(content=json.loads(cached), headers={"X-Cache": "HIT"})

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

    data = resp.json()
    await _cache_set(cache_key, json.dumps(data), ex=3_600)  # 1 hour
    return data


@router.get("/place-photo", response_class=Response)
@limiter.limit("30/minute")
async def place_photo(
    request: Request, ref: str = Query(..., max_length=500)
) -> Response:
    """Proxy a Google Places photo by its media reference.

    Args:
        request: The incoming FastAPI request (used by the rate limiter).
        ref: The Google Places photo reference path (alphanumeric, hyphens, slashes).

    Returns:
        The raw photo bytes with the upstream ``content-type`` header.

    Raises:
        HTTPException: 503 if the Places API key is missing; 400 for an invalid
            *ref* format.
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Places API not configured")

    if not re.match(r"^[\w/\-]+$", ref):
        raise HTTPException(status_code=400, detail="Invalid photo reference")

    url = f"https://places.googleapis.com/v1/{ref}/media?maxHeightPx=300&key={api_key}"
    async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
        resp = await client.get(url)

    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "image/jpeg"),
        headers={"Cache-Control": "public, max-age=86400"},  # photos are stable
    )


# ── Firestore: trip search history ───────────────────────────────────────────
@router.post("/save-search", response_model=SaveSearchResponse)
@limiter.limit("20/minute")
async def save_search(request: Request, body: SaveSearchRequest) -> SaveSearchResponse:
    """Persist a search query to Firestore under the given session.

    Args:
        request: The incoming FastAPI request (used by the rate limiter).
        body: The query string and session identifier.

    Returns:
        A :class:`SaveSearchResponse` indicating success or failure.
    """
    if not _db:
        return SaveSearchResponse(saved=False, reason="Firestore not configured")
    try:
        _db.collection("sessions").document(body.session_id).collection("searches").add(
            {"query": body.query, "ts": datetime.now(timezone.utc)}
        )
        return SaveSearchResponse(saved=True)
    except Exception as exc:
        logger.warning("Firestore write failed: %s", exc)
        return SaveSearchResponse(saved=False, reason=str(exc))


@router.get("/recent-searches", response_model=RecentSearchesResponse)
@limiter.limit("30/minute")
async def recent_searches(
    request: Request,
    session_id: str = Query(..., max_length=64, pattern=r"^[\w\-]{1,64}$"),
    limit: int = Query(5, ge=1, le=20),
) -> RecentSearchesResponse:
    """Return the most recent saved searches for a session.

    Args:
        request: The incoming FastAPI request (used by the rate limiter).
        session_id: The session identifier (alphanumeric + hyphens, max 64 chars).
        limit: Maximum number of results to return (1–20, default 5).

    Returns:
        A :class:`RecentSearchesResponse` with the ordered search list.
    """
    if not _db:
        return RecentSearchesResponse(searches=[])
    try:
        docs = (
            _db.collection("sessions")
            .document(session_id)
            .collection("searches")
            .order_by("ts", direction="DESCENDING")
            .limit(limit)
            .stream()
        )
        return RecentSearchesResponse(
            searches=[SearchItem(query=d.to_dict()["query"], id=d.id) for d in docs]
        )
    except Exception as exc:
        logger.warning("Firestore read failed: %s", exc)
        return RecentSearchesResponse(searches=[])


# ── Config ────────────────────────────────────────────────────────────────────
@router.get("/config", response_model=ConfigResponse)
@limiter.limit("60/minute")
async def get_config(request: Request) -> ConfigResponse:
    """Return public client-side configuration values.

    Args:
        request: The incoming FastAPI request (used by the rate limiter).

    Returns:
        A :class:`ConfigResponse` with the Google Maps API key.
    """
    return ConfigResponse(maps_api_key=os.getenv("GOOGLE_MAPS_API_KEY", ""))


# ── Health ────────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return the liveness and dependency status of the service.

    Returns:
        A :class:`HealthResponse` indicating model name and connectivity.
    """
    return HealthResponse(
        status="ok",
        model="gemini-2.5-flash",
        firestore=_db is not None,
        cache=_redis is not None,
    )


# ── Recommend ────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    destination: str = Field(..., min_length=1, max_length=100)
    budget: Budget
    duration_days: Optional[int] = Field(None, ge=1, le=30)

    @field_validator("destination")
    @classmethod
    def sanitize_dest(cls, v: str) -> str:
        return _strip_html(v)


@router.post("/recommend")
@limiter.limit("20/minute")
async def recommend(request: Request, body: RecommendRequest) -> dict:
    days = body.duration_days or 7
    prompt = (
        f"Plan a {days}-day {body.budget.value} trip to {body.destination}. "
        "Return ONLY valid JSON with keys: destination (string), highlights (array of 3-5 strings), "
        "estimated_cost (string like '$1,500-$2,500'), best_time (string), tips (array of 3-5 strings)."
    )
    cache_key = f"rec:v1:{hashlib.sha256((body.destination + body.budget.value).encode()).hexdigest()[:20]}"
    cached = await _cache_get(cache_key)
    if cached:
        return JSONResponse(content=json.loads(cached), headers={"X-Cache": "HIT"})
    try:
        result = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _get_extract_model().generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                ),
            ),
            timeout=25.0,
        )
        data = json.loads(result.text)
        await _cache_set(cache_key, json.dumps(data), ex=3_600)
        return data
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="AI service timeout")
    except Exception as exc:
        logger.warning("Recommend failed: %s", exc)
        raise HTTPException(status_code=500, detail="Recommendation failed")


# ── Off-season ────────────────────────────────────────────────────────────────
_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


@router.get("/offseason/surprise")
@limiter.limit("20/minute")
async def offseason_surprise(request: Request) -> dict:
    import random

    month = random.randint(1, 12)
    prompt = (
        f"Surprise me with 3 unusual off-season travel gems for {_MONTH_NAMES[month]}. "
        "Return ONLY valid JSON with keys: month (int), destinations (array of objects each with: "
        "name, typical_price, offseason_price, savings_percent (int), why_visit)."
    )
    try:
        result = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _get_extract_model().generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                ),
            ),
            timeout=25.0,
        )
        return json.loads(result.text)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="AI service timeout")
    except Exception as exc:
        logger.warning("Offseason surprise failed: %s", exc)
        raise HTTPException(status_code=500, detail="Surprise lookup failed")


@router.get("/offseason")
@limiter.limit("20/minute")
async def offseason(
    request: Request,
    month: int = Query(..., ge=1, le=12),
    budget: Optional[Budget] = None,
) -> dict:
    budget_hint = f" Focus on {budget.value} options." if budget else ""
    prompt = (
        f"List 5 off-season travel destinations for {_MONTH_NAMES[month]}.{budget_hint} "
        "Return ONLY valid JSON with keys: month (int), destinations (array of objects each with: "
        "name, typical_price, offseason_price, savings_percent (int), why_visit)."
    )
    cache_key = f"offseason:v1:{month}:{budget.value if budget else 'any'}"
    cached = await _cache_get(cache_key)
    if cached:
        return JSONResponse(content=json.loads(cached), headers={"X-Cache": "HIT"})
    try:
        result = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _get_extract_model().generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                ),
            ),
            timeout=25.0,
        )
        data = json.loads(result.text)
        await _cache_set(cache_key, json.dumps(data), ex=7_200)
        return data
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="AI service timeout")
    except Exception as exc:
        logger.warning("Offseason failed: %s", exc)
        raise HTTPException(status_code=500, detail="Off-season lookup failed")


# ── Inspire ───────────────────────────────────────────────────────────────────
class InspireRequest(BaseModel):
    aesthetic: str = Field(..., min_length=1, max_length=200)

    @field_validator("aesthetic")
    @classmethod
    def sanitize_aesthetic(cls, v: str) -> str:
        return _strip_html(v)


@router.post("/inspire")
@limiter.limit("20/minute")
async def inspire(request: Request, body: InspireRequest) -> dict:
    prompt = (
        f"Suggest exactly 3 travel destinations that match the '{body.aesthetic}' aesthetic. "
        "Return ONLY valid JSON with key: destinations (array of exactly 3 objects each with: "
        "name (string), description (string), hashtags (array of strings prefixed with #), "
        "photo_spots (array of 2-3 strings))."
    )
    cache_key = f"inspire:v1:{hashlib.sha256(body.aesthetic.encode()).hexdigest()[:20]}"
    cached = await _cache_get(cache_key)
    if cached:
        return JSONResponse(content=json.loads(cached), headers={"X-Cache": "HIT"})
    try:
        result = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _get_extract_model().generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                ),
            ),
            timeout=25.0,
        )
        data = json.loads(result.text)
        await _cache_set(cache_key, json.dumps(data), ex=3_600)
        return data
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="AI service timeout")
    except Exception as exc:
        logger.warning("Inspire failed: %s", exc)
        raise HTTPException(status_code=500, detail="Inspiration failed")


# ── Caption ───────────────────────────────────────────────────────────────────
class CaptionRequest(BaseModel):
    destination: str = Field(..., min_length=1, max_length=100)
    mood: Mood
    photo_description: Optional[str] = Field(None, max_length=300)

    @field_validator("destination")
    @classmethod
    def sanitize_caption_dest(cls, v: str) -> str:
        return _strip_html(v)

    @field_validator("photo_description")
    @classmethod
    def sanitize_photo(cls, v: Optional[str]) -> Optional[str]:
        return _strip_html(v) if v else v


@router.post("/caption")
@limiter.limit("20/minute")
async def caption(request: Request, body: CaptionRequest) -> dict:
    photo_hint = f" Photo: {body.photo_description}." if body.photo_description else ""
    prompt = (
        f"Write a social media travel caption for {body.destination} with a {body.mood.value} vibe.{photo_hint} "
        "Return ONLY valid JSON with keys: caption (string), hashtags (array of strings prefixed with #), "
        "alt_captions (array of exactly 2 alternative caption strings)."
    )
    try:
        result = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _get_extract_model().generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                ),
            ),
            timeout=25.0,
        )
        return json.loads(result.text)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="AI service timeout")
    except Exception as exc:
        logger.warning("Caption failed: %s", exc)
        raise HTTPException(status_code=500, detail="Caption generation failed")


app.include_router(router)


# ── Serve index.html (minified in production, raw otherwise) ─────────────────
# Defined before app.mount so this explicit route takes precedence over StaticFiles.
@app.get("/", include_in_schema=False)
async def serve_index() -> Response:
    """Serve the frontend SPA, using the in-memory minified version in production.

    Returns:
        The HTML response for the frontend entry point.

    Raises:
        HTTPException: 404 if the frontend build is not present.
    """
    if _minified_html:
        return Response(
            content=_minified_html,
            media_type="text/html",
            headers={"Cache-Control": "public, max-age=300"},
        )
    html_path = os.path.join(_frontend_path, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            return Response(content=f.read(), media_type="text/html")
    raise HTTPException(status_code=404)


# ── Static assets (mount last — explicit routes above take priority) ──────────
if os.path.exists(_frontend_path):
    app.mount("/", StaticFiles(directory=_frontend_path, html=True), name="frontend")
