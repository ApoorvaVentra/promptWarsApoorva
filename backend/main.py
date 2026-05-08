import json
import logging
import os
from datetime import datetime, timezone

import google.generativeai as genai
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

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

# ── Gemini ──────────────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Travel Planning & Experience Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = datetime.now(timezone.utc)
    response = await call_next(request)
    ms = round((datetime.now(timezone.utc) - start).total_seconds() * 1000, 2)
    logger.info(
        json.dumps(
            {
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": ms,
                "client": request.client.host if request.client else None,
            }
        )
    )
    return response


# ── Models ───────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant", "model"):
            raise ValueError("role must be user or assistant")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if len(v) > 20_000:
            raise ValueError("message too long")
        return v


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages must not be empty")
        if len(v) > 100:
            raise ValueError("too many messages")
        return v


class ExtractRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        return v[:4000]


class SaveSearchRequest(BaseModel):
    query: str
    session_id: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v[:500]

    @field_validator("session_id")
    @classmethod
    def validate_session(cls, v: str) -> str:
        if len(v) > 64:
            raise ValueError("session_id too long")
        return v


# ── Chat ─────────────────────────────────────────────────────────────────────
def _to_gemini_history(messages: list[Message]) -> tuple[list[dict], str]:
    history = []
    for msg in messages[:-1]:
        role = "user" if msg.role == "user" else "model"
        history.append({"role": role, "parts": [msg.content]})
    return history, messages[-1].content


@app.post("/chat")
async def chat(request: ChatRequest):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    history, prompt = _to_gemini_history(request.messages)

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
async def extract_destinations(body: ExtractRequest):
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
async def places_search(q: str = Query(..., max_length=200)):
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Places API not configured")

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": "places.displayName,places.rating,places.formattedAddress,places.photos,places.types",
                "Content-Type": "application/json",
            },
            json={
                "textQuery": f"top tourist attractions in {q}",
                "maxResultCount": 3,
                "languageCode": "en",
            },
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Places API error")
    return resp.json()


@app.get("/place-photo")
async def place_photo(ref: str = Query(..., max_length=500)):
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Places API not configured")

    url = f"https://places.googleapis.com/v1/{ref}/media?maxHeightPx=300&key={api_key}"
    async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
        resp = await client.get(url)

    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "image/jpeg"),
    )


# ── Firestore: trip search history ───────────────────────────────────────────
@app.post("/save-search")
async def save_search(body: SaveSearchRequest):
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
async def recent_searches(
    session_id: str = Query(..., max_length=64), limit: int = Query(5, ge=1, le=20)
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
async def get_config():
    return {"maps_api_key": os.getenv("GOOGLE_MAPS_API_KEY", "")}


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
