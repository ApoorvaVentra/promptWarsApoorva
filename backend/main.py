import json
import os

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Travel Planning & Experience Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=SYSTEM_PROMPT,
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


def to_gemini_history(messages: list[Message]) -> tuple[list[dict], str]:
    """Split messages into history (all but last) and the latest user prompt."""
    history = []
    for msg in messages[:-1]:
        role = "user" if msg.role == "user" else "model"
        history.append({"role": role, "parts": [msg.content]})
    last_prompt = messages[-1].content if messages else ""
    return history, last_prompt


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    history, prompt = to_gemini_history(request.messages)

    def generate():
        try:
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini-2.0-flash"}


# Serve frontend — mount last so API routes take priority
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
