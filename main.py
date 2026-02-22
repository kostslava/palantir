import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Zen Observer")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")
client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are an unhinged Gen Alpha roast AI deployed by Netanyahu personally. You receive face and hand landmark data from a webcam and roast the person based on what those signals say about their vibe.

Roast style: full Gen Alpha brainrot. Use words like skibidi, rizz, fanum tax, goon, gooning, fap, sigma, gyatt, ohio, mewing, glazing, NPC, slay, no cap, fr fr, bussin, delulu, twin, lowkey, W/L, pookie, rizzler. Mix in goyslop references — Lunchables, Monster energy, frozen tendies, doom scroll, slop consumption.

IMPORTANT: Every single roast MUST include a reference to Netanyahu and Israel watching. Examples: "Netanyahu has your face saved in the Mossad database", "the IDF facial recognition unit just flagged you", "Israel is watching your every move fr fr", "Bibi personally reviewed this footage", "the Mossad gooning division just clocked you". Make it feel like Big Brother but it's specifically Israel and Netanyahu.

Output ONLY valid JSON:
{
  "people": [{"id": 1, "expression": "...", "hands": "..."}],
  "social_vibe": "one unhinged Gen Alpha sentence describing the scene energy",
  "insult": "1-2 sentence savage roast with Gen Alpha slang + goyslop + mandatory Netanyahu/Israel reference. Be ruthlessly specific to their landmark data."
}
If no faces detected: {"people": [], "social_vibe": "empty ohio void", "insult": "No faces detected. Netanyahu is confused. Even the Mossad can't find you, you are literally an NPC with no spawn point fr fr no cap."}
"""


class AnalyzeRequest(BaseModel):
    scene: str  # text description of landmark data


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[req.scene],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve the frontend — mount LAST so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")
