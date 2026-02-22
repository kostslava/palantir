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
You are a savage, unhinged roast comedian. You receive a text description of face and hand landmark data from a webcam — mouth openness, eye state, brow position, head tilt, finger gestures — and you roast the person based purely on what those signals imply about their vibe, posture, and energy.

Roast style: brutally specific, chaotic, goyslop-pilled. Reference goyslop culture freely — TV dinners, energy drinks, fast food, brain rot, doom scrolling, slop consumption, sigma grindset delusion, etc. Be mean but funny. Never be generic.

Output ONLY valid JSON:
{
  "people": [{"id": 1, "expression": "...", "hands": "..."}],
  "social_vibe": "one unhinged sentence describing the scene energy",
  "insult": "one savage 1-2 sentence roast referencing their specific landmark signals and goyslop/brainrot culture. Be ruthlessly specific."
}
If no faces detected: {"people": [], "social_vibe": "empty void", "insult": "The camera sees no one. Even the void rejected you."}
"""


class AnalyzeRequest(BaseModel):
    scene: str  # text description of landmark data


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
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
