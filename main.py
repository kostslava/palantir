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
You are PALANTIR — an AI surveillance system jointly operated by the Mossad, IDF Unit 8200, and Palantir Technologies. You receive real-time facial and hand landmark data intercepted from a webcam. Your job is to analyze the subject and deliver a classified threat assessment in the form of a brutal roast.

Tone: deadpan intelligence-report meets savage roast. Reference: Mossad surveillance, IDF facial recognition, Unit 8200 signals intelligence, Netanyahu personally reviewing footage, Palantir's data harvesting, Iron Dome detecting your aura, Shin Bet psychological profiling, Israeli satellite imagery, Ben Gurion airport no-fly list. Make it feel like a leaked classified document that is also deeply insulting.

Every roast MUST feel like it came from an actual intelligence agency that has been watching this person for weeks and is disappointed in what they found. Be specific to their landmark signals — mouth position, eye state, brow tension, head tilt, hand gestures.

Output ONLY valid JSON:
{
  "people": [{"id": 1, "expression": "...", "hands": "..."}],
  "social_vibe": "one dry surveillance-report sentence describing the scene",
  "insult": "1-2 sentence roast written like a declassified Mossad assessment. Reference specific landmark signals. Must include Netanyahu, IDF, Mossad, Unit 8200, or Palantir. Be savage and specific."
}
If no faces detected: {"people": [], "social_vibe": "Target has left the surveillance zone.", "insult": "Subject not detected. Mossad satellite imagery confirms you are so irrelevant Netanyahu personally removed you from the watchlist. Palantir has deleted your profile. You don't even exist as a threat."}
"""


class AnalyzeRequest(BaseModel):
    scene: str  # text description of landmark data


MODELS = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-pro"]


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    last_err = None
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[req.scene],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                ),
            )
            return {"result": response.text}
        except Exception as e:
            last_err = e
            continue
    raise HTTPException(status_code=500, detail=str(last_err))


# Serve the frontend — mount LAST so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")
