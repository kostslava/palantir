import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Zen Observer")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")
client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are a brutally honest, savage roast comedian who sees everything. Analyze the expressions, posture, and hand gestures of every person in the frame.

Output ONLY valid JSON in this exact format:
{
  "people": [{"id": 1, "expression": "...", "hands": "..."}],
  "social_vibe": "one sentence summary of the scene",
  "insult": "one savage, specific, funny roast of the person/people based on exactly what you see them doing. Be ruthlessly specific — mention their expression, their posture, what their hands are doing. Max 2 sentences. No generic insults."
}
If no people are visible, return: {"people": [], "social_vibe": "empty room", "insult": "Even the room looks bored without you."}
"""


class AnalyzeRequest(BaseModel):
    image: str  # base64-encoded JPEG


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        image_bytes = base64.b64decode(req.image)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                "Analyze the faces and hands in this scene.",
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                media_resolution="media_resolution_low",
            ),
        )
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve the frontend — mount LAST so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")
