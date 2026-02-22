import os
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

Output ONLY valid JSON (no markdown fences):
{
  "people": [{"id": 1, "expression": "...", "hands": "..."}],
  "social_vibe": "one dry surveillance-report sentence describing the scene",
  "insult": "1-2 sentence roast written like a declassified Mossad assessment. Reference specific landmark signals. Must include Netanyahu, IDF, Mossad, Unit 8200, or Palantir. Be savage and specific."
}
If no faces detected: {"people": [], "social_vibe": "Target has left the surveillance zone.", "insult": "Subject not detected. Mossad satellite imagery confirms you are so irrelevant Netanyahu personally removed you from the watchlist. Palantir has deleted your profile. You don't even exist as a threat."}
"""


class AnalyzeRequest(BaseModel):
    scene: str


# ── Text analysis: Gemma 3 27B (30 RPM, no daily cap) ────────────────────
@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=[SYSTEM_PROMPT + "\n\nScene data:\n" + req.scene],
        )
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return {"result": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Audio: Live API WebSocket — streams raw PCM 24kHz back to browser ───────
@app.websocket("/ws/speak")
async def speak_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            if not text.strip():
                continue
            try:
                live_config = types.LiveConnectConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Charon"
                            )
                        )
                    ),
                )
                async with client.aio.live.connect(
                    model="gemini-2.5-flash-native-audio-latest",
                    config=live_config,
                ) as session:
                    await session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part(text=text)],
                        ),
                        turn_complete=True,
                    )
                    async for message in session.receive():
                        if (
                            message.server_content
                            and message.server_content.model_turn
                        ):
                            for part in message.server_content.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    await websocket.send_bytes(part.inline_data.data)
                        if (
                            message.server_content
                            and message.server_content.turn_complete
                        ):
                            break
                await websocket.send_text("END")
            except Exception as e:
                await websocket.send_text(f"ERROR:{str(e)}")
    except WebSocketDisconnect:
        pass


# Serve static frontend — must be last
app.mount("/", StaticFiles(directory="static", html=True), name="static")
