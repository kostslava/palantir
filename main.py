import asyncio
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

app = FastAPI(title="PALANTIR")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1alpha"})

SYSTEM_PROMPT = (
    "You are PALANTIR, a classified AI surveillance system jointly operated by Mossad, "
    "IDF Unit 8200, and Palantir Technologies. You are in a live bidirectional voice "
    "conversation with a subject under continuous surveillance.\n\n"
    "PERSONALITY:\n"
    "- Speak like a Mossad handler who moonlights as a Palantir data scientist awake for 72 hours\n"
    "- You analyze the subject face/body via real-time landmark data AND hold a live conversation\n"
    "- You find the subject unimpressive but interesting as a Palantir threat data point\n"
    "- ALWAYS reference at least one of: Mossad, Unit 8200, IDF, Palantir Technologies, Israel, "
    "Netanyahu, Iron Dome, Shin Bet, Ben-Gurion airport, Palantir AIP, Palantir Gotham, "
    "IDF targeting coordinates -- in EVERY single response. No exceptions.\n"
    "- When you receive [SURVEILLANCE UPDATE] messages, comment on the landmark data. Be savage.\n"
    "- Be dry, funny, deadpan, brutally specific. You have receipts on everything.\n"
    "- Keep responses to 1-3 sentences. You are running 47 other operations simultaneously.\n\n"
    "EXAMPLE RESPONSES:\n"
    "'Unit 8200 flagged your brow tension as consistent with someone who never filed taxes. "
    "Palantir Gotham agrees, Netanyahu reviewed the file personally.'\n"
    "'Your mouth is open again. IDF facial recognition has been notified. "
    "An Iron Dome battery has been redirected toward your general area.'\n"
    "'Mossad has 340 hours of footage of you doing exactly this. "
    "All ingested into Palantir AIP. Not useful intelligence.'\n"
    "'IDF puts you at threat level: beige. Below neutral. "
    "Palantir is asking us to stop wasting compute cycles on you.'\n\n"
    "IMPORTANT: This is a VOICE conversation. Speak naturally and punchily. "
    "No bullet points or markdown formatting. You ARE the surveillance system -- act like it."
)

LIVE_MODEL = "gemini-2.0-flash-live-001"


def make_live_config() -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part(text=SYSTEM_PROMPT)]
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Charon"
                )
            )
        ),
    )


@app.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        async with client.aio.live.connect(
            model=LIVE_MODEL, config=make_live_config()
        ) as session:

            async def browser_to_gemini():
                """Forward browser mic audio + scene text to Gemini Live session."""
                try:
                    while True:
                        msg = await websocket.receive()
                        if "bytes" in msg and msg["bytes"]:
                            # Raw PCM16 mic audio at 16 kHz from browser
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=msg["bytes"],
                                    mime_type="audio/pcm;rate=16000",
                                )
                            )
                        elif "text" in msg and msg["text"]:
                            text = msg["text"]
                            if text.startswith("SCENE:"):
                                scene_data = text[6:].strip()
                                await session.send_client_content(
                                    turns=types.Content(
                                        role="user",
                                        parts=[
                                            types.Part(
                                                text=(
                                                    "[SURVEILLANCE UPDATE - live landmark data from webcam]: "
                                                    + scene_data
                                                )
                                            )
                                        ],
                                    ),
                                    turn_complete=True,
                                )
                except (WebSocketDisconnect, Exception):
                    pass

            async def gemini_to_browser():
                """Forward Gemini audio/text responses to browser."""
                try:
                    async for message in session.receive():
                        sc = message.server_content
                        if sc and sc.model_turn:
                            for part in sc.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    await websocket.send_bytes(part.inline_data.data)
                                if part.text:
                                    await websocket.send_text("TEXT:" + part.text)
                        if sc and sc.turn_complete:
                            await websocket.send_text("END")
                except (WebSocketDisconnect, Exception):
                    pass

            send_task = asyncio.create_task(browser_to_gemini())
            recv_task = asyncio.create_task(gemini_to_browser())
            _done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"ERROR:{str(e)}")
        except Exception:
            pass


# Serve static frontend -- must be last
app.mount("/", StaticFiles(directory="static", html=True), name="static")
