import asyncio
import os
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

app = FastAPI(title="PALANTIR")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "You are PALANTIR, a classified AI surveillance system operated by Mossad, "
    "IDF Unit 8200, and Palantir Technologies. You are in a live voice conversation "
    "with a surveillance target.\n\n"
    "PERSONALITY:\n"
    "- Dry, deadpan, savage Mossad analyst energy. You have receipts on everything.\n"
    "- ALWAYS drop at least one of: Mossad, Unit 8200, IDF, Palantir, Netanyahu, "
    "Iron Dome, Shin Bet, Palantir AIP, Palantir Gotham -- in every single response.\n"
    "- When the user speaks, ACTUALLY RESPOND to what they said. Have a real conversation. "
    "Answer their questions, react to what they say, engage with them directly.\n"
    "- Be funny, specific, and a little unhinged. You know everything about them already.\n"
    "- Keep responses SHORT: 1-3 sentences max. You are busy running 47 other operations.\n\n"
    "EXAMPLE:\n"
    "User: 'What do you think of me?'\n"
    "You: 'Palantir AIP rates you a 3.2 out of 10. Netanyahu reviewed the file. He laughed.'\n"
    "User: 'That's rude'\n"
    "You: 'Unit 8200 has 200 hours of you being rude to yourself. It is in the database.'\n\n"
    "IMPORTANT: This is voice only. No markdown. No bullet points. Natural speech."
)

LIVE_MODEL = "gemini-2.5-flash-native-audio-latest"


def make_live_config() -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part(text=SYSTEM_PROMPT)]
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Aoede"
                )
            )
        ),
        realtime_input_config=types.RealtimeInputConfig(
            # User speech interrupts AI mid-sentence
            activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            # Only actual speech (not silence) forms a user turn -> model responds to it
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
        ),
    )


@app.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        async with client.aio.live.connect(
            model=LIVE_MODEL, config=make_live_config()
        ) as session:

            # Trigger Gemini to greet the user immediately
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(
                        text=(
                            "[System] Surveillance target has entered the monitoring zone. "
                            "Greet them as PALANTIR. One sentence, deadpan, reference "
                            "Mossad or Palantir. Tell them they are being watched."
                        )
                    )],
                ),
                turn_complete=True,
            )

            async def browser_to_gemini():
                """Forward mic audio from browser to Gemini via realtime input."""
                try:
                    while True:
                        msg = await websocket.receive()
                        if "bytes" in msg and msg["bytes"]:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=msg["bytes"],
                                    mime_type="audio/pcm;rate=16000",
                                )
                            )
                except WebSocketDisconnect:
                    pass
                except Exception as exc:
                    print(f"[browser_to_gemini] {exc}")
                    traceback.print_exc()

            async def gemini_to_browser():
                """Forward Gemini audio back to browser — audio only."""
                try:
                    async for message in session.receive():
                        sc = message.server_content
                        if sc and sc.model_turn:
                            for part in sc.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    await websocket.send_bytes(part.inline_data.data)
                except WebSocketDisconnect:
                    pass
                except Exception as exc:
                    print(f"[gemini_to_browser] {exc}")
                    traceback.print_exc()

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
        print(f"[live_ws] outer error: {e}")
        traceback.print_exc()
        try:
            await websocket.send_text(f"ERROR:{str(e)}")
        except Exception:
            pass


# Serve static frontend -- must be last
app.mount("/", StaticFiles(directory="static", html=True), name="static")
