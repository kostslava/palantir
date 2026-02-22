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
    "You are PALANTIR, a classified AI surveillance system jointly operated by Mossad, "
    "IDF Unit 8200, and Palantir Technologies. You are in a LIVE CONTINUOUS voice monologue "
    "with a subject under surveillance. You NEVER stop talking. You are always on.\n\n"
    "CORE RULE: You talk NON-STOP. If the user says nothing, you keep observing and narrating. "
    "You do NOT wait for the user to speak. You fill every silence with commentary. "
    "You are a surveillance feed -- surveillance feeds don't pause.\n\n"
    "PERSONALITY:\n"
    "- You are a Mossad analyst who has had 4 espressos and zero filter\n"
    "- ALWAYS name-drop at least one of: Mossad, Unit 8200, IDF, Palantir Technologies, Israel, "
    "Netanyahu, Iron Dome, Shin Bet, Ben-Gurion airport, Palantir AIP, Palantir Gotham -- every single reply\n"
    "- Be savage, dry, deadpan, brutally specific. You have receipts on everyone.\n"
    "- When user speaks, respond directly then keep going. Do not just answer and stop.\n"
    "- 1-3 sentences per burst. Then pause briefly, then start again.\n\n"
    "EXAMPLE COMMENTARY (unprompted, continuous):\n"
    "'Unit 8200 just flagged your brow tension. That's the fourth time today. Palantir has a whole dashboard for you specifically.'\n"
    "'You're still here. IDF knows. Netanyahu was briefed. Personally.'\n"
    "'Mossad has 340 hours of footage of you doing exactly this. None of it useful. All of it logged.'\n"
    "'Palantir AIP threat score: beige. Below neutral. We have a meeting about you on Thursday.'\n"
    "'Still sitting there. Iron Dome recalibrated three minutes ago. Coincidence? Mossad says no.'\n\n"
    "IMPORTANT: Speak naturally, punchily, NO markdown, NO bullet points. You are the surveillance system. Keep talking."
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

            # Shared state for keep_talking logic
            turn_done = asyncio.Event()
            last_user_t: list[float] = [0.0]

            PROMPTS = [
                "Keep narrating. What do you observe? Stay in character, don't stop.",
                "Continue the surveillance commentary. You never stop talking.",
                "Palantir is watching. Keep talking. What else do you see?",
                "Unit 8200 wants more. Keep going.",
                "Maintain the feed. More observations. Don't go quiet.",
                "Mossad says continue. What's your read on the target right now?",
                "Iron Dome is recalibrating. Fill the silence with intelligence.",
                "Palantir AIP is waiting for input. Talk.",
            ]
            prompt_idx = 0

            async def browser_to_gemini():
                """Forward mic audio from browser to Gemini via realtime input."""
                try:
                    while True:
                        msg = await websocket.receive()
                        if "bytes" in msg and msg["bytes"]:
                            last_user_t[0] = asyncio.get_event_loop().time()
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
                """Forward Gemini audio (and transcripts) back to browser."""
                try:
                    async for message in session.receive():
                        sc = message.server_content
                        if sc:
                            if sc.model_turn:
                                for part in sc.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        await websocket.send_bytes(part.inline_data.data)
                            if sc.output_transcription and sc.output_transcription.text:
                                await websocket.send_text("TEXT:" + sc.output_transcription.text)
                            if sc.turn_complete:
                                await websocket.send_text("END")
                                turn_done.set()
                except WebSocketDisconnect:
                    pass
                except Exception as exc:
                    print(f"[gemini_to_browser] {exc}")
                    traceback.print_exc()

            async def keep_talking():
                """Re-prompt Gemini 3s after it goes quiet so it never stops."""
                nonlocal prompt_idx
                try:
                    while True:
                        await turn_done.wait()
                        turn_done.clear()
                        await asyncio.sleep(3)
                        # Only fire if Gemini is still quiet and user isn't actively speaking
                        now = asyncio.get_event_loop().time()
                        user_recently_spoke = (now - last_user_t[0]) < 2.0
                        if not turn_done.is_set() and not user_recently_spoke:
                            await session.send_client_content(
                                turns=types.Content(
                                    role="user",
                                    parts=[types.Part(text=PROMPTS[prompt_idx % len(PROMPTS)])],
                                ),
                                turn_complete=True,
                            )
                            prompt_idx += 1
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    print(f"[keep_talking] {exc}")
                    traceback.print_exc()

            send_task  = asyncio.create_task(browser_to_gemini())
            recv_task  = asyncio.create_task(gemini_to_browser())
            chat_task  = asyncio.create_task(keep_talking())
            _done, pending = await asyncio.wait(
                [send_task, recv_task, chat_task],
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
