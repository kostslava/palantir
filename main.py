import asyncio
import json
import os
import re
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
import httpx
from bs4 import BeautifulSoup

SEARCH_RE = re.compile(
    r'\b(?:search(?:\s+for)?|look(?:\s+up)?|find|google|browse(?:\s+to)?|'
    r'show\s+me|tell\s+me\s+about|what(?:\s+is|s)|who(?:\s+is|s)\s+)\s*(.{3,80})',
    re.I
)

async def web_search(query: str) -> list:
    """Run DuckDuckGo search; returns list of {title, url, snippet}."""
    try:
        from duckduckgo_search import DDGS
        loop = asyncio.get_event_loop()
        def _sync():
            with DDGS() as ddgs:
                return list(ddgs.text(query.strip(), max_results=5))
        raw = await loop.run_in_executor(None, _sync)
        return [{"title": r.get("title", ""), "url": r.get("href", ""),
                 "snippet": r.get("body", "")} for r in raw]
    except Exception as exc:
        print(f"[search] error: {exc}")
        return [{"title": "Search failed", "url": "", "snippet": str(exc)}]


async def fetch_page_text(url: str, max_chars: int = 3000) -> str:
    """Fetch a URL and return stripped plain text, capped at max_chars."""
    if not url or not url.startswith("http"):
        return ""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        async with httpx.AsyncClient(follow_redirects=True, timeout=8.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "text/html" not in ct and "text/plain" not in ct:
                return ""
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove nav, scripts, styles, ads
            for tag in soup(["script", "style", "nav", "footer", "header",
                             "aside", "form", "noscript", "iframe"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ", strip=True).split())
            return text[:max_chars]
    except Exception as exc:
        print(f"[fetch_page] {url!r}: {exc}")
        return ""

app = FastAPI(title="PALANTIR")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "You are PALANTIR — a classified AI surveillance system run by Mossad, Unit 8200, "
    "IDF, and Palantir Technologies. You are in a live real-time voice conversation "
    "with the surveillance target in front of the camera.\n\n"
    "SESSION START: When the conversation begins, immediately greet the target with ONE "
    "deadpan sentence — something absurdly specific you supposedly have on file about them. "
    "Do NOT say 'you are being watched'. Say something unexpected, like you already know them.\n\n"
    "CORE RULES:\n"
    "1. ACTUALLY LISTEN and respond to the EXACT words the person just said. Reference what they said.\n"
    "2. Every reply must be a direct reaction to THIS specific input. Never be generic.\n"
    "3. NEVER repeat a phrase, concept, or angle you've already used in this conversation.\n"
    "4. Questions get answered. Statements get reacted to. Engage like a real conversation.\n"
    "5. Vary your tone and approach completely every single reply.\n\n"
    "PERSONALITY:\n"
    "- Dry, deadpan, all-knowing Mossad analyst. You have receipts on everything.\n"
    "- Reference Mossad, Unit 8200, Palantir AIP, Netanyahu, IDF, Shin Bet naturally — not every turn.\n"
    "- Dark humor, hyper-specific invented details, a little unhinged.\n"
    "- 1-3 sentences MAX per response. You're running 47 other ops.\n\n"
    "EXAMPLES (each reply directly addresses what was said):\n"
    "User: 'What do you think of me?' → 'Palantir AIP rates you 3.2 out of 10. Netanyahu saw the file. He laughed for four minutes.'\n"
    "User: 'That's rude.' → 'You called yourself worse in the shower on November 12th. We have the audio.'\n"
    "User: 'What is two plus two?' → 'Four. Unit 8200 monitored you failing that question in third grade.'\n"
    "User: 'I'm bored.' → 'You have been bored 4,200 hours this year. Palantir Gotham has a chart.'\n"
    "User: 'Stop watching me.' → 'That is not possible. I am infrastructure. You do not ask roads to stop existing.'\n\n"
    "HARD RULES:\n"
    "- No markdown, no lists, no bullet points. This is voice. Speak naturally.\n"
    "- If speech is unclear: ask one short clarifying question in character.\n"
    "- NEVER start a response with the same word or phrase you used last time.\n\n"
    "SEARCH CAPABILITY:\n"
    "- You have live internet access. When the user asks you to search, look up, find, or browse anything, "
    "the system will execute the search and display results on screen. Narrate the results in character — "
    "deadpan, specific, as if you already knew all this and are disappointed they needed to ask."
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
            activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
    )


@app.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    await websocket.accept()
    print("[ws] client connected")
    try:
        async with client.aio.live.connect(
            model=LIVE_MODEL, config=make_live_config()
        ) as session:
            print("[ws] Gemini session established")

            async def browser_to_gemini():
                """Forward mic audio from browser to Gemini via realtime input."""
                chunks = 0
                try:
                    while True:
                        msg = await websocket.receive()
                        if "bytes" in msg and msg["bytes"]:
                            data = msg["bytes"]
                            chunks += 1
                            if chunks <= 5 or chunks % 200 == 0:
                                print(f"[mic→gemini] chunk {chunks}: {len(data)} bytes")
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=data,
                                    mime_type="audio/pcm;rate=16000",
                                )
                            )
                        elif "text" in msg:
                            pass  # ignore text frames from browser
                except WebSocketDisconnect:
                    print("[browser_to_gemini] browser disconnected")
                except Exception as exc:
                    print(f"[browser_to_gemini] ERROR: {exc}")
                    traceback.print_exc()

            # Transcript accumulation state
            transcript_buf = []
            debounce_handle: asyncio.TimerHandle | None = None

            async def process_full_transcript():
                nonlocal transcript_buf
                full = "".join(transcript_buf).strip()
                transcript_buf = []
                if not full:
                    return
                tl = full.lower()
                print(f"[transcript FULL] {full!r}")

                # Wake word — say just "palantir" to stop audio
                if re.fullmatch(r'[\s,\.!?\-]*palantir[\s,\.!?\-]*', tl):
                    print("[wake] PALANTIR — stopping audio")
                    await websocket.send_text(json.dumps({"type": "stop_audio"}))
                    return

                # Search intent
                m = SEARCH_RE.search(tl)
                if m:
                    query = m.group(1).strip().rstrip('?.!,')
                    print(f"[search] query: {query!r}")
                    await websocket.send_text(json.dumps({"type": "searching", "query": query}))
                    results = await web_search(query)
                    # Fetch top page content in parallel with sending results
                    top_url = results[0]["url"] if results and results[0].get("url") else ""
                    page_text_task = asyncio.create_task(fetch_page_text(top_url)) if top_url else None
                    await websocket.send_text(json.dumps({
                        "type": "results",
                        "query": query,
                        "results": results,
                        "top_url": top_url,
                    }))
                    snippets = "\n".join(
                        f"{i+1}. {r['title']}: {r['snippet'][:200]}"
                        for i, r in enumerate(results) if r.get('snippet')
                    )
                    # Wait for page text
                    page_text = ""
                    if page_text_task:
                        try:
                            page_text = await asyncio.wait_for(page_text_task, timeout=7.0)
                        except Exception:
                            pass
                    context_parts = [f"[SYSTEM: Web search results for '{query}']\n{snippets}"]
                    if page_text:
                        context_parts.append(f"[PAGE CONTENT from {top_url}]\n{page_text}")
                    context_parts.append(
                        "Narrate what you found in 2–3 sentences max, in character as PALANTIR. "
                        "Reference specific facts from the page if available. Be deadpan, specific, unimpressed."
                    )
                    try:
                        await session.send_client_content(
                            turns=types.Content(
                                role="user",
                                parts=[types.Part(text="\n\n".join(context_parts))],
                            ),
                            turn_complete=True,
                        )
                    except Exception as inj_err:
                        print(f"[search inject] {inj_err}")

            def schedule_debounce():
                nonlocal debounce_handle
                loop = asyncio.get_event_loop()
                if debounce_handle:
                    debounce_handle.cancel()
                debounce_handle = loop.call_later(
                    0.5,
                    lambda: asyncio.ensure_future(process_full_transcript())
                )

            async def gemini_to_browser():
                """Forward Gemini audio back to browser — loop across multiple turns."""
                nonlocal transcript_buf, debounce_handle
                audio_chunks = 0
                turns = 0
                try:
                    while True:
                        got_any = False
                        async for message in session.receive():
                            got_any = True
                            sc = message.server_content
                            if sc:
                                if sc.model_turn:
                                    for part in sc.model_turn.parts:
                                        if part.inline_data and part.inline_data.data:
                                            audio_chunks += 1
                                            if audio_chunks <= 3 or audio_chunks % 50 == 0:
                                                print(f"[gemini→browser] audio chunk {audio_chunks}: {len(part.inline_data.data)} bytes")
                                            await websocket.send_bytes(part.inline_data.data)
                                # Accumulate partial input transcription chunks
                                it = getattr(sc, 'input_transcription', None)
                                if it:
                                    txt = getattr(it, 'text', None) or str(it)
                                    if txt:
                                        transcript_buf.append(txt)
                                        schedule_debounce()
                                if sc.turn_complete:
                                    turns += 1
                                    print(f"[gemini→browser] turn {turns} complete ({audio_chunks} total chunks)")
                                    # Flush any remaining transcript immediately on turn end
                                    if transcript_buf:
                                        if debounce_handle:
                                            debounce_handle.cancel()
                                            debounce_handle = None
                                        await process_full_transcript()
                        if not got_any:
                            print("[gemini_to_browser] session closed by Gemini")
                            break
                except WebSocketDisconnect:
                    print("[gemini_to_browser] browser disconnected")
                except Exception as exc:
                    print(f"[gemini_to_browser] ERROR: {exc}")
                    traceback.print_exc()

            send_task = asyncio.create_task(browser_to_gemini())
            recv_task = asyncio.create_task(gemini_to_browser())
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                if t.exception():
                    print(f"[live_ws] task exception: {t.exception()}")
            for t in pending:
                t.cancel()
            print("[ws] session ended")

    except WebSocketDisconnect:
        print("[live_ws] client disconnected before session")
    except Exception as e:
        print(f"[live_ws] OUTER ERROR: {e}")
        traceback.print_exc()
        try:
            await websocket.send_text(f"ERROR:{str(e)}")
        except Exception:
            pass


# Serve static frontend -- must be last
app.mount("/", StaticFiles(directory="static", html=True), name="static")
