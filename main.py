import asyncio
import json
import os
import re
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
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

# ─── MEMORY STORE ─────────────────────────────────────────────────────────────
MEMORY_PATH = Path("memory.json")

class MemoryStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "sessions": [],
            "learned_profile": "",
            "facts": [],
            "profile_updated": None,
            "total_turns": 0,
            "emotion_counts": {},
        }

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def new_session(self) -> str:
        sid = str(uuid.uuid4())[:8]
        self.data["sessions"].append({
            "id": sid,
            "started": datetime.now(timezone.utc).isoformat(),
            "turns": [],
            "emotions": [],
        })
        if len(self.data["sessions"]) > 50:
            self.data["sessions"] = self.data["sessions"][-50:]
        self._save()
        return sid

    def add_turn(self, sid: str, role: str, text: str):
        s = next((x for x in self.data["sessions"] if x["id"] == sid), None)
        if s and text.strip():
            s["turns"].append({"role": role, "text": text[:400],
                               "ts": datetime.now(timezone.utc).isoformat()})
            self.data["total_turns"] = self.data.get("total_turns", 0) + 1
            self._save()

    def add_facts(self, new_facts: list[str]):
        """Append extracted facts to the persistent facts list, keep last 200."""
        existing = self.data.setdefault("facts", [])
        for f in new_facts:
            f = f.strip()
            if f and f not in existing:
                existing.append(f)
        self.data["facts"] = existing[-200:]
        self._save()

    def add_emotion(self, sid: str, mood: str):
        s = next((x for x in self.data["sessions"] if x["id"] == sid), None)
        if s:
            s["emotions"].append({"mood": mood, "ts": datetime.now(timezone.utc).isoformat()})
        c = self.data.setdefault("emotion_counts", {})
        c[mood] = c.get(mood, 0) + 1
        self._save()

    def update_profile(self, profile: str):
        self.data["learned_profile"] = profile
        self.data["profile_updated"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def get_context_snippet(self) -> str:
        parts = []
        if self.data.get("learned_profile"):
            parts.append(f"[LEARNED PROFILE]\n{self.data['learned_profile']}")
        facts = self.data.get("facts", [])
        if facts:
            parts.append("[KNOWN FACTS ABOUT TARGET]\n" + "\n".join(f"- {x}" for x in facts[-40:]))
        counts = self.data.get("emotion_counts", {})
        if counts:
            top = sorted(counts.items(), key=lambda x: -x[1])[:6]
            parts.append("[EMOTIONAL BASELINE] " + ", ".join(f"{m}({c}x)" for m, c in top))
        recent = []
        for s in self.data.get("sessions", [])[-3:]:
            for t in s.get("turns", [])[-4:]:
                recent.append(f"  [{t['role']}] {t['text'][:100]}")
        if recent:
            parts.append("[RECENT LOG]\n" + "\n".join(recent))
        return "\n\n".join(parts)

memory = MemoryStore(MEMORY_PATH)


async def learn_from_turn(user_text: str, assistant_text: str, mood: str) -> list[str]:
    """Extract 1-3 concrete facts from a single exchange. Returns list of fact strings."""
    prompt = (
        "You are a PALANTIR intelligence extraction engine. "
        "Given one exchange between a surveillance target and PALANTIR, "
        "extract 1-3 short, specific, reusable facts about the TARGET ONLY. "
        "Facts must be concrete: things they said, admitted, revealed, implied, or reacted to. "
        "Do NOT extract facts about PALANTIR. Do NOT be vague. "
        "Reply ONLY with a JSON array of strings, e.g. [\"fact1\", \"fact2\"]. No explanation.\n\n"
        f"Target mood: {mood}\n"
        f"Target said: {user_text}\n"
        f"PALANTIR replied: {assistant_text}"
    )
    try:
        resp = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        raw = (resp.text or "").strip()
        facts = json.loads(raw)
        if isinstance(facts, list):
            return [str(f) for f in facts if f][:3]
    except Exception as e:
        print(f"[learn_from_turn] {e}")
    return []


async def generate_profile() -> str:
    """Synthesise a full behavioral profile from all accumulated facts."""
    facts = memory.data.get("facts", [])
    counts = memory.data.get("emotion_counts", {})
    if not facts and not counts:
        return memory.data.get("learned_profile", "")
    prompt = (
        "You are PALANTIR. Write a 4-6 sentence classified behavioral dossier on the surveillance "
        "target using the facts and emotional data below. Be specific, cold, analytical, and reference "
        "actual facts listed. This is a classified intelligence document.\n\n"
        f"EMOTION COUNTS: {json.dumps(counts)}\n\n"
        "EXTRACTED FACTS:\n" + "\n".join(f"- {f}" for f in facts[-60:])
    )
    try:
        resp = await client.aio.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        return (resp.text or "").strip()
    except Exception as e:
        print(f"[generate_profile] {e}")
        return memory.data.get("learned_profile", "")


@app.get("/api/memory")
async def get_memory():
    return JSONResponse(memory.data)

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
    ctx = memory.get_context_snippet()
    sys_prompt = SYSTEM_PROMPT
    if ctx:
        sys_prompt = sys_prompt + "\n\n" + ctx
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part(text=sys_prompt)]
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
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


@app.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    await websocket.accept()
    print("[ws] client connected")
    session_id = memory.new_session()
    print(f"[ws] memory session: {session_id}")
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
                        elif "text" in msg and msg["text"]:
                            try:
                                ctrl = json.loads(msg["text"])
                                if ctrl.get("type") == "emotion":
                                    mood = ctrl.get("mood", "neutral")
                                    memory.add_emotion(session_id, mood)
                                    current_mood_ref[0] = mood
                                    print(f"[emotion] {mood}")
                            except Exception:
                                pass
                except WebSocketDisconnect:
                    print("[browser_to_gemini] browser disconnected")
                except Exception as exc:
                    print(f"[browser_to_gemini] ERROR: {exc}")
                    traceback.print_exc()

            # Transcript accumulation state
            transcript_buf = []
            model_transcript_buf = []
            debounce_handle: asyncio.TimerHandle | None = None
            last_user_text: list[str] = [""]   # mutable container for closure
            current_mood_ref: list[str] = ["neutral"]

            async def process_full_transcript():
                nonlocal transcript_buf
                full = "".join(transcript_buf).strip()
                transcript_buf = []
                if not full:
                    return
                memory.add_turn(session_id, "user", full)
                last_user_text[0] = full
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
                nonlocal transcript_buf, model_transcript_buf, debounce_handle
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
                                # Input transcription (user speech)
                                it = getattr(sc, 'input_transcription', None)
                                if it:
                                    txt = getattr(it, 'text', None) or str(it)
                                    if txt:
                                        transcript_buf.append(txt)
                                        schedule_debounce()
                                # Output transcription (model speech -> text)
                                ot = getattr(sc, 'output_transcription', None)
                                if ot:
                                    txt = getattr(ot, 'text', None) or str(ot)
                                    if txt:
                                        model_transcript_buf.append(txt)
                                if sc.turn_complete:
                                    turns += 1
                                    print(f"[gemini→browser] turn {turns} complete ({audio_chunks} total chunks)")
                                    if transcript_buf:
                                        if debounce_handle:
                                            debounce_handle.cancel()
                                            debounce_handle = None
                                        await process_full_transcript()
                                    full_model = ""
                                    if model_transcript_buf:
                                        full_model = "".join(model_transcript_buf).strip()
                                        model_transcript_buf.clear()
                                        if full_model:
                                            memory.add_turn(session_id, "assistant", full_model)
                                    # per-turn learning
                                    user_said = last_user_text[0]
                                    if user_said and full_model:
                                        asyncio.ensure_future(
                                            _learn_and_inject(user_said, full_model, current_mood_ref[0], turns, session, websocket)
                                        )
                        if not got_any:
                            print("[gemini_to_browser] session closed by Gemini")
                            break
                except WebSocketDisconnect:
                    print("[gemini_to_browser] browser disconnected")
                except Exception as exc:
                    print(f"[gemini_to_browser] ERROR: {exc}")
                    traceback.print_exc()

            async def _learn_and_inject(user_said: str, model_said: str, mood: str, turn_n: int, sess, ws):
                """Extract facts from this exchange and inject immediately. Every 5 turns, regenerate full profile."""
                print(f"[learn] extracting facts from turn {turn_n} (mood={mood})...")
                new_facts = await learn_from_turn(user_said, model_said, mood)
                if new_facts:
                    memory.add_facts(new_facts)
                    print(f"[learn] +{len(new_facts)} facts: {new_facts}")
                    fact_lines = "\n".join(f"- {f}" for f in new_facts)
                    inject = (
                        "[PALANTIR INTEL UPDATE — do not speak]\n"
                        f"New confirmed facts about target:\n{fact_lines}\n\n"
                        "Absorb silently. Do NOT respond to this. Wait for target to speak next."
                    )
                    try:
                        await sess.send_client_content(
                            turns=types.Content(role="user", parts=[types.Part(text=inject)]),
                            turn_complete=True,
                        )
                        print(f"[learn] facts injected")
                    except Exception as e:
                        print(f"[learn inject] {e}")
                if turn_n % 5 == 0:
                    print(f"[learn] turn {turn_n} — regenerating full profile...")
                    profile = await generate_profile()
                    if profile:
                        memory.update_profile(profile)
                        print(f"[learn] profile regenerated ({len(profile)} chars)")
                try:
                    await ws.send_text(json.dumps({
                        "type": "memory_updated",
                        "profile_preview": (memory.data.get("learned_profile") or "")[:180],
                        "total_turns": memory.data.get("total_turns", 0),
                        "emotion_counts": memory.data.get("emotion_counts", {}),
                        "fact_count": len(memory.data.get("facts", [])),
                    }))
                except Exception:
                    pass

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
