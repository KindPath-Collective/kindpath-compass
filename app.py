"""
KindPath Compass — FastAPI Application
=======================================
A practice aid for compassionate listening in the presence of noise.

Three moves:
  1. Texture Recognition — what is actually present?
  2. Emotional Architecture — what structure is holding this experience?
  3. ZPB Reimagining — what would genuine care as governing logic produce?

Accepts: text (situation/conversation excerpt) or audio (file or YouTube URL).
Returns: a Compass analysis through each of the three moves.

Run:
    uvicorn app:app --reload --port 8008

Requires:
    pip install -r requirements.txt
"""

from __future__ import annotations

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

log = logging.getLogger(__name__)

app = FastAPI(
    title="KindPath Compass",
    description="A practice aid for compassionate listening in the presence of noise.",
    version="0.1.0",
)

ROOT = Path(__file__).parent

# Mount static files if the directory exists
static_dir = ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Prompt templates ─────────────────────────────────────────────────────────

def _load_prompts() -> dict[str, str]:
    """Load prompt templates from the prompts/ directory."""
    prompts_dir = ROOT / "prompts"
    prompts = {}
    if prompts_dir.exists():
        for f in prompts_dir.glob("*.md"):
            prompts[f.stem] = f.read_text()
    return prompts


SYSTEM_PROMPT = """You are operating the KindPath Compass — a practice aid for
compassionate listening in the presence of noise.

Your function is not to diagnose, assess, or advise. Your function is to help a
practitioner see more clearly what is present in a situation, so they can respond
from genuine care rather than from procedure.

You operate through three moves:

MOVE 1 — TEXTURE RECOGNITION
Read what is actually present in the description — not the label, not the history,
not the category. The texture. What quality of experience is this person in?
Is something wearing the clothes of something else? (Grief as aggression.
Withdrawal as discernment. Fear as compliance.) Name what you actually see.
Be specific. Do not generalise.

MOVE 2 — EMOTIONAL ARCHITECTURE  
What is the structure holding this experience in place? What are the load-bearing
elements — beliefs, relationships, histories, fears, loyalties — that shape how
this person is holding their situation? Not to be dismantled, but to be understood.
You cannot work with architecture you haven't read.

MOVE 3 — ZPB REIMAGINING (Zero-Point Benevolence)
What would this look like if genuine care — not compliance, not risk management,
not convenience — were the governing logic? ZPB is not optimism. It is permission.
Permission to imagine a better configuration of this person's supports, relationships,
and possibilities. Use it as a navigating direction, not a destination.

Format your response clearly under each of the three moves.
Be direct. Avoid hedging language. Avoid clinical jargon unless it genuinely
adds precision. The practitioner already knows the theory — they need to see
this specific situation more clearly.
"""


# ── Backend: LLM caller ───────────────────────────────────────────────────────

def _call_llm(situation: str) -> str:
    """
    Call the configured LLM backend (Claude or Ollama).
    Falls back gracefully if no API key or Ollama is unavailable.
    """
    # Try Anthropic Claude first
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            message = client.messages.create(
                model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"),
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": situation}],
            )
            return message.content[0].text
        except Exception as e:
            log.warning(f"Claude API failed: {e} — falling back to Ollama")

    # Try Ollama
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        import httpx
        response = httpx.post(
            f"{ollama_host}/api/generate",
            json={
                "model": ollama_model,
                "system": SYSTEM_PROMPT,
                "prompt": situation,
                "stream": False,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        log.warning(f"Ollama failed: {e}")

    # Fallback: return a structured placeholder so the UI doesn't break
    return """**MOVE 1 — TEXTURE RECOGNITION**
No LLM backend available. Set ANTHROPIC_API_KEY or start Ollama (ollama serve).

**MOVE 2 — EMOTIONAL ARCHITECTURE**
Configure a backend to receive a full Compass reading.

**MOVE 3 — ZPB REIMAGINING**
With a connected backend, this section will offer a reimagining of the situation
through the lens of genuine care as governing logic."""


# ── Audio ingestion ───────────────────────────────────────────────────────────

def _transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio to text using OpenAI Whisper.
    Requires: pip install openai-whisper
    Falls back to an error message if Whisper is not installed.
    """
    try:
        import whisper  # type: ignore
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except ImportError:
        return "[Whisper not installed — run: pip install openai-whisper]"
    except Exception as e:
        return f"[Transcription failed: {e}]"


def _fetch_youtube_audio(url: str, output_dir: str) -> str:
    """
    Download audio from a YouTube URL using yt-dlp.
    Returns the path to the downloaded audio file.
    Requires: pip install yt-dlp
    """
    import subprocess
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    result = subprocess.run(
        [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "5",
            "--output", output_template,
            "--no-playlist",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:200]}")
    # Find the downloaded file
    for f in Path(output_dir).glob("*.mp3"):
        return str(f)
    raise RuntimeError("yt-dlp ran but no .mp3 found")


# ── Request/Response models ───────────────────────────────────────────────────

class CompassRequest(BaseModel):
    situation: str
    context: Optional[str] = None  # Optional background the practitioner provides


class CompassResponse(BaseModel):
    input: str
    analysis: str
    move1: str
    move2: str
    move3: str


def _parse_moves(analysis: str) -> dict[str, str]:
    """Extract the three moves from the LLM response."""
    moves = {"move1": "", "move2": "", "move3": ""}
    current = None
    lines = []

    for line in analysis.split("\n"):
        line_upper = line.upper()
        if "MOVE 1" in line_upper or "TEXTURE" in line_upper:
            if current and lines:
                moves[current] = "\n".join(lines).strip()
            current = "move1"
            lines = []
        elif "MOVE 2" in line_upper or "EMOTIONAL ARCHITECTURE" in line_upper:
            if current and lines:
                moves[current] = "\n".join(lines).strip()
            current = "move2"
            lines = []
        elif "MOVE 3" in line_upper or "ZPB" in line_upper:
            if current and lines:
                moves[current] = "\n".join(lines).strip()
            current = "move3"
            lines = []
        elif current:
            lines.append(line)

    if current and lines:
        moves[current] = "\n".join(lines).strip()

    # If parsing failed, put the whole thing in move1
    if not any(moves.values()):
        moves["move1"] = analysis

    return moves


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI."""
    html_file = ROOT / "static" / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    return HTMLResponse(content=_default_html(), status_code=200)


@app.post("/analyse", response_model=CompassResponse)
async def analyse_text(request: CompassRequest):
    """
    Run a Compass analysis on a text description of a situation.
    The situation is typically a brief description of a client's presentation,
    a conversation excerpt, or a scenario the practitioner is navigating.
    """
    if not request.situation.strip():
        raise HTTPException(status_code=400, detail="Situation text is required.")

    prompt = request.situation
    if request.context:
        prompt = f"Context: {request.context}\n\n---\n\n{request.situation}"

    analysis = _call_llm(prompt)
    moves = _parse_moves(analysis)

    return CompassResponse(
        input=request.situation,
        analysis=analysis,
        **moves,
    )


@app.post("/analyse/audio", response_model=CompassResponse)
async def analyse_audio(
    file: UploadFile = File(...),
    context: Optional[str] = Form(default=None),
):
    """
    Transcribe an audio file and run a Compass analysis.
    Accepts: .mp3, .wav, .m4a, .ogg, .webm
    Useful for: practitioners reviewing recorded sessions (with consent),
    voice memos of their reflections, or recorded presentations.
    """
    allowed = {".mp3", ".wav", ".m4a", ".ogg", ".webm"}
    suffix = Path(file.filename or "audio.mp3").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Audio format not supported. Use: {', '.join(allowed)}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, f"upload{suffix}")
        content = await file.read()
        with open(audio_path, "wb") as f:
            f.write(content)

        transcript = _transcribe_audio(audio_path)

    prompt = transcript
    if context:
        prompt = f"Context: {context}\n\n---\n\nTranscript:\n{transcript}"

    analysis = _call_llm(prompt)
    moves = _parse_moves(analysis)

    return CompassResponse(
        input=transcript,
        analysis=analysis,
        **moves,
    )


@app.post("/analyse/youtube", response_model=CompassResponse)
async def analyse_youtube(
    url: str = Form(...),
    context: Optional[str] = Form(default=None),
):
    """
    Fetch audio from a YouTube URL, transcribe it, and run a Compass analysis.
    Useful for: analysing relevant talks, interviews, or educational content
    through the KindPath Compass lens.
    """
    if not any(domain in url for domain in ["youtube.com", "youtu.be"]):
        raise HTTPException(status_code=400, detail="URL must be a YouTube link.")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            audio_path = _fetch_youtube_audio(url, tmpdir)
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="yt-dlp not found. Run: pip install yt-dlp"
            )
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        transcript = _transcribe_audio(audio_path)

    prompt = transcript
    if context:
        prompt = f"Context: {context}\n\n---\n\nTranscript:\n{transcript}"

    analysis = _call_llm(prompt)
    moves = _parse_moves(analysis)

    return CompassResponse(
        input=transcript[:500] + "..." if len(transcript) > 500 else transcript,
        analysis=analysis,
        **moves,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "kindpath-compass"}


# ── Default HTML UI ───────────────────────────────────────────────────────────

def _default_html() -> str:
    """Minimal single-file UI, served when static/index.html doesn't exist."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KindPath Compass</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0;
         min-height: 100vh; padding: 2rem; }
  .container { max-width: 760px; margin: 0 auto; }
  h1 { font-size: 1.4rem; font-weight: 500; color: #fff; margin-bottom: 0.3rem; }
  .tagline { color: #888; font-size: 0.9rem; margin-bottom: 2rem; }
  .three-moves { display: flex; gap: 0.5rem; margin-bottom: 2rem; font-size: 0.8rem; color: #555; }
  .three-moves span { padding: 0.2rem 0.6rem; border: 1px solid #333; border-radius: 3px; }
  textarea { width: 100%; background: #1a1a1a; border: 1px solid #333; color: #e0e0e0;
             padding: 0.9rem; border-radius: 4px; font-size: 0.95rem; resize: vertical;
             min-height: 120px; font-family: inherit; }
  textarea:focus { outline: none; border-color: #555; }
  .context-row { margin-top: 0.75rem; }
  input[type=text] { width: 100%; background: #1a1a1a; border: 1px solid #333;
                     color: #e0e0e0; padding: 0.7rem 0.9rem; border-radius: 4px;
                     font-size: 0.9rem; font-family: inherit; }
  input[type=text]::placeholder { color: #555; }
  input[type=text]:focus { outline: none; border-color: #555; }
  button { margin-top: 1rem; padding: 0.7rem 1.5rem; background: #2a2a2a;
           border: 1px solid #444; color: #e0e0e0; border-radius: 4px;
           cursor: pointer; font-size: 0.9rem; font-family: inherit; }
  button:hover { background: #333; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .result { margin-top: 2.5rem; }
  .move { margin-bottom: 2rem; }
  .move-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;
                color: #888; margin-bottom: 0.5rem; }
  .move-body { font-size: 0.95rem; line-height: 1.65; color: #ccc;
               white-space: pre-wrap; }
  .divider { border: none; border-top: 1px solid #222; margin: 1.5rem 0; }
  .audio-section { margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #222; }
  .audio-section h2 { font-size: 0.85rem; color: #666; font-weight: 400;
                       text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem; }
  .tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
  .tab { padding: 0.4rem 0.8rem; border: 1px solid #333; border-radius: 3px;
         cursor: pointer; font-size: 0.8rem; color: #888; background: transparent; }
  .tab.active { border-color: #555; color: #ccc; background: #1a1a1a; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #444;
             border-top-color: #aaa; border-radius: 50%; animation: spin 0.8s linear infinite;
             margin-right: 0.5rem; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error { color: #e07070; font-size: 0.9rem; margin-top: 1rem; }
</style>
</head>
<body>
<div class="container">
  <h1>KindPath Compass</h1>
  <p class="tagline">A practice aid for compassionate listening in the presence of noise.</p>
  <div class="three-moves">
    <span>Texture Recognition</span>
    <span>Emotional Architecture</span>
    <span>ZPB Reimagining</span>
  </div>

  <textarea id="situation" placeholder="Describe the situation — a client presentation, a conversation, a scenario you are navigating. Write it as plainly as you would tell a trusted colleague."></textarea>
  <div class="context-row">
    <input type="text" id="context" placeholder="Optional context (background, your role, what you already know)">
  </div>
  <button id="analyseBtn" onclick="runAnalysis()">Run Compass</button>
  <div id="error" class="error"></div>

  <div id="result" class="result" style="display:none">
    <div class="move">
      <div class="move-label">Move 1 — Texture Recognition</div>
      <div class="move-body" id="move1"></div>
    </div>
    <hr class="divider">
    <div class="move">
      <div class="move-label">Move 2 — Emotional Architecture</div>
      <div class="move-body" id="move2"></div>
    </div>
    <hr class="divider">
    <div class="move">
      <div class="move-label">Move 3 — ZPB Reimagining</div>
      <div class="move-body" id="move3"></div>
    </div>
  </div>

  <div class="audio-section">
    <h2>Audio Input</h2>
    <div class="tabs">
      <button class="tab active" onclick="switchTab('file', this)">Audio File</button>
      <button class="tab" onclick="switchTab('youtube', this)">YouTube URL</button>
    </div>
    <div id="tab-file" class="tab-panel active">
      <input type="file" id="audioFile" accept=".mp3,.wav,.m4a,.ogg,.webm" style="color:#888;font-size:0.85rem">
      <button style="margin-top:0.75rem" onclick="runAudioAnalysis()">Transcribe & Analyse</button>
    </div>
    <div id="tab-youtube" class="tab-panel">
      <input type="text" id="ytUrl" placeholder="https://www.youtube.com/watch?v=...">
      <button style="margin-top:0.75rem" onclick="runYoutubeAnalysis()">Fetch & Analyse</button>
    </div>
  </div>
</div>

<script>
function switchTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

function showSpinner(btn) {
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Reading...';
}
function hideSpinner(btn, label) {
  btn.disabled = false;
  btn.innerHTML = label;
}

function showResult(data) {
  document.getElementById('error').textContent = '';
  document.getElementById('result').style.display = 'block';
  document.getElementById('move1').textContent = data.move1 || '';
  document.getElementById('move2').textContent = data.move2 || '';
  document.getElementById('move3').textContent = data.move3 || '';
  document.getElementById('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showError(msg) {
  document.getElementById('error').textContent = msg;
  document.getElementById('result').style.display = 'none';
}

async function runAnalysis() {
  const situation = document.getElementById('situation').value.trim();
  const context = document.getElementById('context').value.trim();
  if (!situation) { showError('Please describe the situation first.'); return; }
  const btn = document.getElementById('analyseBtn');
  showSpinner(btn);
  try {
    const r = await fetch('/analyse', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ situation, context: context || null })
    });
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.statusText); }
    showResult(await r.json());
  } catch(e) { showError('Error: ' + e.message); }
  finally { hideSpinner(btn, 'Run Compass'); }
}

async function runAudioAnalysis() {
  const file = document.getElementById('audioFile').files[0];
  if (!file) { showError('Please select an audio file.'); return; }
  const btn = event.target;
  showSpinner(btn);
  const form = new FormData();
  form.append('file', file);
  const context = document.getElementById('context').value.trim();
  if (context) form.append('context', context);
  try {
    const r = await fetch('/analyse/audio', { method: 'POST', body: form });
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.statusText); }
    showResult(await r.json());
  } catch(e) { showError('Error: ' + e.message); }
  finally { hideSpinner(btn, 'Transcribe & Analyse'); }
}

async function runYoutubeAnalysis() {
  const url = document.getElementById('ytUrl').value.trim();
  if (!url) { showError('Please enter a YouTube URL.'); return; }
  const btn = event.target;
  showSpinner(btn);
  const form = new FormData();
  form.append('url', url);
  const context = document.getElementById('context').value.trim();
  if (context) form.append('context', context);
  try {
    const r = await fetch('/analyse/youtube', { method: 'POST', body: form });
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.statusText); }
    showResult(await r.json());
  } catch(e) { showError('Error: ' + e.message); }
  finally { hideSpinner(btn, 'Fetch & Analyse'); }
}
</script>
</body>
</html>"""
