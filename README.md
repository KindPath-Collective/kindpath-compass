# KindPath Compass

A practice aid for compassionate listening in the presence of noise.

Three moves: texture recognition, emotional architecture, ZPB reimagining.

*"May I join you for a while on your journey."*

---

## What It Does

KindPath Compass takes a text transcript, uploaded audio, or YouTube URL and runs three structured analysis prompts against it to surface:

1. **Texture Recognition** — what is actually present beneath the surface presentation
2. **Emotional Architecture** — what structures hold the situation in place
3. **ZPB Reimagining** — what genuine care as governing logic would produce

The result is a KindPath Compass reading: not a recommendation, a diagnosis, or a plan — a clearer view that the practitioner uses to inform their own judgement.

---

## Install

```bash
# Clone and enter
cd kindpath-compass

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: audio/YouTube support (Whisper + yt-dlp)
pip install openai-whisper yt-dlp
```

### Requirements

- Python 3.10+
- `fastapi`, `uvicorn` — API server
- `httpx` — HTTP client (LLM backend calls)
- `python-multipart` — file upload support
- `openai-whisper` *(optional)* — audio transcription
- `yt-dlp` *(optional)* — YouTube audio download

### LLM Backend

The app will use Claude if `ANTHROPIC_API_KEY` is set in your environment, otherwise falls back to Ollama at `http://localhost:11434`.

```bash
# For Claude (recommended)
export ANTHROPIC_API_KEY=sk-ant-...

# For Ollama (offline)
brew install ollama && brew services start ollama
ollama pull llama3.2
```

---

## Run

```bash
source venv/bin/activate
uvicorn app:app --reload --port 8008
```

Open [http://localhost:8008](http://localhost:8008) in a browser.

---

## API Endpoints

| Method | Path | Input | Description |
|--------|------|-------|-------------|
| `GET`  | `/` | — | HTML UI |
| `POST` | `/analyse` | `{"text": "..."}` | Analyse text transcript |
| `POST` | `/analyse/audio` | multipart file | Transcribe then analyse |
| `POST` | `/analyse/youtube` | `{"url": "..."}` | Download, transcribe, analyse |
| `GET`  | `/health` | — | Health check |

### Example: Text analysis

```bash
curl -X POST http://localhost:8008/analyse \
  -H "Content-Type: application/json" \
  -d '{"text": "He came in and sat down and immediately said he was fine."}'
```

Response includes `texture`, `architecture`, `zpb_reimagining`, and `compass_reading` fields.

---

## Prompt Templates

Prompts are in `prompts/`:
- `texture_recognition.md` — Move 1 prompt
- `emotional_architecture.md` — Move 2 prompt  
- `zpb_reimagining.md` — Move 3 prompt

These can be edited to adapt the Compass to different practice contexts.

---

## Worked Example

See [docs/example_mens_shed.md](docs/example_mens_shed.md) for a full worked example: a Men's Shed bereavement scenario run through all three Compass moves.

---

## Methodology

The KindPath Compass methodology is documented in [kindpath-canon](../kindpath-canon/).

The three moves derive from the practice framework: zones of proximity and belonging (ZPB), texture recognition as the entry point for authentic assessment, and emotional architecture as the structural layer beneath presenting behaviour.

---

*KindPath Collective — built for practice, not for product.*
