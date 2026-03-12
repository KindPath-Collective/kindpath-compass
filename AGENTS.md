# AI Agent Rules for kindpath-compass

## Session Init Protocol

Before reading code or making changes, run:
```bash
cat ~/.kindpath/HANDOVER.md
python3 ~/.kindpath/kp_memory.py dump --domain gotcha
python3 ~/.kindpath/kp_memory.py dump
```

---

## What This Is

KindPath Compass — a practice aid for compassionate listening.
Python/Flask web app with prompt-driven emotional architecture support.

## Structure

```
app.py              — Flask application entry point
prompts/            — Prompt Markdown files (editable, hot-reloaded)
docs/               — Example sessions and documentation
```

## Operational Commands

- **Install**: `pip install -r requirements.txt`
- **Run**: `python app.py`
- **Venv**: `source venv/bin/activate`

## Rules

- Language must be gentle, non-judgmental, and accessible — not clinical
- Prompts in `prompts/` are the heart of this tool — edit them with care
- No client session data stored beyond the current session
- Follow KindPath doctrine: benevolence, syntropy, sovereignty
- Read `prompts/emotional_architecture.md` before modifying prompt structure

## Security Mandates

- No client data in source control
- All session data is transient — nothing persisted to database
- Prompts are plain Markdown — must remain human-readable and editable
