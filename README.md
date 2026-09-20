# AI Mitra 

> **Your Private, Automated AI News Curator**  
> Delivered daily to your mobile phone via Telegram at **8:00 AM IST** with concise text briefings, audio voice notes, on-demand village analogies, and evidence-grounded practical tool quickstarts.

---

## Table of Contents
1. [Overview & Core Philosophy](#overview--core-philosophy)
2. [Key Capabilities & Two News Tracks](#key-capabilities--two-news-tracks)
3. [Security Architecture & Guardrails](#security-architecture--guardrails)
4. [Project Structure](#project-structure)
5. [Prerequisites & Local Environment Setup](#prerequisites--local-environment-setup)
6. [Credentials Setup Guide](#credentials-setup-guide)
7. [Running Automated Tests](#running-automated-tests)
8. [Starting the Application](#starting-the-application)
9. [User Interaction & Command Reference](#user-interaction--command-reference)
10. [Troubleshooting & Diagnostics](#troubleshooting--diagnostics)

---

## Overview & Core Philosophy

In a world filled with sensationalized headlines and AI hype, **AI Mitra** is engineered under a strict **Zero-Blind-Trust** doctrine:

1. **Tri-Partite Content Separation**: AI Mitra strictly differentiates:
   - **Source Facts**: What the primary/official documentation actually stated.
   - **AI Summary**: Neutral, objective synthesis of the development.
   - **AI Mitra's Interpretation**: Everyday village analogies and practical advice, clearly labeled so metaphors are never mistaken for factual claims.
2. **Four Categorical Reliability States**:
   - `🛡️ [Verified Official]`: Confirmed against retrieved candidate primary sources (official blogs, repos, release notes).
   - `🔍 [Cross-Checked]`: Independently confirmed by multiple non-syndicated secondary sources.
   - `📰 [Reported]`: Single-source secondary report awaiting primary confirmation.
   - `⚠️ [Uncertain / Disputed]`: Differing dates, pricing, or conflicting claims explicitly preserved.
3. **Village-Level Analogies**: Translates complex technical AI mechanics (weights, tokens, architectures) into relatable metaphors (farming, cooking, water pumps, village workshops).
4. **Deterministic Metadata Binding**: Model outputs never invent, rewrite, or guess source URLs, canonical publisher names, or reliability states; Python pulls them directly from verified pipeline data.

---

## Key Capabilities & Two News Tracks

- **Track 1: Global AI Developments**: Breakthroughs, frontier foundation models, safety findings, and industry shifts.
- **Track 2: Practical AI Tools & Repos**: Open-source releases, desktop tools, developer libraries, and practical use-cases.
- **8:00 AM IST Scheduled Delivery**: Daily morning broadcast with concise text items and a 60–90 second `.mp3` audio voice note.
- **Non-Blocking Reliability**: If audio synthesis encounters an issue, the text digest is still delivered. If delivery to one user encounters an error, dispatch continues to remaining authorized users.
- **On-Demand Explanations**: Deep-dive breakdowns for any story (`/explain <number>`).
- **Evidence-Based Tool Guides**: Practical quickstarts (`/howto <tool>`) grounded strictly in retrieved documentation text.
- **Safe Link Analysis**: Paste or forward any external news article URL for SSRF-protected extraction and a verified breakdown.
- **Conversational Learning**: Ask general AI and technical questions with user-scoped conversation history stored in SQLite.

---

## Security Architecture & Guardrails

AI Mitra is built with defense-in-depth security principles:

```text
Incoming Telegram Update
  ↓
1. Private-Chat Gate (chat.type == "private", non-private chats dropped)
  ↓
2. Allowlist Check (user.id in ALLOWED_TELEGRAM_USER_IDS)
  ↓
3. Rate Limiter (Token-bucket, 10 requests/minute per user)
  ↓
4. Input-Size Limit (User messages <= 4,000 characters)
  ↓
5. Protected Command / Message Routing
  ↓
6. Gemini / Database / News Pipeline Execution
```

- **Fail-Closed Startup**: If `TELEGRAM_BOT_TOKEN`, `GEMINI_API_KEY`, or `ALLOWED_TELEGRAM_USER_IDS` are missing or set to placeholder text, startup immediately halts.
- **Complete Pre-Execution Isolation**: Unauthorized users are rejected at the middleware level; they never reach Gemini, SQLite database queries, news fetchers, or content extractors.
- **Dual-Stack IPv4 & IPv6 SSRF Protection**:
  - Only `http` and `https` allowed.
  - Resolves hostnames and validates all resolved IPv4 and IPv6 addresses against private, loopback, link-local, carrier-grade NAT, and cloud metadata ranges before connecting.
  - DNS-pinning prevents Time-of-Check to Time-of-Use (TOCTOU) rebinding.
  - Re-validates relative and absolute HTTP redirects through the full SSRF pipeline.
  - TLS certificate verification is enforced (`verify=True`).
- **Prompt Injection Defense**: All external content (RSS feeds, articles, user URLs) is wrapped in `<untrusted_external_content source="...">` markers with attribute escaping.
- **Secret Redaction in Logs**: A custom formatter automatically scrubs bot tokens, API keys, and authorization headers from console and log output.

---

## Project Structure

```text
Ai_agent/
├── .gitignore                      # Excludes .env, data/, *.db, audio, logs
├── .env.example                    # Environment variable template
├── README.md                       # Operational manual and documentation
├── requirements.txt                # Pinned dependencies for Python 3.11+ / 3.13
├── config.py                       # Settings, timezone, limits, fail-closed validation
├── run.py                          # Main application entry point & lifecycle manager
│
├── core/
│   ├── models.py                   # Pydantic data models (NewsFactSheet, SourceFact, etc.)
│   ├── logger.py                   # Secret-redacting logging formatter
│   ├── security.py                 # SSRF shielding, DNS-pinning, input limits, XML wrapping
│   ├── database.py                 # SQLite storage with strict user-scoping in data/
│   │
│   ├── news/
│   │   ├── fetcher.py              # Multi-source RSS and web aggregator
│   │   ├── content_extractor.py    # Safe article content extraction (SSRF-protected)
│   │   ├── source_resolver.py      # Candidate official source resolver and claim verifier
│   │   ├── dedup_crosscheck.py     # Event-level deduplication and conflict detection
│   │   └── classifier.py           # Assigns reliability states (official/cross-checked/etc.)
│   │
│   └── synthesis/
│       ├── explainer.py            # Structured digest generator & deterministic metadata binding
│       ├── tutor.py                # Village-level analogies & evidence-based how-to guides
│       └── voice.py                # Audio voice note generator (gTTS)
│
├── bot/
│   ├── __init__.py                 # Bot package exports
│   ├── middleware.py               # Security gate, private-chat check, rate limiting
│   ├── handlers.py                 # Telegram command and message handlers
│   ├── formatters.py               # Markdown formatters, badges, and safe message chunking
│   └── scheduler.py                # AsyncIOScheduler integration & isolated daily dispatch
│
└── tests/
    ├── test_security.py            # Phase 1 & 2 security, SSRF, DNS-pinning, allowlist tests
    ├── test_reliability.py         # Phase 3 deduplication, official source, conflict tests
    ├── test_synthesis.py           # Phase 4 structured output, fallback, tutor, voice tests
    ├── test_bot.py                 # Phase 5 middleware gate, commands, chunking, scheduler tests
    └── test_e2e_readiness.py       # Phase 6 non-destructive end-to-end wiring readiness test
```

---

## Prerequisites & Local Environment Setup

### 1. Python Environment
- Python **3.11** or **3.13** (tested with Python 3.13.5).
- Operating System: Windows, macOS, or Linux.

```bash
# Clone or navigate to the project directory
cd Ai_agent

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## Credentials Setup Guide

Copy the environment template:
```bash
cp .env.example .env
```

Open `.env` and configure the following three mandatory settings:

### 1. Google Gemini API Key (`GEMINI_API_KEY`)
1. Visit [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click **Get API key** and create a new key.
4. Copy and paste the key into `.env`:
   ```env
   GEMINI_API_KEY=AIzaSyYourActualApiKeyHere
   ```

### 2. Telegram Bot Token (`TELEGRAM_BOT_TOKEN`)
1. Open Telegram on your phone or desktop and search for `@BotFather`.
2. Send `/start` then `/newbot`.
3. Follow the prompts to choose a bot name (e.g., `My AI Mitra`) and username (e.g., `my_ai_mitra_bot`).
4. `@BotFather` will reply with a token formatted like `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZaBcDeFg`.
5. Paste the token into `.env`:
   ```env
   TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZaBcDeFg
   ```

### 3. Allowed Telegram User IDs (`ALLOWED_TELEGRAM_USER_IDS`)
To ensure your bot only interacts with you and authorized family/team members:
1. Open Telegram and search for `@userinfobot` (or `@raw_data_bot`).
2. Send `/start`. The bot will reply with your numeric `Id` (e.g., `987654321`).
3. Paste your numeric ID into `.env` (multiple IDs can be separated by commas):
   ```env
   ALLOWED_TELEGRAM_USER_IDS=987654321
   ```

### 4. Optional Schedule & Regional Settings
```env
# Timezone for daily morning delivery (Default: Asia/Kolkata)
TIMEZONE=Asia/Kolkata

# Dispatch time in 24-hour HH:MM format (Default: 08:00)
SCHEDULE_TIME=08:00

# SQLite storage path
DATABASE_PATH=data/ai_mitra.db

# Logging level
LOG_LEVEL=INFO
```

---

## Running Automated Tests

Run the complete 48-test automated suite across security, reliability, synthesis, bot routing, and end-to-end readiness:

```bash
python -m pytest tests/ -v
```

Expected output:
```text
tests/test_bot.py (14 tests) ............. PASSED
tests/test_e2e_readiness.py (1 test) ..... PASSED
tests/test_reliability.py (9 tests) ...... PASSED
tests/test_security.py (18 tests) ........ PASSED
tests/test_synthesis.py (6 tests) ........ PASSED
============================= 48 passed in 2.17s =============================
```

---

## Starting the Application

Launch the AI Mitra daemon:

```bash
python run.py
```

### Startup Flow:
1. **Security Validation**: Validates `.env` credentials fail-closed.
2. **Database Initialization**: Prepares `data/ai_mitra.db` schema.
3. **AI & Pipeline Wiring**: Instantiates fetcher, classifier, explainer, tutor, and voice engines.
4. **Scheduler Start**: Registers daily morning dispatch for `08:00` in `Asia/Kolkata`.
5. **Polling Loop**: Begins listening for Telegram messages from authorized user IDs.

---

## User Interaction & Command Reference

Interact with your bot in Telegram via private chat:

| Command / Action | Description | Example |
|---|---|---|
| `/start` | Introduces AI Mitra, explains the two tracks, and lists available commands. | `/start` |
| `/today` | Generates and sends today's verified morning intelligence digest and voice note immediately. | `/today` |
| `/explain <number>` | Returns a plain-English, village-level analogy breakdown of story `<number>` from today's digest. | `/explain 1` |
| `/howto <tool>` | Returns an evidence-based quickstart guide for a specified AI tool or repo. | `/howto ollama` |
| **Send a News URL** | Paste any web article link; AI Mitra validates SSRF safety, extracts the content, and provides a verified summary. | `https://techcrunch.com/...` |
| **Ask Any AI Question** | Ask any technical or conceptual question; AI Mitra provides a grounded explanation with village analogies. | `How does attention mechanism work?` |

---

## Troubleshooting & Diagnostics

- **Bot does not start (`[FATAL SECURITY CONFIGURATION ERROR]`)**:
  - Check that `.env` exists and does not contain placeholder values like `your_gemini_api_key_here`.
  - Ensure `ALLOWED_TELEGRAM_USER_IDS` is a valid positive integer.
- **Bot does not reply to messages**:
  - Verify your Telegram numeric user ID matches `ALLOWED_TELEGRAM_USER_IDS`. Messages from non-allowlisted users are rejected.
  - Verify you are chatting in a private 1-on-1 chat; group chats are rejected by design.
- **Rate limit notice (`⏳ Rate limit reached`)**:
  - You have sent more than 10 requests within a single minute. Wait 60 seconds for tokens to replenish.
- **Link blocked notice (`⛔ Blocked Link`)**:
  - The submitted URL resolves to a private, loopback (`127.0.0.1`, `localhost`), or link-local address. AI Mitra blocks non-public URLs to prevent SSRF vulnerabilities.
- **Audio note not received**:
  - If gTTS experiences a transient network timeout, the text digest will still be delivered successfully. Audio dispatch logs warnings without interrupting text delivery.

