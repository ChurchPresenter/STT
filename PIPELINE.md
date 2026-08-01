# Pipeline — from microphone to caption

How one second of audio becomes a caption on screen. Every number here is the shipped default
from `config/config.default.json`.

- [Four processes, not one](#four-processes-not-one)
- [Live audio to a saved row](#live-audio-to-a-saved-row)
- [Why a row was hidden](#why-a-row-was-hidden)
- [Delivery](#delivery)
- [Translation](#translation)
- [What flows back](#what-flows-back)
- [On stop](#on-stop)
- [Where each stage lives](#where-each-stage-lives)

---

## Four processes, not one

The most misleading thing about reading the source top to bottom: capture and transcription do
not run in the web server. They run in a separate spawned process, and the two communicate only
through a `multiprocessing.Manager` dict and the session database.

```mermaid
flowchart TD
    WD["<b>Watchdog</b> — supervisor process<br/>crash recovery · auto-update · headless"]
    WEB["<b>P1</b> — Flask + Socket.IO<br/>web pages · translation · TTS<br/><i>never touches audio</i>"]
    WK["<b>P2</b> — Transcription worker<br/>capture · VAD · PANNs · Whisper<br/><i>spawned, not forked</i>"]
    MG[/"<b>P3</b> — Manager dict<br/>29 shared keys: levels, live text, status"/]
    DB[("Session database<br/>SQLite, WAL")]

    WD -->|spawns / restarts| WEB
    WEB -.->|control_queue: start / stop<br/>config_queue: hot reload| WK
    WK -->|writes every row| DB
    DB -->|read back every 0.5 s| WEB
    WK --> MG
    MG --> WEB
```

**Why spawn, not fork.** A forked child inherits the parent's CUDA context and dies on
`Cannot re-initialize CUDA in forked subprocess`, so the start method is forced to `spawn` and
shared objects are passed as pickled arguments instead of inherited.

---

## Live audio to a saved row

All inside the worker process. Dotted arrows are influences rather than data hand-offs.

```mermaid
flowchart TD
    MIC(["🎙 Microphone"])
    FF["<b>ffmpeg capture</b><br/>16 kHz mono · 1.0 s chunk = 32000 B<br/>10 s stall → restart"]
    PAN["<b>PANNs tagger</b> — own thread<br/>every 0.4 s · smoothing window 4<br/>Speaking / Music / Quiet"]
    BUF["<b>Rolling buffer</b><br/>never drained on a timer<br/>clip 45 s → drop oldest 30 s"]
    VAD{"<b>Speech present?</b><br/>energy 100, then Silero 0.5<br/>exception → fails open"}
    HOLD["Hold the audio<br/>force finalise at 2 s silence"]
    WSP["<b>Whisper</b> — faster-whisper CT2<br/>beam 3 · temp 0.0 forced<br/>prompt = last 200 chars"]
    P2["<b>Whisper pass 2</b><br/><i>whisper_translate modes only</i><br/>decodes straight to the target language"]
    FIN{"<b>Stopped changing?</b><br/>similarity 0.85 · 7 agreeing passes"}
    LIVE["<b>Live line → screen</b><br/>LocalAgreement-2<br/><i>never saved as a row</i>"]
    SPL["<b>Split into sentences</b><br/>fragment held back<br/>release at 30 words or 10 s"]
    FLT{"<b>Six filters</b><br/>CJK → hallucination → music<br/>→ profanity → short → duplicate"}
    REJ["<b>Rejected row</b><br/>denied = 1, kept with a reason"]
    DB[("<b>Session database</b><br/>final rows + partials every 1000 ms")]

    MIC --> FF
    FF --> BUF
    FF --> PAN
    PAN -.->|music_prob &gt; 0.5<br/>overrides the gate| VAD
    BUF --> VAD
    VAD -->|no| HOLD
    HOLD -.-> BUF
    VAD -->|yes| WSP
    WSP --> FIN
    WSP --> P2
    FIN --> LIVE
    FIN --> SPL
    SPL --> FLT
    FLT -->|pass| DB
    FLT -->|fail| REJ
    REJ --> DB
    P2 --> DB
```

**Music always wins.** A confident music reading forces the gate to accept, so singing is always
transcribed; `transcribe_detected_music` decides only whether the row is *shown*.

---

## Why a row was hidden

Nothing is deleted. A rejected row is written with a reason so the corrections page can restore
it. `denied_reason` is the complete vocabulary — the first rule to fire wins.

| Order | `denied_reason` | Fires when |
|-------|-----------------|------------|
| 1 | `cjk` | CJK characters in a non-CJK session — a classic Whisper artefact |
| 1b | `cjk_shadow` | Partial strip: the cleaned text is saved *and* the original kept beside it |
| 2 | `hallucination` | Substring match against known artefact stems (subtitle credits, "thank you for watching") |
| 3 | `music:0.5` | Tagged Music while `transcribe_detected_music` is off. The threshold is baked into the reason so the corrections page can compare against the row's own value |
| 4 | *(none)* | Profanity filter — rewrites the text in place and keeps the verbatim original. Not a rejection |
| 5 | `short` | Fewer words than `min_words`. Off by default |
| 6 | `dup` | Fuzzy match ≥ 0.85 against anything already saved this session |

---

## Delivery

Text is only one of three things leaving the worker. Raw audio is forwarded on its own bounded
queue, and levels and status travel through the shared-state proxy without touching the database.

```mermaid
flowchart TD
    subgraph WORKER["P2 · worker outputs"]
        R1["Transcript rows<br/><i>via the database</i>"]
        R2["Raw audio<br/><i>queue max 10, drops when full</i>"]
        R3["Levels &amp; status<br/><i>shared-state proxy</i>"]
    end

    subgraph WEBP["P1 · emit loops, every 0.5 s"]
        L1["Transcript loop"]
        L2["Translation loop<br/>max 3 fresh / cycle<br/>newest 2 + oldest 1"]
        L3["Audio loop<br/>PCM passthrough"]
        L4["TTS loop<br/>flush after 4.0 s"]
        SP["Service phase<br/>own tick, every 20 s"]
        DLY{"Output delay?<br/>off, or 2–30 s<br/>default 7 s"}
        SIO["Socket.IO broadcast<br/>rooms: audio_stream · tts_audio"]
    end

    subgraph CLIENT["Browser / OBS"]
        C1["Caption screen<br/>transcribe · translate · both"]
        C2["OBS browser source<br/>the same page, as a URL"]
        C3["🔊 Audio monitor<br/><i>opt-in</i>"]
        C4["🗣 Spoken translation<br/>queue &gt; 5 → keep newest 3"]
    end

    R1 --> L1
    R1 --> L2
    R2 --> L3
    R3 --> L1
    R1 --> SP
    L1 --> DLY
    L2 --> DLY
    DLY --> SIO
    L3 --> SIO
    L4 --> SIO
    SP --> SIO
    SIO --> C1
    SIO --> C2
    SIO --> C3
    SIO --> C4
```

**Output delay** holds captions so an operator can correct a row before the audience ever sees
it; released rows are back-dated to their real time. **Rooms** mean audio and speech only reach
clients that explicitly joined, so a caption screen is never sent audio it will not play.

---

## Translation

Four paths, one dispatcher. Declining is an ordinary outcome, not an error — a wrong caption in
front of a congregation is worse than a slower one.

```mermaid
flowchart LR
    IN(["A finalised row"])
    RM{"<b>Remote</b><br/>paired machine over HTTP<br/>timeout 15 s · cap 8000 chars"}
    LM{"<b>LLM</b><br/>local GGUF or OpenAI-compatible<br/>temp 0.0 · n_ctx 2048"}
    NM["<b>NMT</b><br/>NLLB or MADLAD<br/>truncates at 1024 tokens · beams 2"]
    OUT(["Caption"])

    IN --> RM
    RM -->|success| OUT
    RM -->|unreachable<br/>fallback = skip| SRC(["Source text, untranslated"])
    RM -->|unreachable<br/>fallback = local| LM
    LM -->|accepted| OUT
    LM -->|declines validation| NM
    NM --> OUT
```

**The fourth path skips all three.** In `whisper_translate` modes the translation is produced in
the worker by decoding the audio a second time, so it never reaches this dispatcher — the web
process only reads the result.

**The LLM's answer is thrown away when:**

| Rejected when | Because |
|---------------|---------|
| It refuses — *"I can't assist…"*, *"As an AI…"* | A refusal is not a translation |
| It narrates — *"Okay, let's…"*, *"Here's the translation:"* | Reasoning models ignore instructions to stop |
| Cyrillic survives into a Latin-script target | The source leaked through untranslated |
| The output is multi-paragraph | A caption is one utterance; a document is a recitation |
| It runs past `min(3 × source words, source + 24)` | Commentary and recited scripture present as length |
| The input would not fit the context window | Checked *before* generating, so an overflow is a decline and never a crash |

Each caption is an independent two-message call — nothing accumulates between sentences, so
there is no context to clear.

---

## What flows back

The diagrams above are one-way; the system is not.

```mermaid
flowchart TD
    OP1["Start / stop<br/>and device selection"]
    OP2["Settings change"]
    OP3["Correct a caption"]
    WEB["<b>P1</b> — queues and write-back<br/>rewrites the row, drops it from the<br/>translation cache, re-broadcasts"]
    WK["<b>P2</b> — applies mid-session<br/><i>no restart, no dropped audio</i>"]
    SCR["Every connected screen"]

    OP1 -.-> WEB
    OP2 -.-> WEB
    OP3 -.-> WEB
    WEB -.->|control_queue| WK
    WEB -.->|config_queue| WK
    WEB -->|re-emit| SCR
```

---

## On stop

```mermaid
flowchart LR
    S(["Session ends"])
    S --> A[".srt + .translated.srt<br/>both languages"]
    S --> B[".html transcript<br/>with word highlighting"]
    S --> C[".ts during · .wav at stop"]
    A --> W["Retire the write-ahead log"]
    B --> W
    C --> W
    W --> M["SMB / NAS move<br/>after a 10 s pause for handles"]
```

The write-ahead log is retired *before* anything is moved, so what leaves the machine is a single
portable file rather than a database plus sidecars.

---

## Where each stage lives

| Stage | Reference |
|-------|-----------|
| Process split — why spawn, not fork | `speech_to_text.py:2986-3000` |
| Shared state proxy (29 keys) | `speech_to_text.py:3023` |
| ffmpeg capture loop | `stt/audio_capture.py:151` · `:284` |
| Rolling buffer / clip rule | `speech_to_text.py:2706-2723` |
| VAD + music bypass | `speech_to_text.py:16266` · `:16901` |
| PANNs detector thread | `speech_to_text.py:3354-3432` · `stt/segments.py:8` |
| Whisper decode params | `speech_to_text.py:316` · `:17013` |
| Finalisation rule | `speech_to_text.py:2758-2918` |
| Live-line stabiliser | `stt/hypothesis_buffer.py:34` |
| Sentence split + pending buffer | `stt/text_utils.py:152` · `speech_to_text.py:17180` |
| Filters | `speech_to_text.py:17214-17244` |
| Row insert + partials | `speech_to_text.py:17208` · `:17670` |
| Emit loops | `speech_to_text.py:14243` · `:14826` |
| Translation dispatcher | `speech_to_text.py:14735` |
| LLM validation | `stt/llm_translate.py:335` |
| Service phase detector | `stt/service_phase.py` · `speech_to_text.py:14184` |
| Teardown / exports | `speech_to_text.py:17955-18030` · `stt/file_mover.py:445` |

---

Palette and typography for the UI these stages feed are documented in [DESIGN.md](DESIGN.md).
Installation and system requirements are in [INSTALL.md](INSTALL.md).
