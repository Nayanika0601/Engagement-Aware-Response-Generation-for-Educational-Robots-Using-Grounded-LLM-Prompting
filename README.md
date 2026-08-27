# Engagement-Aware-Response-Generation-for-Educational-Robots-Using-Grounded-LLM-Prompting


This repository contains the research implementation of a spoken maths tutor that monitors a child with a webcam and uses a locally hosted LLM to generate engagement interventions. Its central design question is how much decision-making authority to give the model: in **Mode A**, Python classifies the engagement event and the LLM only verbalizes it; in **Mode B**, the LLM receives raw sensor state and decides for itself whether an intervention is needed.

Both modes write the same log schema — including the exact prompt sent, the raw model output, and every rejected retry — so the two conditions can be compared offline without re-running sessions.

> [!IMPORTANT]
> The reviewed snapshot runs from a fresh clone, but it is **not publication-ready**. Before release, add `requirements.txt`, `LICENSE`, `CITATION.cff`, and `config/settings.example.json`; resolve the duplicated trigger logic in `engagement_manager.py`; and correct the threshold values printed in the `main.py` console banner. See [Required corrections before publishing](#required-corrections-before-publishing).

## Experimental conditions

| Condition | Name | Information available to the LLM | Intervention authority |
|---|---|---|---|
| **Mode A** | Trigger-labelled | A named trigger plus the sensor block | Python (`main.py` classifies the event) |
| **Mode B** | Autonomous | The sensor block only | LLM (returns `NONE` when no intervention is needed) |
| **Mode B-fail** | Retained negative result | Sensor block **and** leaked trigger names | LLM in name only |

Mode A defines nine trigger prompts:

`no_movement`, `no_answer`, `head_turned`, `face_absent`, `hand_raised_opening`, `hand_raised_followup`, `hand_raised_closing_asked`, `hand_raised_closing_no`, and `hand_raised_closing_silence`.

Each trigger carries a quiz variant, a monitor variant, and a deterministic fallback string used when the model fails or every candidate is rejected.

Mode B replaces all nine with a single prompt per session mode. `config/prompts_mode_b_fail.json` is an earlier Mode B design that named the triggers inside the prompt, which collapsed it into Mode A. It is retained as a documented negative result and is **not** loaded at runtime.

## Repository hierarchy

Arrange the repository as follows. All modules resolve `config/` and `models/` relative to their own file location, so the flat root layout must be preserved.

```text
.
├── README.md
├── requirements.txt                # Add before public release
├── .gitignore                      # Add before public release
├── LICENSE                         # Add before public release
├── CITATION.cff                    # Add/update when publication details are final
├── config/
│   ├── settings.json               # Local file; copy from settings.example.json
│   ├── settings.example.json       # Add before public release
│   ├── prompts_mode_a.json
│   ├── prompts_mode_b.json
│   └── prompts_mode_b_fail.json    # Retained negative result; not loaded at runtime
├── models/
│   ├── face_landmarker.task
│   └── pose_landmarker_lite.task
├── mode_a/                         # Created automatically; normally not committed
├── mode_b1/                        # Created automatically; normally not committed
├── engagement_manager.py           # Currently unused by main.py; see corrections
├── llm_responder.py
├── logger.py
├── mediapipe_processor.py
├── question_bank.py
├── quiz_manager.py
├── speech_input.py
├── speech_output.py
└── main.py                         # Live-system entry point
```

Do not commit `__pycache__/`, `.pyc` files, session logs under `mode_a/` and `mode_b1/`, or `observations_<timestamp>.json` files written by the vision module's standalone mode.

## Requirements

### Software

- **Python 3.10 or newer.** The source uses `X | None` annotation syntax throughout.
- **Ollama** running locally or at a reachable HTTP endpoint.
- The Ollama model **`phi3:mini`**, matching the configured default.
- Git and a terminal.

### Python packages

| Package | Used by | Purpose |
|---|---|---|
| `opencv-python` | `main.py`, `mediapipe_processor.py` | Camera capture, frame overlay, window and key handling |
| `mediapipe` | `mediapipe_processor.py` | Face Landmarker and Pose Landmarker (Tasks API) |
| `numpy` | `speech_input.py` | Audio block handling and RMS gating |
| `scipy` | `speech_input.py` | WAV writing for the Whisper handoff |
| `sounddevice` | `speech_input.py` | Microphone input stream |
| `faster-whisper` | `speech_input.py` | CTranslate2 speech-to-text, `tiny` model, int8 on CPU |
| `requests` | `llm_responder.py` | Ollama HTTP requests |
| `pyttsx3` | `speech_output.py` | Cross-platform text-to-speech fallback |
| `pywin32` | `speech_output.py` | Windows SAPI voices; optional, Windows only |

### Hardware

- A **webcam** is required for `main.py` and `mediapipe_processor.py`.
- A **microphone and speakers** are required for the live spoken tutor.
- `quiz_manager.py`, `logger.py`, and `question_bank.py` self-tests run without any hardware.
- A **GPU is recommended** for responsive local generation. `phi3:mini` runs on CPU, but end-to-end latency will dominate the interaction. Ollama can also be configured on another reachable machine via `ollama_url`.
- **Windows is the primary target.** `speech_output.py` prefers Windows SAPI through `win32com` and falls back to `pyttsx3` on other platforms; the fallback path has known threading limitations (see [Known limitations](#known-limitations)).
- There is **no embedded-hardware dependency**. Despite the "robot" naming, the pipeline uses a generic `cv2.VideoCapture` webcam and has no GPIO, serial, or motor-control layer.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<organization-or-user>/<repository-name>.git
cd <repository-name>
```

### 2. Create and activate a virtual environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Until `requirements.txt` is added, install directly:

```bash
python -m pip install opencv-python mediapipe numpy scipy sounddevice faster-whisper requests pyttsx3
python -m pip install pywin32          # Windows only
```

Pin your versions before archiving. Both `mediapipe` (Tasks API) and `faster-whisper` introduce breaking changes across minor releases:

```bash
python -m pip freeze > requirements-lock.txt
```

### 4. Install Ollama and pull the model

Install Ollama for your operating system, then run:

```bash
ollama pull phi3:mini
ollama serve
```

In another terminal, confirm that the model is available:

```bash
ollama list
```

The code uses Ollama's non-streaming generate endpoint at `http://localhost:11434/api/generate` by default.

### 5. Create the local configuration

macOS or Linux:

```bash
cp config/settings.example.json config/settings.json
```

Windows PowerShell:

```powershell
Copy-Item config/settings.example.json config/settings.json
```

`llm_responder.py`, `engagement_manager.py`, `logger.py`, and `main.py` all read `config/settings.json` at import time. `llm_responder.py` additionally requires `config/prompts_mode_a.json` and `config/prompts_mode_b.json` to exist, regardless of which mode is active.

## Configuration

Edit `config/settings.json`:

```json
{
  "ollama_url": "http://localhost:11434/api/generate",
  "ollama_model": "phi3:mini",
  "llm_timeout_sec": 120,
  "llm_mode": "a",
  "llm_history_length": 3,
  "llm_min_words": 5,
  "llm_max_words": 35,
  "thresholds": {
    "no_movement_sec": 10,
    "no_answer_sec": 15,
    "face_absent_sec": 3,
    "gaze_away_thresh": 0.3
  },
  "cooldown_sec": 15
}
```

| Setting | Purpose |
|---|---|
| `ollama_url` | Full Ollama `/api/generate` endpoint. Use a reachable URL or tunnel when the model runs remotely. |
| `ollama_model` | Ollama model tag. |
| `llm_timeout_sec` | HTTP timeout for one request. On timeout the trigger fallback is spoken and `llm_used` is logged as `false`. |
| `llm_mode` | `a` selects trigger-specific prompts. `b` passes only the sensor block and accepts `NONE` as a valid reply. Also selects the log directory: `mode_a/` or `mode_b1/`. |
| `llm_history_length` | Number of recent responses fed back into the prompt and used for the similarity check. |
| `llm_min_words` | Candidates below this word count are rejected and retried. |
| `llm_max_words` | Candidates above this are trimmed to the last complete sentence, not rejected. |
| `thresholds.no_movement_sec` | Stillness in a 5 s observation window before a `no_movement` alert. |
| `thresholds.no_answer_sec` | Read by `engagement_manager.py` only; `main.py` does not use it. |
| `thresholds.face_absent_sec` | Continuous face absence before a `face_absent` alert. |
| `thresholds.gaze_away_thresh` | Gaze score below this triggers `head_turned`. |
| `cooldown_sec` | Minimum interval before the same trigger can fire again. Tracked per trigger. |

### Constants not read from `settings.json`

These are hard-coded near the top of their modules:

| Constant | File | Default | Effect |
|---|---|---|---|
| `OBSERVATION_INTERVAL_SEC` | `mediapipe_processor.py` | 5.0 | Aggregation window length |
| `NO_MOVEMENT_TIMEOUT_SEC` | `mediapipe_processor.py` | 10.0 | Sets `still_there_prompt` on the observation and drives the overlay warning |
| `CAMERA_INDEX` | `mediapipe_processor.py` | 0 | OpenCV capture device |
| `SHOW_WINDOW` | `mediapipe_processor.py` | `True` | Displays the annotated preview window |
| `HEAD_MOVEMENT_THRESHOLD_DEG` | `mediapipe_processor.py` | 2.0 | Yaw/pitch delta counted as head movement |
| `GAZE_ON_ROBOT_MAX_YAW_DEG` | `mediapipe_processor.py` | 20.0 | Yaw at which the gaze score reaches 0 |
| `MAX_QUESTIONS` | `main.py` | 5 | Questions per session |
| `_HAND_RAISE_SEC` | `main.py` | 3.0 | Sustained hand raise before an interrupt |

Move these into `settings.json` before running condition comparisons, or record them alongside each session.

### Vision models

The task models belong in `models/`. If they are absent, `mediapipe_processor.py` downloads them from Google-hosted MediaPipe `latest` URLs at import time. For exact reproducibility, version the reviewed files or publish their hashes rather than relying on a `latest` URL.

Reviewed model hashes:

```text
face_landmarker.task
SHA-256: 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

pose_landmarker_lite.task
SHA-256: 59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a
```

## Running the live system

Start Ollama, then run from the repository root:

```bash
python main.py
```

Press `Q` or `Esc` in the camera window to exit at any point.

### Session phases

| Phase | Behaviour |
|---|---|
| 1 | 10 s camera warm-up. Reports whether face, hand raise, and head movement were detected, then asks by voice whether to start the quiz. |
| 2+3 | Quiz loop of `MAX_QUESTIONS` questions with the camera thread running. Engagement events interrupt the current question; the question is repeated afterwards with a "Let's get back to the question" preface. |
| 5 | Post-quiz monitoring. Camera and engagement alerts remain active until `Q`. |
| Monitor-only | Entered instead of phases 2–5 if the child declines the quiz in phase 1. |

### Threading model

```
camera thread ──► MediaPipe ──► ObservationBuilder ──► SharedState
  (per frame)     face+pose      (5 s windows)            │
                                                          ▼
                                              interrupt event
                                                          │
main thread ──► QuizManager ──► TTS/STT ◄─────────────────┘
                    │                        │
                    ▼                        ▼
              question_bank            llm_responder ──► Ollama HTTP
                                             │
                                             ▼
                                       SessionLogger ──► session_*.json
```

The camera thread never blocks on speech and the main thread never blocks on frames. They share a `SharedState` object holding the latest observation, a `threading.Event` used as the interrupt signal, and per-trigger cooldown timestamps. `stop_event` is passed into both `speak_async` polling loops and `listen_for_answer`, so a hand raise can cut off speech or recording mid-utterance.

To run condition B, set `"llm_mode": "b"` and restart. The mode is read at import time; changing it mid-session has no effect.

## Run individual components

Each module is independently executable as a self-test.

### Question-bank self-test

```bash
python question_bank.py
```

Expected result: 90 questions, 30 each at levels 1, 2, and 3.

### Quiz-logic self-test

```bash
python quiz_manager.py
```

Seven tests covering answer normalisation, number-word conversion, pool loading, difficulty adjustment, completion, and question resume. No camera, microphone, or speaker required.

### Logger self-test

```bash
python logger.py
```

Four tests covering observation, quiz-result, and engagement-event logging, file write with a computed `pipeline_score`, and the summary printout. Writes to the system temp directory.

### LLM response-layer self-test

```bash
python llm_responder.py
```

Exercises all nine Mode A triggers plus Mode B, including Mode B's `auto` path. This script reports passing results even when Ollama is unavailable, because a spoken fallback counts as a valid response. Confirm `llm_used: true` in the output before treating a run as a real model test.

### Engagement-manager self-test

```bash
python engagement_manager.py
```

Runs against mocked LLM and microphone functions. Note that this module is currently not imported by `main.py`; see [Required corrections](#required-corrections-before-publishing).

### Camera observation only

```bash
python mediapipe_processor.py
```

Press `Q` or `Esc` to stop. Observations are saved as `observations_<timestamp>.json` in the current working directory.

### Speech self-tests

```bash
python speech_output.py     # 4 tests; needs a speaker
python speech_input.py      # 3 tests; needs a microphone
```

`speech_input.py` loads and warms the Whisper model at import time, so expect a delay before the first test runs.

## Test the installation

Validate JSON configuration:

```bash
python -m json.tool config/settings.json
python -m json.tool config/prompts_mode_a.json
python -m json.tool config/prompts_mode_b.json
```

Compile the source tree:

```bash
python -m compileall -q .
```

Run the hardware-free self-tests:

```bash
python question_bank.py
python quiz_manager.py
python logger.py
```

Confirm Ollama directly:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "phi3:mini",
  "prompt": "Reply with the word ready.",
  "stream": false
}'
```

Then run a one-session smoke test with `MAX_QUESTIONS` temporarily set to 1, and confirm the written session file contains engagement events with:

```json
{
  "llm_used": true
}
```

A completed session can consist entirely of deterministic fallbacks, so a clean exit alone does not prove that Ollama generated any response.

## Vision signals

| Field | Derivation |
|---|---|
| `face_detected` | Face landmarks present in more than 50 % of frames in the window |
| `body_detected` | Pose landmarks present in more than 30 % of frames |
| `head_yaw_deg` | Nose offset from the eye midpoint, normalised by eye span, scaled to ±90° |
| `head_pitch_deg` | Nose offset from the frame centre, normalised by face height, scaled to ±30° |
| `gaze_on_robot` | `1 − abs(yaw) / 20°`, clamped to 0 |
| `head_moving` | Yaw or pitch delta above 2° between consecutive frames |
| `hand_raised` | Wrist above shoulder in pose landmarks with visibility above 0.5, in more than 30 % of body frames |
| `hand_raise_side` | Modal side across the raised-hand frames: `left`, `right`, `both`, or `none` |
| `no_movement_sec` | Time since the last frame flagged as head movement |
| `still_there_prompt` | `no_movement_sec >= 10.0` |
| `speech_energy` | Declared on the dataclass and passed to the LLM prompt, but never populated; always `"low"` |

## Answer handling

`quiz_manager.normalise_answer()` strips filler phrases (`the answer is`, `i think it's`, `um`, `uh`, `so`, `well`), then converts number words to digits. It handles bare digits, single words, hyphenated and spaced compounds ("twenty four" → `24`), and hundreds ("three hundred seven" → `307`). The result is matched against the question's `accepted_answers` list.

Difficulty is adaptive: three correct in a row raises the level, three wrong lowers it, bounded to levels 1–3. Level changes are announced aloud.

Each `question_bank.py` entry also carries `common_misconception`, `wrong_answers`, and `expected_response_times` fields. These are documentation for scenario design and are **not** read by the runtime.

## Validation and retry behavior

`llm_responder.generate()` retries up to three times at temperatures `0.7`, `0.9`, and `1.1`. A candidate is rejected when it:

- leaks a sensor field name (`gaze`, `yaw`, `head_pitch`, `speech_energy`, `face_detected`, and related terms);
- refers to the child in the third person (`the student`, `the child`, `they seem`, and related phrases);
- overlaps 60 % or more of its words with any of the last `llm_history_length` responses;
- is empty, under 3 characters, or below `llm_min_words`.

Candidates above `llm_max_words` are trimmed to the last complete sentence rather than rejected. In Mode B, the literal reply `NONE` is accepted and returned as a valid "no intervention needed" decision; `main.py` substitutes a scripted line when `NONE` arrives in a hand-raise dialogue, where a reply is always required.

Every attempt is recorded in `llm_retry_log` with its temperature, rejection reason, and raw output, so rejection rates can be compared across Mode A and Mode B after the fact.

On HTTP error, connection failure, timeout, or three rejected candidates, the trigger's fallback string is spoken and `llm_used` is logged as `false`. A single HTTP 500 is retried once after a 2 s pause before falling back.

## Output JSON files

```text
mode_a/session_<YYYYMMDD_HHMMSS>.json
mode_b1/session_<YYYYMMDD_HHMMSS>.json
```

The directory is selected from `llm_mode` at logger construction. Each file contains:

| Key | Contents |
|---|---|
| `session_id` | `session_<timestamp>_<6 hex chars>` |
| `observations` | Every 5 s vision window |
| `quiz_results` | Per question: answer, correctness, level, face presence, and a full latency breakdown (TTS question, mic listen, Whisper transcribe, TTS feedback, total cycle) |
| `engagement_events` | Trigger, spoken response, `llm_used`, latency, the exact prompt sent, the raw model output, and the per-attempt retry log |
| `summary` | Detection rates, speech capture rate, answer accuracy per level, trigger counts, average latencies, and `pipeline_score` |

The log is rewritten in full after every question and every engagement event, so an interrupted session still leaves a valid file.

> [!WARNING]
> `pipeline_score` is an unweighted mean of four percentages: face detection rate, speech capture rate, answer accuracy, and quiz completion rate. Answer accuracy measures the child; the other three measure the system. Treat it as a session-health indicator, not a research metric, and do not report it as a system-performance figure.

## Known limitations

1. **`gaze_on_robot` is a head-yaw proxy, not gaze.** No iris or eye landmarks feed it. A child looking sideways with their head square to the camera scores 1.0.
2. **`head_pitch_deg` is camera-dependent.** It measures nose offset from the frame centre, so raising or lowering the camera changes reported pitch with no change in head angle. Pitch is not comparable across physical setups.
3. **`speech_energy` is never computed.** It is sent to the LLM in every prompt as the constant `"low"`.
4. **`llm_min_words = 5` rejects valid short replies.** "Nice work, keep going!" is four words and triggers a retry at a higher temperature. This inflates measured retry rates.
5. **TTS lock handling is fragile off Windows.** In the `pyttsx3` path, `speak_async` acquires the shared lock on the calling thread and releases it on a background thread, and `stop()` releases a lock it may not hold while swallowing the resulting `RuntimeError`. Barge-in interruption is unreliable on non-Windows machines.
6. **Import-time side effects.** Importing `speech_input` loads and warms the Whisper model; importing `mediapipe_processor` may download roughly 9 MB of model files. Both make the modules slow to import and awkward to unit test.
7. **Cooldowns are not persisted.** All cooldown state lives in memory and resets between phases only insofar as `SharedState` is reconstructed, so trigger spacing is not comparable across a restart.

## Required corrections before publishing

1. **Remove or wire in `engagement_manager.py`.** `main.py` imports none of it and reimplements trigger detection inline. The two implementations already disagree: `engagement_manager._condition_met` treats `face_absent` as `not face and no_movement_sec >= 3`, while `main.py` uses a dedicated face-absence timer. `no_answer` exists only in the unused module, so the `thresholds.no_answer_sec` setting has no runtime effect.
2. **Correct the console banner in `main.py`.** It prints "Hand raise 5s" and "Face absent 5s"; the actual values are `_HAND_RAISE_SEC = 3.0` and `face_absent_sec = 3`. Anyone reading the console will record the wrong condition.
3. **Consolidate configuration.** Move `MAX_QUESTIONS`, `_HAND_RAISE_SEC`, and the `mediapipe_processor.py` constants into `settings.json`.
4. **Reconcile the two stillness thresholds.** `NO_MOVEMENT_TIMEOUT_SEC` (10.0, vision module, sets `still_there_prompt` and the overlay warning) and `thresholds.no_movement_sec` (10, engagement alert) are numerically equal but independently defined. Decide whether they are one stage or two, then rename or merge them.
5. **Add `requirements.txt` with pinned versions**, plus `requirements-lock.txt` for the environment used to produce reported results.
6. **Add `config/settings.example.json`** and gitignore `config/settings.json`, `mode_a/`, `mode_b1/`, `observations_*.json`, `__pycache__/`, and `*.pyc`.
7. **Either compute `speech_energy` or remove it** from the `Observation` dataclass and the prompt block. A constant field in the prompt is a confound in the Mode B condition, where the model is supposed to reason from sensor state.
8. **Rename `gaze_on_robot`** to reflect that it measures head orientation, or implement iris-based gaze.
9. **Add a license and finalized citation metadata** before describing the repository as open source.

## Privacy and research-data handling

The pipeline processes camera frames, speech transcripts, learner answers, prompts, and response timing. Frames are not written to disk, but transcripts, answers, and any spoken question the child asks during a hand-raise dialogue are stored verbatim in `engagement_events[].student_question`.

De-identify logs before sharing them. Do not publish recordings, identifiable transcripts, or participant-level outputs without the applicable consent and institutional approvals. Keep raw session output under the ignored `mode_a/` and `mode_b1/` directories and commit only de-identified artifacts to a tracked results directory.

## License

No license was included in the reviewed files. Select a license, add the corresponding `LICENSE` file, and replace this section before describing the repository as open source.
