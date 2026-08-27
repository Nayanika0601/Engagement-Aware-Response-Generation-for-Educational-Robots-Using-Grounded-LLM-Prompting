
import time
import re
import json
import os
import requests

_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

with open(os.path.join(_CFG_DIR, "settings.json"), "r") as f:
    _settings = json.load(f)

with open(os.path.join(_CFG_DIR, "prompts_mode_a.json"), "r") as f:
    _prompts_a = json.load(f)

with open(os.path.join(_CFG_DIR, "prompts_mode_b.json"), "r") as f:
    _prompts_b = json.load(f)

OLLAMA_URL     = _settings["ollama_url"]
OLLAMA_MODEL   = _settings["ollama_model"]
OLLAMA_TIMEOUT = _settings["llm_timeout_sec"]
LLM_MODE       = _settings["llm_mode"]
HISTORY_LENGTH = _settings["llm_history_length"]
MIN_WORDS      = _settings["llm_min_words"]
MAX_WORDS      = _settings["llm_max_words"]

_history: list[str] = []


def _add_to_history(text: str) -> None:
    _history.append(text)
    while len(_history) > HISTORY_LENGTH:
        _history.pop(0)


def _history_block() -> str:
    if not _history:
        return ""
    lines = "\n".join(f"- {h}" for h in _history)
    return (
        "Recent responses you have already given (do not repeat or paraphrase these — say something completely different):\n"
        f"{lines}\n\n"
    )


def _build_obs_block(obs: dict) -> str:
    face_detected = obs.get("face_detected", False)
    face = "visible" if face_detected else "not visible"
    body = "detected" if obs.get("body_detected", False) else "not detected"

    if face_detected:
        head = "moving" if obs.get("head_moving", False) else "still"
        gaze = f"{obs.get('gaze_on_robot', 0.0):.2f}"
        yaw  = f"{obs.get('head_yaw_deg', 0.0):.1f}"
        pitch = f"{obs.get('head_pitch_deg', 0.0):.1f}"
        no_mov = f"{obs.get('no_movement_sec', 0.0):.0f}"
        hand = "yes" if obs.get("hand_raised", False) else "no"
        hand_side = obs.get("hand_raise_side", "none")
    else:
        head = "N/A (face not visible)"
        gaze = "N/A"
        yaw  = "N/A"
        pitch = "N/A"
        no_mov = "N/A"
        hand = "N/A"
        hand_side = "N/A"

    energy = obs.get("speech_energy", "low")
    prompted = "yes" if obs.get("already_prompted", False) else "no"

    return (
        "Student state right now:\n"
        f"- Face: {face}\n"
        f"- Body: {body}\n"
        f"- Head: {head}\n"
        f"- Gaze on robot: {gaze} out of 1.0\n"
        f"- Head yaw: {yaw} degrees\n"
        f"- Head pitch: {pitch} degrees\n"
        f"- No movement for: {no_mov} seconds\n"
        f"- Speech energy: {energy}\n"
        f"- Hand raised: {hand}\n"
        f"- Already prompted this window: {prompted}\n"
        f"- Hand raise side: {hand_side}\n"
    )


def _build_prompt_a(obs: dict, trigger: str, extra_context: str | None,
                    mode: str) -> tuple[str, str, str]:
    sys_key = "system_prompt_quiz" if mode == "quiz" else "system_prompt_monitor"
    system = _prompts_a[sys_key]

    trig_data = _prompts_a["triggers"].get(trigger, {})
    prompt_key = "quiz" if mode == "quiz" else "monitor"
    situation = trig_data.get(prompt_key, "Say something encouraging and kind. Respond in between 5 and 25 words.")
    if "{student_question}" in situation and extra_context:
        situation = situation.replace("{student_question}", extra_context)

    fallback_key = "fallback_quiz" if mode == "quiz" else "fallback_monitor"
    fallback = trig_data.get(fallback_key, "I'm here with you!")

    obs_block = _build_obs_block(obs)
    history = _history_block()

    prompt = f"{obs_block}\n{history}{situation}"
    return system, prompt, fallback


def _build_prompt_b(obs: dict, trigger: str, extra_context: str | None,
                    mode: str) -> tuple[str, str, str]:
    sys_key = "system_prompt_quiz" if mode == "quiz" else "system_prompt_monitor"
    system = _prompts_b[sys_key]

    prompt_key = "prompt_quiz" if mode == "quiz" else "prompt_monitor"
    situation = _prompts_b[prompt_key]
    fallback_key = "fallback_quiz" if mode == "quiz" else "fallback_monitor"
    fallback = _prompts_b[fallback_key]

    obs_block = _build_obs_block(obs)

    if extra_context:
        obs_block += f"- Student just said: {extra_context}\n"

    history = _history_block()

    prompt = f"{obs_block}\n{history}{situation}"
    return system, prompt, fallback


def _clean_text(text: str) -> str:
    text = text.split("\n")[0].strip()
    text = text.strip('"\'').strip()
    if ":" in text and text.index(":") < 15:
        text = text.split(":", 1)[1].strip()
    text = text.strip('"\'').strip()
    if text.startswith("- "):
        text = text[2:]
    text = re.sub(r"['\u2018\u2019\u0092]", "'", text)
    text = re.sub(r"Let'se", "Let's e", text)
    text = re.sub(r"Let'self", "Let's", text)
    text = re.sub(r"(\w)'s([a-z])", r"\1's \2", text)
    return text


def _is_invalid_response(text: str) -> bool:
    lower = text.lower()

    SENSOR_FIELDS = [
        "gaze", "pitch", "no_movement", "yaw",
        "face_detected", "body_detected", "head_yaw",
        "head_pitch", "speech_energy", "hand_raise_side",
        "sensor", "detected by", "not detected",
        "heart rate", "body is not", "head is not",
    ]

    THIRD_PERSON = [
        "their gaze", "the student", "the child",
        "they have been", "they are", "they seem",
        "student might", "child might", "student needs",
        "child needs", "the learner", "this child",
        "student appears", "child appears",
        "student is", "child is", "student has",
        "child has", "student seems", "child seems",
        "student was", "child was", "student looks",
        "child looks", "student could", "child could",
    ]

    for field in SENSOR_FIELDS:
        if field in lower:
            print(f"  [llm] Rejected — sensor field '{field}' leaked in response")
            return True

    for phrase in THIRD_PERSON:
        if phrase in lower:
            print(f"  [llm] Rejected — third person reference '{phrase}' in response")
            return True

    return False


def _is_too_similar(text: str) -> bool:
    if not _history:
        return False

    new_words = set(text.lower().split())
    if not new_words:
        return False

    for prev in _history:
        prev_words = set(prev.lower().split())
        if not prev_words:
            continue
        overlap = new_words & prev_words
        smaller = min(len(new_words), len(prev_words))
        ratio = len(overlap) / smaller
        if ratio >= 0.6:
            print(f"  [llm] Rejected — too similar to previous response "
                  f"({ratio:.0%} word overlap)")
            return True

    return False


def generate(obs: dict, trigger: str,
             extra_context: str | None = None,
             mode: str = "quiz") -> dict:

    active_mode = LLM_MODE

    if active_mode == "b":
        system, prompt, fallback = _build_prompt_b(obs, trigger, extra_context, mode)
    else:
        system, prompt, fallback = _build_prompt_a(obs, trigger, extra_context, mode)

    llm_prompt_sent = prompt
    llm_raw_response = ""

    MAX_RETRIES = 3
    temperatures = [0.7, 0.9, 1.1]
    retry_log = []

    try:
        t0 = time.time()

        for retry in range(MAX_RETRIES):
            temp = temperatures[retry]
            if retry > 0:
                print(f"  [llm] Retry {retry}/{MAX_RETRIES - 1} "
                      f"(temperature={temp})")

            payload = {
                "model": OLLAMA_MODEL,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temp,
                    "num_predict": 60,
                },
            }

            resp = None
            for attempt in range(2):
                resp = requests.post(OLLAMA_URL, json=payload,
                                     timeout=OLLAMA_TIMEOUT)
                if resp.status_code == 200:
                    break
                if resp.status_code == 500 and attempt == 0:
                    print(f"  [llm] HTTP 500, retrying in 2s ...")
                    time.sleep(2)
                    continue
                break

            latency = round(time.time() - t0, 3)

            if resp.status_code != 200:
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": f"http_{resp.status_code}",
                    "raw_response": "",
                })
                print(f"  [llm] HTTP {resp.status_code} after retry, using fallback")
                return {"response_text": fallback, "llm_used": False,
                        "llm_latency_sec": latency,
                        "llm_prompt_sent": llm_prompt_sent,
                        "llm_raw_response": llm_raw_response,
                        "llm_retry_log": retry_log}

            data = resp.json()
            llm_raw_response = data.get("response", "").strip()
            text = _clean_text(llm_raw_response)

            if not text or len(text) < 3:
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": "empty_or_too_short",
                    "raw_response": llm_raw_response,
                })
                print(f"  [llm] Empty/too-short response, retrying...")
                continue

            if _is_invalid_response(text):
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": "invalid_response (sensor_leak or third_person)",
                    "raw_response": llm_raw_response,
                })
                continue

            if _is_too_similar(text):
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": "too_similar_to_previous",
                    "raw_response": llm_raw_response,
                })
                continue

            if text.strip().upper() == "NONE":
                print(f"  [llm] LLM said NONE — nothing needs attention ({latency:.3f}s)")
                return {"response_text": "NONE", "llm_used": True,
                        "llm_latency_sec": latency,
                        "llm_prompt_sent": llm_prompt_sent,
                        "llm_raw_response": llm_raw_response,
                        "llm_retry_log": retry_log}

            words = text.split()
            if len(words) < MIN_WORDS:
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": f"too_few_words ({len(words)})",
                    "raw_response": llm_raw_response,
                })
                print(f"  [llm] Too few words ({len(words)}), retrying...")
                continue
            if len(words) > MAX_WORDS:
                import re as _re
                sentences = _re.split(r'(?<=[.!?])\s+', text)
                kept = ""
                for sent in sentences:
                    candidate = (kept + " " + sent).strip() if kept else sent
                    if len(candidate.split()) <= MAX_WORDS:
                        kept = candidate
                    else:
                        break
                if kept:
                    text = kept
                    print(f"  [llm] Trimmed to last complete sentence "
                          f"({len(text.split())} words)")

            _add_to_history(text)

            retry_log.append({
                "attempt": retry + 1,
                "temperature": temp,
                "rejection_reason": None,
                "raw_response": llm_raw_response,
            })

            print(f"  [llm] Generated ({latency:.3f}s, mode={active_mode}): "
                  f"'{text}'")
            return {"response_text": text, "llm_used": True,
                    "llm_latency_sec": latency,
                    "llm_prompt_sent": llm_prompt_sent,
                    "llm_raw_response": llm_raw_response,
                    "llm_retry_log": retry_log}

        latency = round(time.time() - t0, 3)
        print(f"  [llm] All {MAX_RETRIES} attempts failed, using fallback")
        return {"response_text": fallback, "llm_used": False,
                "llm_latency_sec": latency,
                "llm_prompt_sent": llm_prompt_sent,
                "llm_raw_response": llm_raw_response,
                "llm_retry_log": retry_log}

    except requests.exceptions.ConnectionError:
        print("  [llm] Ollama not running, using fallback")
        return {"response_text": fallback, "llm_used": False,
                "llm_latency_sec": 0.0,
                "llm_prompt_sent": llm_prompt_sent,
                "llm_raw_response": llm_raw_response,
                "llm_retry_log": retry_log}

    except requests.exceptions.Timeout:
        print(f"  [llm] Timeout ({OLLAMA_TIMEOUT}s), using fallback")
        return {"response_text": fallback, "llm_used": False,
                "llm_latency_sec": OLLAMA_TIMEOUT,
                "llm_prompt_sent": llm_prompt_sent,
                "llm_raw_response": llm_raw_response,
                "llm_retry_log": retry_log}

    except Exception as e:
        print(f"  [llm] Error: {e}, using fallback")
        return {"response_text": fallback, "llm_used": False,
                "llm_latency_sec": 0.0,
                "llm_prompt_sent": llm_prompt_sent,
                "llm_raw_response": llm_raw_response,
                "llm_retry_log": retry_log}


if __name__ == "__main__":
    print("=" * 60)
    print("  llm_responder.py — self-test (all 9 triggers, Mode A + B)")
    print("=" * 60)

    sample_obs = {
        "face_detected": True,
        "gaze_on_robot": 0.2,
        "head_yaw_deg": -15.3,
        "head_pitch_deg": -30.0,
        "head_moving": False,
        "body_detected": True,
        "no_movement_sec": 12.5,
        "speech_energy": "low",
        "hand_raised": False,
        "already_prompted": False,
        "hand_raise_side": "none",
    }

    triggers = [
        ("no_movement", None),
        ("no_answer", None),
        ("head_turned", None),
        ("face_absent", None),
        ("hand_raised_opening", None),
        ("hand_raised_followup", None),
        ("hand_raised_closing_asked", "What does multiply mean?"),
        ("hand_raised_closing_no", None),
        ("hand_raised_closing_silence", None),
    ]

    for test_mode_label, test_llm_mode in [("Mode A", "a"), ("Mode B", "b")]:
        print(f"\n{'-' * 60}")
        print(f"  Testing {test_mode_label}")
        print(f"{'-' * 60}")

        import llm_responder as _self
        _orig_mode = _self.LLM_MODE
        _self.LLM_MODE = test_llm_mode
        _self._history.clear()

        passed = 0
        for trigger, extra in triggers:
            print(f"\n[{trigger}]")
            result = generate(sample_obs, trigger, extra_context=extra)
            text = result["response_text"]
            used = result["llm_used"]
            lat = result["llm_latency_sec"]
            engine = "LLM" if used else "FALLBACK"
            print(f"  Response: '{text}'")
            print(f"  Engine: {engine}, Latency: {lat:.3f}s")
            if text and len(text) > 2:
                passed += 1
                print(f"  PASS")
            else:
                print(f"  FAIL (empty response)")

        if test_llm_mode == "b":
            print(f"\n[auto — Mode B autonomous detection]")
            result = generate(sample_obs, "auto", extra_context=None)
            text = result["response_text"]
            used = result["llm_used"]
            print(f"  Response: '{text}'")
            print(f"  Engine: {'LLM' if used else 'FALLBACK'}")
            print(f"  (NONE means LLM decided nothing needs attention — valid)")
            if text and len(text) > 2:
                passed += 1
            total_triggers = len(triggers) + 1
        else:
            total_triggers = len(triggers)

        print(f"\n  {test_mode_label} Result: {passed}/{total_triggers} PASS")
        _self.LLM_MODE = _orig_mode

    print("=" * 60)
    print("  SELF-TEST COMPLETE")
    print("=" * 60)
