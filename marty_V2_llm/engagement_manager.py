
import time
import json
import os
import threading

from speech_output import speak, is_speaking, stop as speech_stop
from speech_input import listen_for_answer
from llm_responder import generate as llm_generate

_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
with open(os.path.join(_CFG_DIR, "settings.json"), "r") as f:
    _settings = json.load(f)

_thresholds = _settings["thresholds"]
COOLDOWN_SEC        = float(_settings["cooldown_sec"])
GAZE_AWAY_THRESH    = float(_thresholds["gaze_away_thresh"])
NO_ANSWER_TIMEOUT   = float(_thresholds["no_answer_sec"])
NO_MOVEMENT_TIMEOUT = float(_thresholds["no_movement_sec"])
FACE_ABSENT_TIMEOUT = float(_thresholds["face_absent_sec"])
_LLM_MODE           = _settings["llm_mode"]

PRIORITY_ORDER = [
    "hand_raised",
    "face_absent",
    "no_answer",
    "head_turned",
    "no_movement",
]

_YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "uh huh", "ok", "okay"}
_NO_WORDS  = {"no", "nope", "nah", "not really", "never mind", "nevermind"}


def _is_yes(text: str) -> bool:
    t = text.strip().lower().rstrip(".!?,")
    return any(w in t for w in _YES_WORDS)


def _is_no(text: str) -> bool:
    t = text.strip().lower().rstrip(".!?,")
    return any(w in t for w in _NO_WORDS)


def _condition_met(trigger: str, obs: dict, no_answer_sec: float) -> bool:
    face   = obs.get("face_detected", False)
    gaze   = obs.get("gaze_on_robot", 0.0)
    hand   = obs.get("hand_raised", False)
    no_mov = obs.get("no_movement_sec", 0.0)

    if trigger == "hand_raised":
        return face and hand
    if trigger == "face_absent":
        return not face and no_mov >= FACE_ABSENT_TIMEOUT
    if trigger == "no_answer":
        return face and no_answer_sec >= NO_ANSWER_TIMEOUT
    if trigger == "head_turned":
        return face and gaze < GAZE_AWAY_THRESH
    if trigger == "no_movement":
        return face and no_mov >= NO_MOVEMENT_TIMEOUT
    return False


class EngagementManager:
    def __init__(self):
        self._last_triggered: dict[str, float] = {}
        self.no_answer_sec: float = 0.0
        self.quiz_active: bool = False
        self.interrupt_speech = threading.Event()

    def _cooldown_ok(self, trigger: str) -> bool:
        last = self._last_triggered.get(trigger, 0.0)
        return (time.time() - last) >= COOLDOWN_SEC

    def _fire(self, trigger: str) -> None:
        self._last_triggered[trigger] = time.time()

    def _make_event(self, trigger: str, response_text: str,
                    obs: dict, tts_latency_sec: float = 0.0,
                    student_question: str = "",
                    listen_latency: dict | None = None,
                    llm_used: bool = False,
                    llm_latency_sec: float = 0.0) -> dict:
        event = {
            "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
            "trigger":          trigger,
            "response_text":    response_text,
            "face_detected":    obs.get("face_detected", False),
            "gaze_on_robot":    obs.get("gaze_on_robot", 0.0),
            "hand_raised":      obs.get("hand_raised", False),
            "no_movement_sec":  obs.get("no_movement_sec", 0.0),
            "tts_latency_sec":  tts_latency_sec,
            "llm_used":         llm_used,
            "llm_latency_sec":  llm_latency_sec,
        }
        if student_question:
            event["student_question"] = student_question
        if listen_latency:
            event["mic_listen_sec"] = listen_latency.get("mic_listen_sec", 0.0)
            event["whisper_transcribe_sec"] = listen_latency.get(
                "whisper_transcribe_sec", 0.0)
        return event

    def _handle_hand_raised(self, obs: dict, mode: str) -> dict:
        if is_speaking():
            self.interrupt_speech.set()
            speech_stop()
            self.interrupt_speech.clear()

        if _LLM_MODE == "b":
            llm_result = llm_generate(obs, "auto", mode=mode)
        else:
            llm_result = llm_generate(obs, "hand_raised_opening", mode=mode)
        response = llm_result["response_text"]
        if response.strip().upper() == "NONE":
            response = "I see your hand is up! Do you have a question?"
        tts_lat = speak(response)

        print("  [engagement] Listening for yes/no ...")
        yn_result = listen_for_answer(timeout_sec=8)
        yn_text = yn_result["transcript"]
        student_q = ""
        listen_result = yn_result
        total_llm_lat = llm_result["llm_latency_sec"]
        llm_used = llm_result["llm_used"]

        if _is_yes(yn_text):
            if _LLM_MODE == "b":
                followup = llm_generate(obs, "auto",
                                        extra_context="Yes, I have a question.",
                                        mode=mode)
            else:
                followup = llm_generate(obs, "hand_raised_followup", mode=mode)
            if followup["response_text"].strip().upper() == "NONE":
                followup["response_text"] = "Go ahead, ask your question!"
            speak(followup["response_text"])
            total_llm_lat += followup["llm_latency_sec"]
            llm_used = llm_used and followup["llm_used"]

            print("  [engagement] Listening for student question ...")
            q_result = listen_for_answer(timeout_sec=10)
            student_q = q_result["transcript"]
            listen_result = q_result

            if student_q:
                print(f"  [engagement] Student asked: '{student_q}'")
                if _LLM_MODE == "b":
                    closing = llm_generate(obs, "auto",
                                           extra_context=student_q,
                                           mode=mode)
                else:
                    closing = llm_generate(obs, "hand_raised_closing_asked",
                                           extra_context=student_q, mode=mode)
                if closing["response_text"].strip().upper() == "NONE":
                    closing["response_text"] = "Thanks for asking! Let's continue."
                speak(closing["response_text"])
                total_llm_lat += closing["llm_latency_sec"]
                llm_used = llm_used and closing["llm_used"]
            else:
                if _LLM_MODE == "b":
                    closing = llm_generate(obs, "auto",
                                           extra_context="(student stayed silent)",
                                           mode=mode)
                else:
                    closing = llm_generate(obs, "hand_raised_closing_silence",
                                           mode=mode)
                if closing["response_text"].strip().upper() == "NONE":
                    closing["response_text"] = "No worries! Let's keep going."
                speak(closing["response_text"])
                total_llm_lat += closing["llm_latency_sec"]
                llm_used = llm_used and closing["llm_used"]
        else:
            if _LLM_MODE == "b":
                closing = llm_generate(obs, "auto",
                                       extra_context="No, I don't have a question.",
                                       mode=mode)
            else:
                closing = llm_generate(obs, "hand_raised_closing_no", mode=mode)
            if closing["response_text"].strip().upper() == "NONE":
                closing["response_text"] = "Okay, let's continue!"
            speak(closing["response_text"])
            total_llm_lat += closing["llm_latency_sec"]
            llm_used = llm_used and closing["llm_used"]

        return self._make_event("hand_raised", response, obs,
                                tts_latency_sec=tts_lat,
                                student_question=student_q,
                                listen_latency=listen_result,
                                llm_used=llm_used,
                                llm_latency_sec=total_llm_lat)

    def _check_mode_a(self, obs: dict, mode: str) -> dict | None:
        for trigger in PRIORITY_ORDER:
            if not _condition_met(trigger, obs, self.no_answer_sec):
                continue
            if not self._cooldown_ok(trigger):
                continue

            self._fire(trigger)

            if trigger == "hand_raised":
                return self._handle_hand_raised(obs, mode)

            llm_result = llm_generate(obs, trigger, mode=mode)
            response = llm_result["response_text"]
            tts_lat = speak(response)
            return self._make_event(trigger, response, obs,
                                    tts_latency_sec=tts_lat,
                                    llm_used=llm_result["llm_used"],
                                    llm_latency_sec=llm_result["llm_latency_sec"])

        return None

    def _check_mode_b(self, obs: dict, mode: str) -> dict | None:
        face = obs.get("face_detected", False)
        hand = obs.get("hand_raised", False)

        if face and hand and self._cooldown_ok("hand_raised"):
            self._fire("hand_raised")
            return self._handle_hand_raised(obs, mode)

        if not self._cooldown_ok("auto"):
            return None

        llm_result = llm_generate(obs, "auto", mode=mode)
        response = llm_result["response_text"]

        if response.strip().upper() == "NONE":
            print("  [engagement] Mode B: LLM said NONE, no action needed")
            return None

        self._fire("auto")
        tts_lat = speak(response)
        return self._make_event("llm_auto", response, obs,
                                tts_latency_sec=tts_lat,
                                llm_used=llm_result["llm_used"],
                                llm_latency_sec=llm_result["llm_latency_sec"])

    def check(self, obs: dict) -> dict | None:
        mode = "quiz" if self.quiz_active else "monitor"

        if _LLM_MODE == "b":
            return self._check_mode_b(obs, mode)
        else:
            return self._check_mode_a(obs, mode)

    def check_answer_with_face(self, obs: dict, answer_received: bool) -> dict | None:
        mode = "quiz" if self.quiz_active else "monitor"
        face = obs.get("face_detected", False)
        if not face and answer_received and self._cooldown_ok("face_absent_answer"):
            self._fire("face_absent_answer")
            llm_result = llm_generate(obs, "face_absent", mode=mode)
            response = llm_result["response_text"]
            tts_lat = speak(response)
            return self._make_event("face_absent", response, obs,
                                    tts_latency_sec=tts_lat,
                                    llm_used=llm_result["llm_used"],
                                    llm_latency_sec=llm_result["llm_latency_sec"])
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("  engagement_manager.py — self-test")
    print("  (all tests use mocks — no camera, mic, speaker, or LLM)")
    print("=" * 60)

    tests_passed = 0
    total_tests = 13

    import sys
    _this = sys.modules[__name__]

    _orig_speak = speak
    _orig_listen = listen_for_answer
    _orig_llm = llm_generate
    _orig_is_speaking = is_speaking
    _orig_speech_stop = speech_stop

    _mock_speak_fn = lambda t: (print(f"    [mock speak] '{t}'"), 0.05)[1]
    _this.speak = _mock_speak_fn
    _this.is_speaking = lambda: False
    _this.speech_stop = lambda: None

    _mock_llm_calls = []

    def _mock_llm(obs, trigger, extra_context=None, mode="quiz"):
        _mock_llm_calls.append({"trigger": trigger, "mode": mode,
                                "extra_context": extra_context})
        return {
            "response_text": f"[LLM mock for {trigger}]",
            "llm_used": True,
            "llm_latency_sec": 0.1,
        }
    _this.llm_generate = _mock_llm

    _call_counter = [0]
    def _mock_listen(timeout_sec=10):
        _call_counter[0] += 1
        if _call_counter[0] % 2 == 1:
            return {"transcript": "Yes", "mic_listen_sec": 1.0,
                    "whisper_transcribe_sec": 0.3}
        else:
            return {"transcript": "What is algebra?", "mic_listen_sec": 2.0,
                    "whisper_transcribe_sec": 0.5}
    _this.listen_for_answer = _mock_listen

    def fresh(quiz_active=False):
        em = EngagementManager()
        em.quiz_active = quiz_active
        return em

    obs_hand = {"face_detected": True, "gaze_on_robot": 0.8,
                "hand_raised": True, "no_movement_sec": 0.0}

    _orig_llm_mode = _this._LLM_MODE
    _this._LLM_MODE = "a"

    print("\n[Test 1] Mode A: Hand raised + yes -> llm_used?")
    _call_counter[0] = 0
    _mock_llm_calls.clear()
    em = fresh(quiz_active=True)
    ev = em.check(obs_hand)
    if (ev and ev["trigger"] == "hand_raised"
            and ev.get("llm_used") == True
            and ev.get("llm_latency_sec", 0) > 0):
        print(f"  PASS  (llm_used={ev['llm_used']}, "
              f"llm_lat={ev['llm_latency_sec']}s)")
        tests_passed += 1
    else:
        print(f"  FAIL  (got {ev})")

    print("\n[Test 2] Mode A: Hand raised + no -> no question?")
    _call_counter[0] = 0
    _mock_llm_calls.clear()
    _this.listen_for_answer = lambda timeout_sec=10: {
        "transcript": "No", "mic_listen_sec": 1.0,
        "whisper_transcribe_sec": 0.3}
    em = fresh(quiz_active=True)
    ev = em.check(obs_hand)
    if (ev and ev["trigger"] == "hand_raised"
            and ev.get("llm_used") == True
            and ev.get("student_question", "") == ""):
        print("  PASS")
        tests_passed += 1
    else:
        print(f"  FAIL  (got {ev})")
    _this.listen_for_answer = _mock_listen

    print("\n[Test 3] Mode A: head_turned -> llm_used?")
    _mock_llm_calls.clear()
    em = fresh()
    obs2 = {"face_detected": True, "gaze_on_robot": 0.1,
            "hand_raised": False, "no_movement_sec": 0.0}
    ev = em.check(obs2)
    if ev and ev["trigger"] == "head_turned" and ev.get("llm_used") == True:
        print("  PASS")
        tests_passed += 1
    else:
        print(f"  FAIL  (got {ev})")

    print("\n[Test 4] Mode A: face_absent -> llm_used?")
    _mock_llm_calls.clear()
    em = fresh()
    obs3 = {"face_detected": False, "gaze_on_robot": 0.0,
            "hand_raised": False, "no_movement_sec": 12.0}
    ev = em.check(obs3)
    if ev and ev["trigger"] == "face_absent" and ev.get("llm_used") == True:
        print("  PASS")
        tests_passed += 1
    else:
        print(f"  FAIL  (got {ev})")

    print("\n[Test 5] Mode A: no_answer -> llm_used?")
    _mock_llm_calls.clear()
    em = fresh(quiz_active=True)
    em.no_answer_sec = 25.0
    obs4 = {"face_detected": True, "gaze_on_robot": 0.7,
            "hand_raised": False, "no_movement_sec": 3.0}
    ev = em.check(obs4)
    if ev and ev["trigger"] == "no_answer" and ev.get("llm_used") == True:
        print("  PASS")
        tests_passed += 1
    else:
        print(f"  FAIL  (got {ev})")

    print("\n[Test 6] Mode A: no_movement -> llm_used?")
    _mock_llm_calls.clear()
    em = fresh()
    obs5 = {"face_detected": True, "gaze_on_robot": 0.7,
            "hand_raised": False, "no_movement_sec": 25.0}
    ev = em.check(obs5)
    if ev and ev["trigger"] == "no_movement" and ev.get("llm_used") == True:
        print("  PASS")
        tests_passed += 1
    else:
        print(f"  FAIL  (got {ev})")

    print("\n[Test 7] Mode A: Cooldown — not twice in 15s?")
    _mock_llm_calls.clear()
    em = fresh()
    obs6 = {"face_detected": True, "gaze_on_robot": 0.1,
            "hand_raised": False, "no_movement_sec": 0.0}
    ev1 = em.check(obs6)
    ev2 = em.check(obs6)
    if ev1 is not None and ev2 is None:
        print("  PASS")
        tests_passed += 1
    else:
        print(f"  FAIL  (first={ev1 is not None}, second={ev2 is None})")

    print("\n[Test 8] Mode A: LLM fallback -> llm_used=False?")
    def _mock_llm_fail(obs, trigger, extra_context=None, mode="quiz"):
        return {"response_text": "Fallback text", "llm_used": False,
                "llm_latency_sec": 0.0}
    _this.llm_generate = _mock_llm_fail
    em = fresh()
    ev = em.check(obs5)
    if ev and ev.get("llm_used") == False:
        print("  PASS")
        tests_passed += 1
    else:
        print(f"  FAIL  (got {ev})")
    _this.llm_generate = _mock_llm

    print("\n[Test 9] Mode A: Priority — hand_raised beats head_turned?")
    _mock_llm_calls.clear()
    em = fresh(quiz_active=True)
    obs_both = {"face_detected": True, "gaze_on_robot": 0.1,
                "hand_raised": True, "no_movement_sec": 25.0}
    ev = em.check(obs_both)
    if ev and ev["trigger"] == "hand_raised":
        print("  PASS")
        tests_passed += 1
    else:
        print(f"  FAIL  (got trigger={ev.get('trigger') if ev else None})")

    print("\n[Test 10] Mode A: quiz_active controls mode?")
    _mock_llm_calls.clear()
    _this.listen_for_answer = lambda timeout_sec=10: {
        "transcript": "No", "mic_listen_sec": 1.0,
        "whisper_transcribe_sec": 0.3}
    em = fresh(quiz_active=True)
    ev = em.check(obs_hand)
    quiz_modes = [c["mode"] for c in _mock_llm_calls]
    _mock_llm_calls.clear()
    em2 = fresh(quiz_active=False)
    ev2 = em2.check(obs_hand)
    monitor_modes = [c["mode"] for c in _mock_llm_calls]
    if all(m == "quiz" for m in quiz_modes) and all(m == "monitor" for m in monitor_modes):
        print(f"  PASS  (quiz->{quiz_modes}, monitor->{monitor_modes})")
        tests_passed += 1
    else:
        print(f"  FAIL  (quiz->{quiz_modes}, monitor->{monitor_modes})")
    _this.listen_for_answer = _mock_listen

    _this._LLM_MODE = "b"

    print("\n[Test 11] Mode B: LLM auto-responds to bad obs?")
    _mock_llm_calls.clear()
    em = fresh()
    obs_bad = {"face_detected": True, "gaze_on_robot": 0.1,
               "hand_raised": False, "no_movement_sec": 25.0}
    ev = em.check(obs_bad)
    if (ev and ev["trigger"] == "llm_auto"
            and ev.get("llm_used") == True):
        trigger_sent = _mock_llm_calls[0]["trigger"] if _mock_llm_calls else None
        print(f"  PASS  (trigger sent to LLM: '{trigger_sent}', "
              f"event trigger: 'llm_auto')")
        tests_passed += 1
    else:
        print(f"  FAIL  (got {ev})")

    print("\n[Test 12] Mode B: LLM says NONE -> returns None?")
    def _mock_llm_none(obs, trigger, extra_context=None, mode="quiz"):
        _mock_llm_calls.append({"trigger": trigger, "mode": mode})
        return {"response_text": "NONE", "llm_used": True,
                "llm_latency_sec": 0.05}
    _this.llm_generate = _mock_llm_none
    _mock_llm_calls.clear()
    em = fresh()
    obs_ok = {"face_detected": True, "gaze_on_robot": 0.9,
              "hand_raised": False, "no_movement_sec": 2.0}
    ev = em.check(obs_ok)
    if ev is None:
        print(f"  PASS  (returned None, LLM decided nothing needed)")
        tests_passed += 1
    else:
        print(f"  FAIL  (expected None, got {ev})")
    _this.llm_generate = _mock_llm

    print("\n[Test 13] Mode B: Hand raise still Python-detected?")
    _mock_llm_calls.clear()
    _this.listen_for_answer = lambda timeout_sec=10: {
        "transcript": "No", "mic_listen_sec": 1.0,
        "whisper_transcribe_sec": 0.3}
    em = fresh(quiz_active=True)
    ev = em.check(obs_hand)
    if ev and ev["trigger"] == "hand_raised":
        hr_triggers = [c["trigger"] for c in _mock_llm_calls]
        print(f"  PASS  (hand raise detected, LLM calls: {hr_triggers})")
        tests_passed += 1
    else:
        print(f"  FAIL  (got {ev})")

    _this._LLM_MODE = _orig_llm_mode
    _this.speak = _orig_speak
    _this.listen_for_answer = _orig_listen
    _this.llm_generate = _orig_llm
    _this.is_speaking = _orig_is_speaking
    _this.speech_stop = _orig_speech_stop

    print("-" * 60)
    print(f"  Result: {tests_passed}/{total_tests} PASS")
    if tests_passed == total_tests:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
