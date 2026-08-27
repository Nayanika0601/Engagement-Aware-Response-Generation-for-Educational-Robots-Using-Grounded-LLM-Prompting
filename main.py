
import cv2
import time
import json
import os
import threading

from mediapipe_processor import (
    face_landmarker, pose_landmarker,
    MovementTracker, ObservationBuilder,
    process_frame_data, draw_overlay,
    OBSERVATION_INTERVAL_SEC, NO_MOVEMENT_TIMEOUT_SEC,
    CAMERA_INDEX, SHOW_WINDOW,
)
from speech_output import speak, try_speak
from speech_input import listen_for_answer
from quiz_manager import QuizManager
from logger import SessionLogger
from llm_responder import generate as llm_generate, LLM_MODE as _LLM_MODE

_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
with open(os.path.join(_CFG_DIR, "settings.json"), "r") as f:
    _settings = json.load(f)

_thresholds = _settings["thresholds"]
_COOLDOWN_SEC        = float(_settings["cooldown_sec"])
_HAND_RAISE_SEC      = 3.0
_FACE_ABSENT_SEC     = float(_thresholds["face_absent_sec"])
_NO_MOVEMENT_SEC     = float(_thresholds["no_movement_sec"])
_GAZE_AWAY_THRESH    = float(_thresholds["gaze_away_thresh"])

MAX_QUESTIONS = 5

_YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "uh huh", "ok", "okay"}


def _is_yes(text: str) -> bool:
    t = text.strip().lower().rstrip(".!?,")
    return any(w in t for w in _YES_WORDS)


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_obs: dict | None = None
        self._obs_count: int = 0
        self._new_obs_ready = threading.Event()
        self.running = True

        self.interrupt_event = threading.Event()
        self.interrupt_reason = ""
        self.pending_alert_text = ""

        self._alert_cooldown: dict[str, float] = {}

        self.detection_time: float = 0.0


        self._hand_raise_start: float = 0.0
        self._hand_last_seen: float = 0.0
        self._hand_tracking: bool = False

        self._face_absent_start: float = 0.0
        self._face_tracking_absent: bool = False

    def push_observation(self, obs_dict: dict) -> None:
        with self._lock:
            self._latest_obs = obs_dict
            self._obs_count += 1
        self._new_obs_ready.set()

    def pop_observation(self, timeout: float = 1.0) -> dict | None:
        if self._new_obs_ready.wait(timeout=timeout):
            self._new_obs_ready.clear()
            with self._lock:
                return self._latest_obs
        return None

    def get_latest(self) -> dict | None:
        with self._lock:
            return self._latest_obs

    @property
    def obs_count(self) -> int:
        with self._lock:
            return self._obs_count

    def cooldown_ok(self, trigger: str) -> bool:
        last = self._alert_cooldown.get(trigger, 0.0)
        return (time.time() - last) >= _COOLDOWN_SEC

    def fire_cooldown(self, trigger: str) -> None:
        self._alert_cooldown[trigger] = time.time()

    def request_interrupt(self, reason: str, alert_text: str = "") -> None:
        self.interrupt_reason = reason
        self.pending_alert_text = alert_text
        self.interrupt_event.set()


    def update_hand_raise(self, hand_raised: bool) -> float:
        now = time.time()
        if hand_raised:
            self._hand_last_seen = now
            if not self._hand_tracking:
                self._hand_raise_start = now
                self._hand_tracking = True
            return now - self._hand_raise_start
        else:
            if self._hand_tracking:
                gap = now - self._hand_last_seen
                if gap > 1.0:
                    self._hand_tracking = False
                    self._hand_raise_start = 0.0
                    return 0.0
                else:
                    return now - self._hand_raise_start
            return 0.0


    def update_face_absent(self, face_detected: bool) -> float:
        now = time.time()
        if face_detected:
            self._face_tracking_absent = False
            self._face_absent_start = 0.0
            return 0.0
        else:
            if not self._face_tracking_absent:
                self._face_absent_start = now
                self._face_tracking_absent = True
            return now - self._face_absent_start


def camera_thread(cap, tracker, builder, log, state: SharedState):
    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_data = process_frame_data(rgb, w, h, tracker)
        builder.add_frame(frame_data)

        face = frame_data.get("face_detected", False)
        hand = frame_data.get("hand_raised", False)


        hand_dur = state.update_hand_raise(hand)
        if hand_dur >= _HAND_RAISE_SEC and state.cooldown_ok("hand_raised"):
            state.fire_cooldown("hand_raised")
            state.detection_time = time.time()
            print(f"  [camera {time.strftime('%H:%M:%S')}] "
                  f"Hand raised detected ({hand_dur:.1f}s sustained)")
            state.request_interrupt("hand_raised")
            state._hand_tracking = False
            state._hand_raise_start = 0.0

        face_absent_dur = state.update_face_absent(face)
        if face_absent_dur >= _FACE_ABSENT_SEC and state.cooldown_ok("face_absent"):
            state.fire_cooldown("face_absent")
            state.detection_time = time.time()
            print(f"  [camera {time.strftime('%H:%M:%S')}] "
                  f"Face absent detected ({face_absent_dur:.1f}s)")
            state.request_interrupt("alert", "face_absent")

        if builder.ready():
            obs = builder.build()
            obs_dict = obs.to_dict()
            log.log_observation(obs_dict)
            state.push_observation(obs_dict)

            gaze   = obs_dict.get("gaze_on_robot", 0.0)
            no_mov = obs_dict.get("no_movement_sec", 0.0)
            obs_face = obs_dict.get("face_detected", False)

            if obs_face and no_mov >= _NO_MOVEMENT_SEC and state.cooldown_ok("no_movement"):
                state.fire_cooldown("no_movement")
                state.detection_time = time.time()
                print(f"  [camera {time.strftime('%H:%M:%S')}] "
                      f"No movement detected ({no_mov:.1f}s)")
                state.request_interrupt("alert", "no_movement")

            elif obs_face and gaze < _GAZE_AWAY_THRESH and state.cooldown_ok("head_turned"):
                state.fire_cooldown("head_turned")
                state.detection_time = time.time()
                print(f"  [camera {time.strftime('%H:%M:%S')}] "
                      f"Head turned detected (gaze={gaze:.2f})")
                state.request_interrupt("alert", "head_turned")

        if SHOW_WINDOW:
            countdown = max(0.0, OBSERVATION_INTERVAL_SEC -
                            (time.time() - builder._last_obs_time))
            draw_overlay(frame, frame_data, countdown)
            cv2.imshow("Marty — Edu Robot", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                state.running = False
                break


def phase1_camera_check(cap, tracker, builder, log):
    print("\n" + "=" * 55)
    print("  PHASE 1 — Camera check (10 seconds)")
    print("=" * 55)

    saw_face = False
    saw_hand = False
    saw_head_move = False
    start = time.time()

    while time.time() - start < 10.0:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame.")
            return False

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_data = process_frame_data(rgb, w, h, tracker)
        builder.add_frame(frame_data)

        if frame_data["face_detected"]:
            saw_face = True
        if frame_data["hand_raised"]:
            saw_hand = True
        if frame_data["head_moving"]:
            saw_head_move = True

        if builder.ready():
            obs = builder.build()
            log.log_observation(obs.to_dict())
            print(f"  [phase1] Obs: face={obs.face_detected} "
                  f"hand={obs.hand_raised} head_moving={obs.head_moving}")

        if SHOW_WINDOW:
            countdown = max(0.0, OBSERVATION_INTERVAL_SEC -
                            (time.time() - builder._last_obs_time))
            draw_overlay(frame, frame_data, countdown)
            cv2.imshow("Marty — Phase 1 Camera Check", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            return False

    print("\n  Phase 1 results:")
    print(f"    Face detected    : {'YES' if saw_face else 'NO'}")
    print(f"    Hand raise       : {'YES' if saw_hand else 'NO'}")
    print(f"    Head movement    : {'YES' if saw_head_move else 'NO'}")

    if saw_face:
        print("\n  Camera OK — face detected.")
    else:
        print("\n  WARNING: No face detected during warm-up.")

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    speak("Camera check complete. Are you ready to start the quiz? Say yes or no.")
    result = listen_for_answer(timeout_sec=10)
    answer = result["transcript"].strip().lower()
    print(f"  Heard: '{answer}'")

    if _is_yes(answer):
        speak("Great!")
        return "quiz"
    elif answer == "":
        speak("I didn't hear you. Say yes to start or no to just watch.")
        result = listen_for_answer(timeout_sec=10)
        answer = result["transcript"].strip().lower()
        return "quiz" if _is_yes(answer) else "monitor"
    else:
        speak("No problem! I'll just keep watching for now.")
        return "monitor"


def handle_hand_raise(log, obs: dict | None = None,
                      mode: str = "quiz") -> None:
    if obs is None:
        obs = {}

    if _LLM_MODE == "b":
        llm_open = llm_generate(obs, "auto", mode=mode)
    else:
        llm_open = llm_generate(obs, "hand_raised_opening", mode=mode)
    response = llm_open["response_text"]
    if response.strip().upper() == "NONE":
        response = "I see your hand is up! Do you have a question?"
    speak(response)
    total_llm_lat = llm_open["llm_latency_sec"]
    llm_used = llm_open["llm_used"]

    yn = listen_for_answer(timeout_sec=8)
    yn_text = yn["transcript"]
    student_q = ""

    if _is_yes(yn_text):
        if _LLM_MODE == "b":
            llm_fu = llm_generate(obs, "auto",
                                  extra_context="Yes, I have a question.",
                                  mode=mode)
        else:
            llm_fu = llm_generate(obs, "hand_raised_followup", mode=mode)
        if llm_fu["response_text"].strip().upper() == "NONE":
            llm_fu["response_text"] = "Go ahead, ask your question!"
        speak(llm_fu["response_text"])
        total_llm_lat += llm_fu["llm_latency_sec"]
        llm_used = llm_used and llm_fu["llm_used"]

        q_result = listen_for_answer(timeout_sec=10)
        student_q = q_result["transcript"]

        if student_q:
            print(f"  [hand raise] Student asked: '{student_q}'")
            if _LLM_MODE == "b":
                llm_close = llm_generate(obs, "auto",
                                         extra_context=student_q,
                                         mode=mode)
            else:
                llm_close = llm_generate(obs, "hand_raised_closing_asked",
                                         extra_context=student_q, mode=mode)
        else:
            if _LLM_MODE == "b":
                llm_close = llm_generate(obs, "auto",
                                         extra_context="(student stayed silent)",
                                         mode=mode)
            else:
                llm_close = llm_generate(obs, "hand_raised_closing_silence",
                                         mode=mode)
        if llm_close["response_text"].strip().upper() == "NONE":
            llm_close["response_text"] = "Thanks for sharing! Let's continue."
        speak(llm_close["response_text"])
        total_llm_lat += llm_close["llm_latency_sec"]
        llm_used = llm_used and llm_close["llm_used"]
    else:
        if _LLM_MODE == "b":
            llm_close = llm_generate(obs, "auto",
                                     extra_context="No, I don't have a question.",
                                     mode=mode)
        else:
            llm_close = llm_generate(obs, "hand_raised_closing_no", mode=mode)
        if llm_close["response_text"].strip().upper() == "NONE":
            llm_close["response_text"] = "Okay, let's continue!"
        speak(llm_close["response_text"])
        total_llm_lat += llm_close["llm_latency_sec"]
        llm_used = llm_used and llm_close["llm_used"]

    log.log_engagement_event({
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trigger":          "hand_raised",
        "response_text":    response,
        "student_question": student_q,
        "llm_used":         llm_used,
        "llm_latency_sec":  total_llm_lat,
        "llm_prompt_sent":  llm_open.get("llm_prompt_sent", ""),
        "llm_raw_response": llm_open.get("llm_raw_response", ""),
        "llm_retry_log":    llm_open.get("llm_retry_log", []),
    })


def phase234_main_loop(cap, tracker, builder, log):
    state = SharedState()
    quiz  = QuizManager(start_level=1, max_questions=MAX_QUESTIONS)
    resume_question = False

    cam_thread = threading.Thread(
        target=camera_thread,
        args=(cap, tracker, builder, log, state),
        daemon=True,
        name="camera-thread",
    )
    cam_thread.start()

    print("\n" + "=" * 60)
    print(f"  PHASE 2+3 — Quiz ({MAX_QUESTIONS} questions) + Engagement")
    print("  Camera runs continuously in background")
    print("  Hand raise 5s → interrupt | Face absent 5s → alert")
    print("  Press Q in camera window to quit early")
    print("=" * 60)

    speak(f"Let's begin. I'll ask you {MAX_QUESTIONS} maths questions.")

    try:
        while state.running and not quiz.is_done:
            obs_dict = state.get_latest() or {}
            face_now = obs_dict.get("face_detected", False)

            print(f"\n  [quiz {quiz.questions_asked + 1}/{MAX_QUESTIONS}]")

            if state.interrupt_event.is_set():
                reason = state.interrupt_reason
                alert_text = state.pending_alert_text
                state.interrupt_event.clear()

                if reason == "hand_raised":
                    gap = time.time() - state.detection_time
                    print(f"  [pending {time.strftime('%H:%M:%S')}] "
                          f"Hand raised — LLM triggered "
                          f"({gap:.1f}s after detection)")
                    latest_obs = state.get_latest() or {}
                    handle_hand_raise(log, latest_obs, mode="quiz")
                    resume_question = True
                    continue
                elif reason == "alert" and alert_text:
                    gap = time.time() - state.detection_time
                    print(f"  [pending {time.strftime('%H:%M:%S')}] "
                          f"Alert ({alert_text}) — LLM triggered "
                          f"({gap:.1f}s after detection)")
                    latest_obs = state.get_latest() or {}
                    llm_result = llm_generate(latest_obs,
                                              "auto" if _LLM_MODE == "b" else alert_text,
                                              mode="quiz")
                    response = llm_result["response_text"]
                    print(f"  [llm->speak {time.strftime('%H:%M:%S')}] "
                          f"'{response}'")
                    speak(response)
                    log.log_engagement_event({
                        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "trigger":         alert_text,
                        "response_text":   response,
                        "llm_used":        llm_result["llm_used"],
                        "llm_latency_sec": llm_result["llm_latency_sec"],
                        "llm_prompt_sent":  llm_result.get("llm_prompt_sent", ""),
                        "llm_raw_response": llm_result.get("llm_raw_response", ""),
                        "llm_retry_log":    llm_result.get("llm_retry_log", []),
                    })
                    resume_question = True
                    continue

            state.interrupt_event.clear()
            state.interrupt_reason = ""

            result = quiz.ask_next_question(
                listen_timeout=15,
                face_detected=face_now,
                resume=resume_question,
                stop_event=state.interrupt_event,
            )
            resume_question = False

            if result.get("interrupted"):
                reason = state.interrupt_reason
                alert_text = state.pending_alert_text
                state.interrupt_event.clear()

                if reason == "hand_raised":
                    gap = time.time() - state.detection_time
                    print(f"  [interrupt {time.strftime('%H:%M:%S')}] "
                          f"Hand raised — LLM triggered "
                          f"({gap:.1f}s after detection)")
                    latest_obs = state.get_latest() or {}
                    handle_hand_raise(log, latest_obs, mode="quiz")
                    resume_question = True
                    continue

                elif reason == "alert":
                    gap = time.time() - state.detection_time
                    print(f"  [interrupt {time.strftime('%H:%M:%S')}] "
                          f"Alert ({alert_text}) — LLM triggered "
                          f"({gap:.1f}s after detection)")
                    latest_obs = state.get_latest() or {}
                    llm_result = llm_generate(latest_obs,
                                              "auto" if _LLM_MODE == "b" else alert_text,
                                              mode="quiz")
                    response = llm_result["response_text"]
                    print(f"  [llm→speak {time.strftime('%H:%M:%S')}] "
                          f"'{response}'")
                    speak(response)
                    log.log_engagement_event({
                        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "trigger":         alert_text,
                        "response_text":   response,
                        "llm_used":        llm_result["llm_used"],
                        "llm_latency_sec": llm_result["llm_latency_sec"],
                        "llm_prompt_sent":  llm_result.get("llm_prompt_sent", ""),
                        "llm_raw_response": llm_result.get("llm_raw_response", ""),
                        "llm_retry_log":    llm_result.get("llm_retry_log", []),
                    })
                    resume_question = True
                    continue

            log.log_quiz_result(result)

            latest = state.get_latest() or obs_dict
            if not latest.get("face_detected", False):
                if result["student_answer"]:
                    llm_fa = llm_generate(latest,
                                          "auto" if _LLM_MODE == "b" else "face_absent",
                                          mode="quiz")
                    speak(llm_fa["response_text"])
                    log.log_engagement_event({
                        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "trigger":         "face_absent_answer",
                        "response_text":   llm_fa["response_text"],
                        "answer_still_checked": True,
                        "llm_used":        llm_fa["llm_used"],
                        "llm_latency_sec": llm_fa["llm_latency_sec"],
                        "llm_prompt_sent":  llm_fa.get("llm_prompt_sent", ""),
                        "llm_raw_response": llm_fa.get("llm_raw_response", ""),
                        "llm_retry_log":    llm_fa.get("llm_retry_log", []),
                    })

            log.save()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    if quiz.is_done:
        speak(f"That's all {MAX_QUESTIONS} questions done! Great effort!")
    else:
        speak("Quiz ended. Well done!")

    log.summarise()
    log.save()

    print("\n" + "=" * 60)
    print("  PHASE 5 — Post-quiz camera monitoring")
    print("  Camera continues observing every 5s")
    print("  Engagement alerts still active")
    print("  Press Q in camera window to exit")
    print("=" * 60)

    speak("Quiz is done. I'll keep watching. Press Q to exit.")

    try:
        while state.running:
            obs_dict = state.pop_observation(timeout=2.0)
            if obs_dict is None:
                continue

            face = obs_dict.get("face_detected", False)
            no_mov = obs_dict.get("no_movement_sec", 0.0)
            print(f"  [monitor] face={face} no_mov={no_mov:.1f}s "
                  f"obs={state.obs_count}")

            if state.interrupt_event.is_set():
                reason = state.interrupt_reason
                alert_text = state.pending_alert_text
                state.interrupt_event.clear()

                if reason == "alert" and alert_text:
                    gap = time.time() - state.detection_time
                    print(f"  [post-quiz {time.strftime('%H:%M:%S')}] "
                          f"Alert ({alert_text}) — LLM triggered "
                          f"({gap:.1f}s after detection)")
                    latest_obs = state.get_latest() or {}
                    llm_result = llm_generate(latest_obs,
                                              "auto" if _LLM_MODE == "b" else alert_text,
                                              mode="monitor")
                    response = llm_result["response_text"]
                    print(f"  [llm→speak {time.strftime('%H:%M:%S')}] "
                          f"'{response}'")
                    speak(response)
                    log.log_engagement_event({
                        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "trigger":         "post_quiz_" + alert_text,
                        "response_text":   response,
                        "llm_used":        llm_result["llm_used"],
                        "llm_latency_sec": llm_result["llm_latency_sec"],
                        "llm_prompt_sent":  llm_result.get("llm_prompt_sent", ""),
                        "llm_raw_response": llm_result.get("llm_raw_response", ""),
                        "llm_retry_log":    llm_result.get("llm_retry_log", []),
                    })
                elif reason == "hand_raised":
                    gap = time.time() - state.detection_time
                    print(f"  [post-quiz {time.strftime('%H:%M:%S')}] "
                          f"Hand raised — LLM triggered "
                          f"({gap:.1f}s after detection)")
                    latest_obs = state.get_latest() or {}
                    handle_hand_raise(log, latest_obs, mode="monitor")

                log.save()

    except KeyboardInterrupt:
        print("\nExiting.")

    state.running = False
    cam_thread.join(timeout=3.0)
    log.save()


def phase5_monitor_only(cap, tracker, builder, log):
    state = SharedState()

    cam_thread = threading.Thread(
        target=camera_thread,
        args=(cap, tracker, builder, log, state),
        daemon=True,
        name="camera-thread",
    )
    cam_thread.start()

    print("\n" + "=" * 60)
    print("  MONITOR MODE — Camera monitoring only (no quiz)")
    print("  Engagement alerts active")
    print("  Press Q in camera window to exit")
    print("=" * 60)

    speak("I'll just keep watching for now. Press Q whenever you want to stop.")

    try:
        while state.running:
            obs_dict = state.pop_observation(timeout=2.0)
            if obs_dict is None:
                continue

            face = obs_dict.get("face_detected", False)
            no_mov = obs_dict.get("no_movement_sec", 0.0)
            print(f"  [monitor] face={face} no_mov={no_mov:.1f}s "
                  f"obs={state.obs_count}")

            if state.interrupt_event.is_set():
                reason = state.interrupt_reason
                alert_text = state.pending_alert_text
                state.interrupt_event.clear()

                if reason == "alert" and alert_text:
                    gap = time.time() - state.detection_time
                    print(f"  [monitor {time.strftime('%H:%M:%S')}] "
                          f"Alert ({alert_text}) — LLM triggered "
                          f"({gap:.1f}s after detection)")
                    latest_obs = state.get_latest() or {}
                    llm_result = llm_generate(latest_obs,
                                              "auto" if _LLM_MODE == "b" else alert_text,
                                              mode="monitor")
                    response = llm_result["response_text"]
                    print(f"  [llm→speak {time.strftime('%H:%M:%S')}] "
                          f"'{response}'")
                    speak(response)
                    log.log_engagement_event({
                        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "trigger":         "monitor_" + alert_text,
                        "response_text":   response,
                        "llm_used":        llm_result["llm_used"],
                        "llm_latency_sec": llm_result["llm_latency_sec"],
                        "llm_prompt_sent":  llm_result.get("llm_prompt_sent", ""),
                        "llm_raw_response": llm_result.get("llm_raw_response", ""),
                        "llm_retry_log":    llm_result.get("llm_retry_log", []),
                    })
                elif reason == "hand_raised":
                    gap = time.time() - state.detection_time
                    print(f"  [monitor {time.strftime('%H:%M:%S')}] "
                          f"Hand raised — LLM triggered "
                          f"({gap:.1f}s after detection)")
                    latest_obs = state.get_latest() or {}
                    handle_hand_raise(log, latest_obs, mode="monitor")

                log.save()

    except KeyboardInterrupt:
        print("\nExiting.")

    state.running = False
    cam_thread.join(timeout=3.0)
    log.save()


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera. Check CAMERA_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    tracker = MovementTracker()
    builder = ObservationBuilder()
    log     = SessionLogger(log_dir=".")

    print("=" * 60)
    print("  Marty Edu Robot — Main Pipeline (threaded)")
    print("=" * 60)

    mode = phase1_camera_check(cap, tracker, builder, log)
    if mode == "quiz":
        phase234_main_loop(cap, tracker, builder, log)
    else:
        phase5_monitor_only(cap, tracker, builder, log)

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    pose_landmarker.close()
    print("\nSession ended. Goodbye!")


if __name__ == "__main__":
    main()
