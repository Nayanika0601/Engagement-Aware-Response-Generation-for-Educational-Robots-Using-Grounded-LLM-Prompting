
import json
import time
import os
import uuid
import threading


class SessionLogger:
    def __init__(self, log_dir: str = "."):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{ts}_{uuid.uuid4().hex[:6]}"
        self.start_time = time.strftime("%Y-%m-%dT%H:%M:%S")

        try:
            _cfg_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "config")
            with open(os.path.join(_cfg_dir, "settings.json"), "r") as f:
                _mode = json.load(f).get("llm_mode", "a")
            mode_folder = "mode_b1" if _mode == "b" else f"mode_{_mode}"
        except Exception:
            mode_folder = "mode_a"

        self._log_dir = os.path.join(log_dir, mode_folder)
        os.makedirs(self._log_dir, exist_ok=True)
        self._filename = os.path.join(self._log_dir, f"session_{ts}.json")
        self._lock = threading.Lock()

        self._data = {
            "session_id":         self.session_id,
            "start_time":         self.start_time,
            "observations":       [],
            "quiz_results":       [],
            "engagement_events":  [],
            "summary":            {},
        }


    def log_observation(self, obs_dict: dict) -> None:
        with self._lock:
            self._data["observations"].append(obs_dict)

    def log_quiz_result(self, result_dict: dict) -> None:
        with self._lock:
            self._data["quiz_results"].append(result_dict)

    def log_engagement_event(self, event_dict: dict) -> None:
        with self._lock:
            self._data["engagement_events"].append(event_dict)


    def save(self) -> str:
        with self._lock:
            self._data["summary"] = self._build_summary()
            with open(self._filename, "w") as f:
                json.dump(self._data, f, indent=2)
        print(f"  [logger] Saved -> {self._filename}")
        return self._filename


    def _build_summary(self) -> dict:
        observations = self._data["observations"]
        results      = self._data["quiz_results"]
        engagement   = self._data["engagement_events"]

        def _avg(lst):
            return round(sum(lst) / len(lst), 3) if lst else 0.0

        def _pct(count, total):
            return round(count / total * 100, 1) if total else 0.0


        total_obs = len(observations)
        face_detected_count = sum(
            1 for o in observations if o.get("face_detected"))
        body_detected_count = sum(
            1 for o in observations if o.get("body_detected"))
        head_moving_count = sum(
            1 for o in observations if o.get("head_moving"))
        hand_raised_count = sum(
            1 for o in observations if o.get("hand_raised"))
        gaze_scores = [o.get("gaze_on_robot", 0.0) for o in observations
                       if o.get("face_detected")]

        detection_accuracy = {
            "total_observations":     total_obs,
            "face_detection_rate_pct": _pct(face_detected_count, total_obs),
            "body_detection_rate_pct": _pct(body_detected_count, total_obs),
            "head_moving_rate_pct":    _pct(head_moving_count, total_obs),
            "hand_raised_count":       hand_raised_count,
            "avg_gaze_score":          _avg(gaze_scores),
        }


        if results:
            total_q = len(results)
            completed = [r for r in results if not r.get("interrupted")]
            total_completed = len(completed)
            speech_heard = sum(
                1 for r in completed if r.get("student_answer"))
            correct = sum(
                1 for r in completed if r.get("is_correct"))
            face_during = sum(
                1 for r in completed if r.get("face_detected_during"))
            interrupted = sum(
                1 for r in results if r.get("interrupted"))

            speech_accuracy = {
                "total_questions":          total_q,
                "completed_questions":      total_completed,
                "interrupted_questions":    interrupted,
                "speech_captured_count":    speech_heard,
                "speech_capture_rate_pct":  _pct(speech_heard, total_completed),
                "correct_answers":          correct,
                "answer_accuracy_pct":      _pct(correct, total_completed),
                "face_present_during_pct":  _pct(face_during, total_completed),
                "completion_rate_pct":      _pct(total_completed, total_q),
            }
        else:
            speech_accuracy = {
                "total_questions":          0,
                "speech_capture_rate_pct":  0.0,
                "answer_accuracy_pct":      0.0,
            }


        per_level = {}
        if results:
            levels: dict[int, dict] = {}
            for r in results:
                if r.get("interrupted"):
                    continue
                lvl = r.get("difficulty_level", 0)
                if lvl not in levels:
                    levels[lvl] = {"correct": 0, "total": 0}
                levels[lvl]["total"] += 1
                if r.get("is_correct"):
                    levels[lvl]["correct"] += 1

            for lvl, d in sorted(levels.items()):
                per_level[f"level_{lvl}"] = {
                    "correct":      d["correct"],
                    "total":        d["total"],
                    "accuracy_pct": _pct(d["correct"], d["total"]),
                }


        trigger_counts: dict[str, int] = {}
        for e in engagement:
            t = e.get("trigger", "unknown")
            trigger_counts[t] = trigger_counts.get(t, 0) + 1

        engagement_accuracy = {
            "total_events":       len(engagement),
            "triggers_by_type":   trigger_counts,
        }


        completed_results = [r for r in results if not r.get("interrupted")]

        tts_q_lats   = [r["tts_question_latency_sec"] for r in completed_results
                        if r.get("tts_question_latency_sec")]
        tts_fb_lats  = [r["tts_feedback_latency_sec"] for r in completed_results
                        if r.get("tts_feedback_latency_sec")]
        mic_lats     = [r["mic_listen_sec"] for r in completed_results
                        if r.get("mic_listen_sec")]
        whisper_lats = [r["whisper_transcribe_sec"] for r in completed_results
                        if r.get("whisper_transcribe_sec")]
        cycle_lats   = [r["total_cycle_sec"] for r in completed_results
                        if r.get("total_cycle_sec")]
        resp_times   = [r["response_time_sec"] for r in completed_results
                        if r.get("response_time_sec")]

        latency = {
            "avg_tts_question_sec":   _avg(tts_q_lats),
            "avg_tts_feedback_sec":   _avg(tts_fb_lats),
            "avg_mic_listen_sec":     _avg(mic_lats),
            "avg_whisper_transcribe_sec": _avg(whisper_lats),
            "avg_response_time_sec":  _avg(resp_times),
            "avg_total_cycle_sec":    _avg(cycle_lats),
        }


        det_score   = detection_accuracy["face_detection_rate_pct"]
        speech_score = speech_accuracy.get("speech_capture_rate_pct", 0.0)
        quiz_score   = speech_accuracy.get("answer_accuracy_pct", 0.0)
        comp_score   = speech_accuracy.get("completion_rate_pct", 0.0)

        pipeline_score = round(
            (det_score * 0.25) +
            (speech_score * 0.25) +
            (quiz_score * 0.25) +
            (comp_score * 0.25), 1)

        return {
            "detection_accuracy":   detection_accuracy,
            "speech_accuracy":      speech_accuracy,
            "per_level":            per_level,
            "engagement":           engagement_accuracy,
            "latency":              latency,
            "pipeline_score":       pipeline_score,
        }

    def summarise(self) -> None:
        with self._lock:
            s = self._build_summary()

        det  = s.get("detection_accuracy", {})
        spc  = s.get("speech_accuracy", {})
        eng  = s.get("engagement", {})
        lat  = s.get("latency", {})
        plvl = s.get("per_level", {})

        print("\n" + "=" * 60)
        print("  FULL PIPELINE SUMMARY")
        print("=" * 60)

        print("\n  DETECTION ACCURACY (camera + MediaPipe)")
        print(f"    Total observations   : {det.get('total_observations', 0)}")
        print(f"    Face detection rate   : {det.get('face_detection_rate_pct', 0)}%")
        print(f"    Body detection rate   : {det.get('body_detection_rate_pct', 0)}%")
        print(f"    Head moving rate      : {det.get('head_moving_rate_pct', 0)}%")
        print(f"    Hand raised count     : {det.get('hand_raised_count', 0)}")
        print(f"    Avg gaze score        : {det.get('avg_gaze_score', 0)}")

        print("\n  SPEECH RECOGNITION ACCURACY (Whisper)")
        print(f"    Total questions       : {spc.get('total_questions', 0)}")
        print(f"    Completed             : {spc.get('completed_questions', 0)}")
        print(f"    Interrupted           : {spc.get('interrupted_questions', 0)}")
        print(f"    Speech captured       : {spc.get('speech_captured_count', 0)}")
        print(f"    Speech capture rate   : {spc.get('speech_capture_rate_pct', 0)}%")
        print(f"    Face present during   : {spc.get('face_present_during_pct', 0)}%")
        print(f"    Completion rate       : {spc.get('completion_rate_pct', 0)}%")

        print("\n  QUIZ ACCURACY")
        print(f"    Correct answers       : {spc.get('correct_answers', 0)}")
        print(f"    Answer accuracy       : {spc.get('answer_accuracy_pct', 0)}%")
        for lvl_key in sorted(plvl):
            d = plvl[lvl_key]
            print(f"    {lvl_key}: {d['correct']}/{d['total']} "
                  f"({d['accuracy_pct']}%)")

        print("\n  ENGAGEMENT")
        print(f"    Total events          : {eng.get('total_events', 0)}")
        triggers = eng.get("triggers_by_type", {})
        for t, c in sorted(triggers.items()):
            print(f"      {t}: {c}")

        print("\n  LATENCY")
        print(f"    Avg TTS (question)    : {lat.get('avg_tts_question_sec', 0)}s")
        print(f"    Avg TTS (feedback)    : {lat.get('avg_tts_feedback_sec', 0)}s")
        print(f"    Avg mic listen        : {lat.get('avg_mic_listen_sec', 0)}s")
        print(f"    Avg Whisper transcr.  : {lat.get('avg_whisper_transcribe_sec', 0)}s")
        print(f"    Avg response time     : {lat.get('avg_response_time_sec', 0)}s")
        print(f"    Avg total cycle       : {lat.get('avg_total_cycle_sec', 0)}s")

        print("\n  OVERALL PIPELINE SCORE")
        print(f"    Score: {s.get('pipeline_score', 0)} / 100")
        print(f"      Detection  (25%): {det.get('face_detection_rate_pct', 0)}%")
        print(f"      Speech     (25%): {spc.get('speech_capture_rate_pct', 0)}%")
        print(f"      Quiz       (25%): {spc.get('answer_accuracy_pct', 0)}%")
        print(f"      Completion (25%): {spc.get('completion_rate_pct', 0)}%")
        print("=" * 60)


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  logger.py — self-test")
    print("=" * 60)

    tests_passed = 0
    total_tests = 4

    log = SessionLogger(log_dir=tempfile.gettempdir())

    print("\n[Test 1] log_observation works?")
    log.log_observation({"timestamp": "2025-01-01T00:00:00",
                         "face_detected": True, "body_detected": True,
                         "gaze_on_robot": 0.85, "head_moving": True,
                         "hand_raised": False, "no_movement_sec": 0.0})
    log.log_observation({"timestamp": "2025-01-01T00:00:05",
                         "face_detected": True, "body_detected": True,
                         "gaze_on_robot": 0.72, "head_moving": False,
                         "hand_raised": False, "no_movement_sec": 3.0})
    log.log_observation({"timestamp": "2025-01-01T00:00:10",
                         "face_detected": False, "body_detected": False,
                         "gaze_on_robot": 0.0, "head_moving": False,
                         "hand_raised": False, "no_movement_sec": 8.0})
    if len(log._data["observations"]) == 3:
        print("  PASS")
        tests_passed += 1
    else:
        print("  FAIL")

    print("\n[Test 2] log_quiz_result + log_engagement_event?")
    log.log_quiz_result({
        "question_id": "L1_Q01", "question_text": "What is 6 times 4?",
        "correct_answer": "24", "student_answer": "twenty four",
        "is_correct": True, "response_time_sec": 4.5,
        "difficulty_level": 1, "face_detected_during": True,
        "tts_question_latency_sec": 1.2, "tts_feedback_latency_sec": 0.8,
        "mic_listen_sec": 3.5, "whisper_transcribe_sec": 1.0,
        "total_cycle_sec": 6.5, "interrupted": False,
    })
    log.log_quiz_result({
        "question_id": "L1_Q02", "question_text": "What is 15 minus 8?",
        "correct_answer": "7", "student_answer": "nine",
        "is_correct": False, "response_time_sec": 6.1,
        "difficulty_level": 1, "face_detected_during": True,
        "tts_question_latency_sec": 1.0, "tts_feedback_latency_sec": 1.1,
        "mic_listen_sec": 4.2, "whisper_transcribe_sec": 0.9,
        "total_cycle_sec": 7.2, "interrupted": False,
    })
    log.log_quiz_result({
        "question_id": "L1_Q03", "question_text": "What is 9 times 3?",
        "correct_answer": "27", "student_answer": "",
        "is_correct": False, "response_time_sec": 0.0,
        "difficulty_level": 1, "face_detected_during": False,
        "tts_question_latency_sec": 1.1, "tts_feedback_latency_sec": 0.0,
        "mic_listen_sec": 2.0, "whisper_transcribe_sec": 0.0,
        "total_cycle_sec": 3.1, "interrupted": True,
    })
    log.log_engagement_event({
        "timestamp": "2025-01-01T00:00:10",
        "trigger": "no_movement",
        "response_text": "Are you still there?",
    })
    log.log_engagement_event({
        "timestamp": "2025-01-01T00:00:25",
        "trigger": "hand_raised",
        "response_text": "Do you have a question?",
        "student_question": "What is algebra?",
    })
    ok = (len(log._data["quiz_results"]) == 3 and
          len(log._data["engagement_events"]) == 2)
    if ok:
        print("  PASS")
        tests_passed += 1
    else:
        print("  FAIL")

    print("\n[Test 3] save() writes file with pipeline score?")
    path = log.save()
    if os.path.isfile(path):
        with open(path) as f:
            saved = json.load(f)
        has_score = "pipeline_score" in saved.get("summary", {})
        has_det = "detection_accuracy" in saved.get("summary", {})
        size = os.path.getsize(path)
        if has_score and has_det:
            print(f"  PASS  ({size} bytes, pipeline_score="
                  f"{saved['summary']['pipeline_score']})")
        else:
            print(f"  FAIL  (missing pipeline_score or detection_accuracy)")
        tests_passed += 1 if (has_score and has_det) else 0
        os.remove(path)
    else:
        print(f"  FAIL  (file not found: {path})")

    print("\n[Test 4] summarise() prints full pipeline report?")
    try:
        log.summarise()
        print("  PASS")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")

    print("-" * 60)
    print(f"  Result: {tests_passed}/{total_tests} PASS")
    if tests_passed == total_tests:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
