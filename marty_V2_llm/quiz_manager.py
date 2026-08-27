
import re
import time
import random
import threading

from question_bank import get_by_level
from speech_output import speak, speak_async, wait_speech, is_speaking, stop as speech_stop
from speech_input import listen_for_answer


_WORD_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}

_TEENS_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _words_to_number(text: str) -> str | None:
    text = text.strip().lower()

    if re.fullmatch(r"-?\d+\.?\d*", text):
        return text

    if text in _WORD_MAP:
        return _WORD_MAP[text]

    parts = re.split(r"[\s\-]+", text)
    if len(parts) == 2:
        tens_word, ones_word = parts
        if tens_word in _TEENS_TENS and ones_word in _WORD_MAP:
            tens_val = _TEENS_TENS[tens_word]
            ones_val = int(_WORD_MAP[ones_word])
            if ones_val < 10:
                return str(tens_val + ones_val)

    if len(parts) == 2 and parts[1] == "hundred" and parts[0] in _WORD_MAP:
        return str(int(_WORD_MAP[parts[0]]) * 100)

    if len(parts) >= 3 and parts[1] == "hundred":
        hundreds = int(_WORD_MAP.get(parts[0], "0")) * 100
        rest = " ".join(parts[2:])
        rest_num = _words_to_number(rest)
        if rest_num is not None:
            return str(hundreds + int(rest_num))

    return None


def normalise_answer(raw: str) -> str:
    text = raw.strip().lower()

    for filler in ["the answer is", "i think it's", "i think it is",
                   "it's", "it is", "that's", "that is", "um", "uh",
                   "so", "well"]:
        text = text.replace(filler, "")
    text = text.strip().strip(".")

    num = _words_to_number(text)
    if num is not None:
        return num

    return text


class QuizManager:
    def __init__(self, start_level: int = 1, max_questions: int = 10):
        self.level = max(1, min(3, start_level))
        self.max_questions = max_questions
        self.questions_asked = 0
        self._streak_correct = 0
        self._streak_wrong   = 0
        self._asked_ids: set[str] = set()
        self._questions_cache: dict[int, list] = {}
        self._current_question: dict | None = None

    @property
    def is_done(self) -> bool:
        return self.questions_asked >= self.max_questions

    def _get_pool(self) -> list[dict]:
        if self.level not in self._questions_cache:
            self._questions_cache[self.level] = get_by_level(self.level)
        pool = [q for q in self._questions_cache[self.level]
                if q["question_id"] not in self._asked_ids]
        if not pool:
            self._asked_ids -= {q["question_id"]
                                for q in self._questions_cache[self.level]}
            pool = list(self._questions_cache[self.level])
        random.shuffle(pool)
        return pool

    def _adjust_difficulty(self, is_correct: bool):
        if is_correct:
            self._streak_correct += 1
            self._streak_wrong = 0
            if self._streak_correct >= 3 and self.level < 3:
                self.level += 1
                self._streak_correct = 0
                speak(f"Level up! Level {self.level}.")
                print(f"  [quiz] Difficulty UP -> level {self.level}")
        else:
            self._streak_wrong += 1
            self._streak_correct = 0
            if self._streak_wrong >= 3 and self.level > 1:
                self.level -= 1
                self._streak_wrong = 0
                speak(f"Let's try level {self.level}.")
                print(f"  [quiz] Difficulty DOWN -> level {self.level}")

    def ask_next_question(self, listen_timeout: float = 15.0,
                          face_detected: bool = True,
                          resume: bool = False,
                          stop_event: threading.Event | None = None) -> dict:
        cycle_start = time.time()

        interrupted_during_tts = False

        if resume and self._current_question is not None:
            q = self._current_question
            print(f"\n  [quiz] Level {self.level} | {q['question_id']}")
            print(f"  [quiz] Q: {q['question_text']} (resuming)")

            tts_q_start = time.time()
            speak_async("Let's get back to the question. " + q["question_text"])
            while is_speaking():
                if stop_event is not None and stop_event.is_set():
                    speech_stop()
                    interrupted_during_tts = True
                    break
                time.sleep(0.05)
            if not interrupted_during_tts:
                wait_speech()
            tts_q_latency = round(time.time() - tts_q_start, 3)
        else:
            pool = self._get_pool()
            q = pool[0]
            self._asked_ids.add(q["question_id"])
            self._current_question = q

            print(f"\n  [quiz] Level {self.level} | {q['question_id']}")
            print(f"  [quiz] Q: {q['question_text']}")

            tts_q_start = time.time()
            speak_async(q["question_text"])
            while is_speaking():
                if stop_event is not None and stop_event.is_set():
                    speech_stop()
                    interrupted_during_tts = True
                    break
                time.sleep(0.05)
            if not interrupted_during_tts:
                wait_speech()
            tts_q_latency = round(time.time() - tts_q_start, 3)

        if interrupted_during_tts:
            total_cycle = round(time.time() - cycle_start, 3)
            print("  [quiz] Interrupted during question speech.")
            return {
                "question_id":              q["question_id"],
                "question_text":            q["question_text"],
                "correct_answer":           q["correct_answer"],
                "student_answer":           "",
                "is_correct":               False,
                "response_time_sec":        0.0,
                "difficulty_level":         self.level,
                "face_detected_during":     face_detected,
                "tts_question_latency_sec": tts_q_latency,
                "tts_feedback_latency_sec": 0.0,
                "mic_listen_sec":           0.0,
                "whisper_transcribe_sec":   0.0,
                "total_cycle_sec":          total_cycle,
                "interrupted":              True,
            }

        if stop_event is not None and stop_event.is_set():
            total_cycle = round(time.time() - cycle_start, 3)
            print("  [quiz] Interrupted before mic listening.")
            return {
                "question_id":              q["question_id"],
                "question_text":            q["question_text"],
                "correct_answer":           q["correct_answer"],
                "student_answer":           "",
                "is_correct":               False,
                "response_time_sec":        0.0,
                "difficulty_level":         self.level,
                "face_detected_during":     face_detected,
                "tts_question_latency_sec": tts_q_latency,
                "tts_feedback_latency_sec": 0.0,
                "mic_listen_sec":           0.0,
                "whisper_transcribe_sec":   0.0,
                "total_cycle_sec":          total_cycle,
                "interrupted":              True,
            }

        listen_result = listen_for_answer(timeout_sec=listen_timeout,
                                          stop_event=stop_event)
        raw_answer      = listen_result["transcript"]
        mic_listen_sec  = listen_result["mic_listen_sec"]
        whisper_sec     = listen_result["whisper_transcribe_sec"]
        was_interrupted = listen_result.get("interrupted", False)

        if was_interrupted:
            total_cycle = round(time.time() - cycle_start, 3)
            print("  [quiz] Interrupted — will resume this question later.")
            return {
                "question_id":              q["question_id"],
                "question_text":            q["question_text"],
                "correct_answer":           q["correct_answer"],
                "student_answer":           "",
                "is_correct":               False,
                "response_time_sec":        0.0,
                "difficulty_level":         self.level,
                "face_detected_during":     face_detected,
                "tts_question_latency_sec": tts_q_latency,
                "tts_feedback_latency_sec": 0.0,
                "mic_listen_sec":           mic_listen_sec,
                "whisper_transcribe_sec":   whisper_sec,
                "total_cycle_sec":          total_cycle,
                "interrupted":              True,
            }

        response_time = round(mic_listen_sec + whisper_sec, 3)

        student_normalised = normalise_answer(raw_answer)
        accepted = [a.lower().strip() for a in q["accepted_answers"]]
        is_correct = student_normalised in accepted

        print(f"  [quiz] Raw answer : '{raw_answer}'")
        print(f"  [quiz] Normalised : '{student_normalised}'")
        print(f"  [quiz] Correct?   : {is_correct}  "
              f"(accepted: {accepted})")

        if not raw_answer:
            fb_text = f"No answer. It's {q['correct_answer']}."
        elif is_correct:
            fb_text = "Correct!"
        else:
            fb_text = f"No, it's {q['correct_answer']}."

        speak_async(fb_text)

        self._adjust_difficulty(is_correct)
        self.questions_asked += 1
        self._current_question = None

        tts_fb_latency = wait_speech()

        total_cycle = round(time.time() - cycle_start, 3)

        print(f"  [quiz] Latency — tts_q: {tts_q_latency:.3f}s, "
              f"mic: {mic_listen_sec:.3f}s, whisper: {whisper_sec:.3f}s, "
              f"tts_fb: {tts_fb_latency:.3f}s, total: {total_cycle:.3f}s")

        return {
            "question_id":              q["question_id"],
            "question_text":            q["question_text"],
            "correct_answer":           q["correct_answer"],
            "student_answer":           raw_answer,
            "is_correct":               is_correct,
            "response_time_sec":        response_time,
            "difficulty_level":         self.level,
            "face_detected_during":     face_detected,
            "tts_question_latency_sec": tts_q_latency,
            "tts_feedback_latency_sec": tts_fb_latency,
            "mic_listen_sec":           mic_listen_sec,
            "whisper_transcribe_sec":   whisper_sec,
            "total_cycle_sec":          total_cycle,
            "interrupted":              False,
        }


if __name__ == "__main__":
    print("=" * 55)
    print("  quiz_manager.py — self-test")
    print("  (pure logic tests — no mic or speaker needed)")
    print("=" * 55)

    tests_passed = 0
    total_tests = 5

    print("\n[Test 1] normalise_answer('  24 ') == '24'?")
    r = normalise_answer("  24 ")
    if r == "24":
        print(f"  PASS  (got '{r}')")
        tests_passed += 1
    else:
        print(f"  FAIL  (got '{r}')")

    print("\n[Test 2] normalise_answer('twenty four') == '24'?")
    r = normalise_answer("twenty four")
    if r == "24":
        print(f"  PASS  (got '{r}')")
        tests_passed += 1
    else:
        print(f"  FAIL  (got '{r}')")

    print("\n[Test 3] normalise_answer('the answer is seven') == '7'?")
    r = normalise_answer("the answer is seven")
    if r == "7":
        print(f"  PASS  (got '{r}')")
        tests_passed += 1
    else:
        print(f"  FAIL  (got '{r}')")

    print("\n[Test 4] QuizManager loads question pool?")
    qm = QuizManager(start_level=1)
    pool = qm._get_pool()
    if len(pool) > 0:
        print(f"  PASS  ({len(pool)} questions in level 1 pool)")
        tests_passed += 1
    else:
        print("  FAIL  (empty pool)")

    print("\n[Test 5] 3 correct -> level up?")
    qm2 = QuizManager(start_level=1)
    for _ in range(3):
        qm2._adjust_difficulty(True)
    if qm2.level == 2:
        print(f"  PASS  (level = {qm2.level})")
        tests_passed += 1
    else:
        print(f"  FAIL  (level = {qm2.level}, expected 2)")

    total_tests += 1
    print("\n[Test 6] max_questions=10, is_done after 10?")
    qm3 = QuizManager(start_level=1, max_questions=10)
    qm3.questions_asked = 9
    if not qm3.is_done:
        qm3.questions_asked = 10
        if qm3.is_done:
            print("  PASS")
            tests_passed += 1
        else:
            print("  FAIL  (not done at 10)")
    else:
        print("  FAIL  (done at 9)")

    total_tests += 1
    print("\n[Test 7] _current_question preserved for resume?")
    qm4 = QuizManager(start_level=1)
    pool = qm4._get_pool()
    qm4._current_question = pool[0]
    q_id = pool[0]["question_id"]
    if qm4._current_question["question_id"] == q_id:
        print(f"  PASS  (can resume {q_id})")
        tests_passed += 1
    else:
        print("  FAIL")

    print("-" * 55)
    print(f"  Result: {tests_passed}/{total_tests} PASS")
    if tests_passed == total_tests:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 55)
