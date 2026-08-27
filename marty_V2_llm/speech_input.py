
import time
import tempfile
import os
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


SAMPLE_RATE        = 16000
CHANNELS           = 1
BLOCK_DURATION_SEC = 0.3
BLOCK_SAMPLES      = int(SAMPLE_RATE * BLOCK_DURATION_SEC)

RMS_THRESHOLD      = 0.015
SILENCE_AFTER_SEC  = 1.5


print("Loading faster-whisper tiny model (CTranslate2, int8) ...")
_model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("Faster-whisper model loaded. Warming up ...")
try:
    import scipy.io.wavfile as _wav
    _warmup_audio = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
    _warmup_path = os.path.join(tempfile.gettempdir(), "whisper_warmup.wav")
    _wav.write(_warmup_path, SAMPLE_RATE, (_warmup_audio * 32767).astype(np.int16))
    _warmup_segments, _ = _model.transcribe(_warmup_path, language="en")
    _ = list(_warmup_segments)
    try:
        os.remove(_warmup_path)
    except OSError:
        pass
    print("Whisper warmed up and ready.")
except RuntimeError as e:
    print(f"Whisper warm-up skipped (memory): {e}")
    print("First transcription may be slower.")


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block ** 2)))


def listen_for_answer(timeout_sec: float = 15.0,
                      stop_event: threading.Event | None = None) -> dict:
    recorded_blocks: list[np.ndarray] = []
    speech_started = False
    silence_start: float | None = None
    interrupted = False

    mic_start = time.time()
    deadline = mic_start + timeout_sec

    print(f"  [mic] Listening (timeout {timeout_sec}s) ...")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", blocksize=BLOCK_SAMPLES) as stream:
        while time.time() < deadline:
            if stop_event is not None and stop_event.is_set():
                print("  [mic] Interrupted by event.")
                interrupted = True
                break

            block, _overflowed = stream.read(BLOCK_SAMPLES)
            block = block.flatten()
            energy = _rms(block)

            if energy >= RMS_THRESHOLD:
                if not speech_started:
                    print("  [mic] Speech detected — recording ...")
                    speech_started = True
                silence_start = None
                recorded_blocks.append(block)

            elif speech_started:
                recorded_blocks.append(block)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_AFTER_SEC:
                    print("  [mic] Silence detected — done recording.")
                    break

    mic_listen_sec = round(time.time() - mic_start, 3)

    if not recorded_blocks:
        print("  [mic] Nothing heard.")
        return {
            "transcript":             "",
            "mic_listen_sec":         mic_listen_sec,
            "whisper_transcribe_sec": 0.0,
            "interrupted":            interrupted,
        }

    audio = np.concatenate(recorded_blocks)

    tmp_path = os.path.join(tempfile.gettempdir(), "marty_answer.wav")
    import scipy.io.wavfile as wav
    wav.write(tmp_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    whisper_start = time.time()
    segments, _info = _model.transcribe(tmp_path, language="en")
    transcript = " ".join(s.text for s in segments).strip()
    whisper_sec = round(time.time() - whisper_start, 3)

    print(f"  [mic] Transcript: '{transcript}'")
    print(f"  [mic] Latency — mic: {mic_listen_sec:.3f}s, "
          f"whisper: {whisper_sec:.3f}s")

    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return {
        "transcript":             transcript,
        "mic_listen_sec":         mic_listen_sec,
        "whisper_transcribe_sec": whisper_sec,
        "interrupted":            interrupted,
    }


if __name__ == "__main__":
    print("=" * 55)
    print("  speech_input.py — self-test (faster-whisper)")
    print("=" * 55)

    tests_passed = 0
    total_tests = 3

    print("\n[Test 1] Faster-whisper model loaded?")
    try:
        assert _model is not None
        print("  PASS")
        tests_passed += 1
    except AssertionError:
        print("  FAIL — model is None")

    print("\n[Test 2] Mic stream opens?")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32", blocksize=BLOCK_SAMPLES) as s:
            data, _ = s.read(BLOCK_SAMPLES)
            assert data.shape[0] == BLOCK_SAMPLES
        print("  PASS")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")

    print("\n[Test 3] Live listen — say 'hello' or wait 5s for timeout ...")
    try:
        result = listen_for_answer(timeout_sec=5)
        ok = (isinstance(result, dict)
              and "transcript" in result
              and "interrupted" in result)
        if ok:
            t = result["transcript"]
            ml = result["mic_listen_sec"]
            wt = result["whisper_transcribe_sec"]
            if t:
                print(f"  Heard: '{t}' (mic={ml:.3f}s, whisper={wt:.3f}s) — PASS")
            else:
                print(f"  Nothing heard (mic={ml:.3f}s, timeout works) — PASS")
            tests_passed += 1
        else:
            print(f"  FAIL — unexpected return: {result}")
    except Exception as e:
        print(f"  FAIL — {e}")

    print("-" * 55)
    print(f"  Result: {tests_passed}/{total_tests} PASS")
    if tests_passed == total_tests:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 55)
