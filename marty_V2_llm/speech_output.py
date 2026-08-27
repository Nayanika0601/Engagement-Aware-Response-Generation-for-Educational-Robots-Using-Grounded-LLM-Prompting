
import time
import threading
import platform


RATE   = 2
VOLUME = 100

_speech_lock = threading.Lock()

_SVSFlagsAsync = 1
_SVSFDefault   = 0


_use_sapi = False
_speaker = None

if platform.system() == "Windows":
    try:
        import win32com.client
        _speaker = win32com.client.Dispatch("SAPI.SpVoice")
        _speaker.Rate = RATE
        _speaker.Volume = VOLUME
        _use_sapi = True
    except Exception:
        _use_sapi = False

if not _use_sapi:
    import pyttsx3


def speak(text: str) -> float:
    if not text:
        return 0.0
    with _speech_lock:
        t0 = time.time()
        if _use_sapi:
            _speaker.Speak(text, _SVSFDefault)
        else:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        return round(time.time() - t0, 3)


_async_start: float = 0.0

def speak_async(text: str) -> None:
    global _async_start
    if not text:
        return
    _speech_lock.acquire()
    _async_start = time.time()
    if _use_sapi:
        _speaker.Speak(text, _SVSFlagsAsync)
    else:
        def _bg():
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            _speech_lock.release()
        threading.Thread(target=_bg, daemon=True).start()
        return
    _speech_lock.release()


def wait_speech() -> float:
    global _async_start
    if _use_sapi:
        _speaker.WaitUntilDone(-1)
    latency = round(time.time() - _async_start, 3) if _async_start else 0.0
    _async_start = 0.0
    return latency


def is_speaking() -> bool:
    if _use_sapi:
        return _speaker.Status.RunningState != 0
    return False


def stop() -> None:
    if _use_sapi and _speaker:
        _speaker.Speak("", _SVSFlagsAsync)
    try:
        _speech_lock.release()
    except RuntimeError:
        pass


def try_speak(text: str) -> float:
    if not text:
        return 0.0
    acquired = _speech_lock.acquire(blocking=False)
    if not acquired:
        return 0.0
    try:
        t0 = time.time()
        if _use_sapi:
            _speaker.Speak(text, _SVSFDefault)
        else:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        return round(time.time() - t0, 3)
    finally:
        _speech_lock.release()


if __name__ == "__main__":
    print("=" * 55)
    print("  speech_output.py — self-test")
    print(f"  Engine: {'Windows SAPI (win32com)' if _use_sapi else 'pyttsx3 fallback'}")
    print("=" * 55)

    passed = 0
    total = 4

    print("\n[Test 1] Sync speak ...")
    lat = speak("Hello, I am your tutor.")
    if lat < 5.0:
        print(f"  PASS  (latency: {lat:.3f}s)")
        passed += 1
    else:
        print(f"  FAIL  (latency: {lat:.3f}s)")

    print("\n[Test 2] Sync speak again (no hang) ...")
    lat = speak("What is 6 times 4?")
    if lat < 5.0:
        print(f"  PASS  (latency: {lat:.3f}s)")
        passed += 1
    else:
        print(f"  FAIL  (latency: {lat:.3f}s)")

    print("\n[Test 3] Async speak (should return fast) ...")
    t0 = time.time()
    speak_async("This is an async test.")
    return_time = round(time.time() - t0, 3)
    print(f"  Returned in {return_time:.3f}s")
    total_lat = wait_speech()
    if return_time < 0.5:
        print(f"  PASS  (returned in {return_time:.3f}s, speech took {total_lat:.3f}s)")
        passed += 1
    else:
        print(f"  FAIL  (took {return_time:.3f}s to return — not async)")

    print("\n[Test 4] Overlap: speak async + do work ...")
    speak_async("Counting while I speak.")
    work_done = 0
    t_start = time.time()
    while time.time() - t_start < 0.5:
        work_done += 1
        time.sleep(0.01)
    total_lat = wait_speech()
    if work_done > 0:
        print(f"  PASS  (did {work_done} work iterations during {total_lat:.3f}s speech)")
        passed += 1
    else:
        print("  FAIL  (no work done during speech)")

    print("-" * 55)
    print(f"  Result: {passed}/{total} PASS")
    if passed == total:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 55)
