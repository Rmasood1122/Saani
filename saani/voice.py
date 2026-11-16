"""Voice: wake word -> speech-to-text -> command, and text-to-speech replies.

Optional dependencies (``pip install "saani[voice]"``):
  sounddevice + numpy  microphone capture
  faster-whisper       local, offline speech recognition
  pyttsx3              fallback speech synthesis (the OS voice is used first)

Everything here is written against small interfaces (frame source, STT, TTS) so
the logic is testable without a microphone.
"""
from __future__ import annotations

import array
import io
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import wave
from typing import Callable, Iterable, Iterator

RATE = 16000
FRAME_MS = 30
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2


def rms(frame: bytes) -> float:
    """Loudness of a little-endian 16-bit PCM frame."""
    if len(frame) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(frame[: len(frame) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class Segmenter:
    """Energy-based voice activity detection: turns a stream of frames into utterances."""

    def __init__(self, threshold: float | None = None, start_frames: int = 3, silence_ms: int = 800,
                 max_s: float = 30.0, min_s: float = 0.3, preroll_frames: int = 8):
        self.threshold = threshold
        self.start_frames, self.preroll_frames = start_frames, preroll_frames
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.max_frames = int(max_s * 1000 / FRAME_MS)
        self.min_frames = int(min_s * 1000 / FRAME_MS)
        self._noise: list[float] = []
        self._pre: list[bytes] = []
        self._buf: list[bytes] = []
        self._loud = self._quiet = self._voiced = 0
        self._active = False

    def reset(self) -> None:
        self._pre, self._buf = [], []
        self._loud = self._quiet = self._voiced = 0
        self._active = False

    def feed(self, frame: bytes) -> bytes | None:
        level = rms(frame)
        if self.threshold is None:            # first second: learn the room's noise floor
            self._noise.append(level)
            if len(self._noise) >= 1000 // FRAME_MS:
                self.threshold = max(350.0, 3.0 * sorted(self._noise)[len(self._noise) // 2])
            return None
        loud = level >= self.threshold
        if not self._active:
            self._pre = (self._pre + [frame])[-self.preroll_frames:]
            self._loud = self._loud + 1 if loud else 0
            if self._loud >= self.start_frames:
                self._active, self._buf, self._quiet, self._voiced = True, list(self._pre), 0, self._loud
            return None
        self._buf.append(frame)
        if loud:
            self._voiced += 1
            self._quiet = 0
        else:
            self._quiet += 1
        if self._quiet >= self.silence_frames or len(self._buf) >= self.max_frames:
            audio, voiced = b"".join(self._buf), self._voiced
            self.reset()
            return audio if voiced >= self.min_frames else None
        return None


def pcm_to_wav(pcm: bytes, path: str | None = None) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    data = buf.getvalue()
    if path:
        with open(path, "wb") as f:
            f.write(data)
    return data


# ------------------------------------------------------------------ wake word
_FILLER = r"(?:hey|hi|ok|okay|hello|yo)"


def extract_command(text: str, wake_words: Iterable[str]) -> tuple[bool, str]:
    """Return (wake word heard, command after it). The wake word must open the sentence."""
    t = " ".join((text or "").strip().split())
    if not t:
        return False, ""
    words = "|".join(re.escape(w) for w in sorted(set(wake_words), key=len, reverse=True) if w)
    if not words:
        return False, ""
    m = re.match(rf"^[\W_]*(?:{_FILLER}[\s,.!]+)?(?:{words})\b[\s,.!?:;-]*", t, re.I)
    if not m:
        return False, ""
    return True, t[m.end():].strip()


class WakeGate:
    """'Saani, do X' fires at once. 'Saani.' alone arms the gate: the next sentence is the command."""

    def __init__(self, wake_words: Iterable[str], arm_seconds: float = 8.0):
        self.wake_words = list(wake_words)
        self.arm_seconds = arm_seconds
        self._armed_until = 0.0
        self.just_armed = False

    def hear(self, text: str, now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        self.just_armed = False
        heard, cmd = extract_command(text, self.wake_words)
        if heard and cmd:
            self._armed_until = 0.0
            return cmd
        if heard:
            self._armed_until = now + self.arm_seconds
            self.just_armed = True
            return None
        if now < self._armed_until and text.strip():
            self._armed_until = 0.0
            return text.strip()
        return None


# ------------------------------------------------------------------------ TTS
class TTS:
    def speak(self, text: str) -> None: ...
    available = True


class NullTTS(TTS):
    available = False

    def speak(self, text: str) -> None:
        pass


class SystemTTS(TTS):
    """The operating system's own voice: `say` (macOS), SAPI (Windows), spd-say/espeak (Linux)."""

    def __init__(self):
        self.cmd = self._detect()
        self.available = self.cmd is not None
        self._lock = threading.Lock()

    @staticmethod
    def _detect() -> list[str] | None:
        if sys.platform == "darwin" and shutil.which("say"):
            return ["say"]
        if os.name == "nt":
            ps = shutil.which("powershell") or shutil.which("pwsh")
            if ps:
                return [ps, "-NoProfile", "-NonInteractive", "-Command",
                        "Add-Type -AssemblyName System.Speech; "
                        "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak([Console]::In.ReadToEnd())"]
            return None
        for exe, args in (("spd-say", ["--wait", "-e"]), ("espeak-ng", ["--stdin"]), ("espeak", ["--stdin"])):
            if shutil.which(exe):
                return [exe] + args
        return None

    def speak(self, text: str) -> None:
        if not self.cmd or not text.strip():
            return
        with self._lock:
            try:   # text goes through stdin, never the command line
                subprocess.run(self.cmd, input=text, text=True, timeout=120,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError):
                pass


class PyttsxTTS(TTS):
    """pyttsx3 must live on one thread, so it gets its own."""

    def __init__(self):
        import pyttsx3  # type: ignore  # noqa: F401  (fail early if missing)
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True, name="saani-tts").start()

    def _loop(self):
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        while True:
            text, done = self._q.get()
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
            done.set()

    def speak(self, text: str) -> None:
        done = threading.Event()
        self._q.put((text, done))
        done.wait(120)


def make_tts(mode: str = "auto") -> TTS:
    if mode == "off":
        return NullTTS()
    if mode in ("auto", "system"):
        s = SystemTTS()
        if s.available or mode == "system":
            return s if s.available else NullTTS()
    try:
        return PyttsxTTS()
    except Exception:
        return NullTTS()


# ------------------------------------------------------------------------ STT
class STT:
    def transcribe(self, pcm: bytes) -> str:
        raise NotImplementedError


class WhisperSTT(STT):
    """Local, offline recognition with faster-whisper."""

    def __init__(self, model: str = "base", language: str = ""):
        from faster_whisper import WhisperModel  # type: ignore
        self.model = WhisperModel(model, device="cpu", compute_type="int8")
        self.language = language or None

    def transcribe(self, pcm: bytes) -> str:
        path = os.path.join(tempfile.gettempdir(), f"saani-{uuid.uuid4().hex}.wav")
        pcm_to_wav(pcm, path)
        try:
            segments, _ = self.model.transcribe(path, language=self.language, vad_filter=True, beam_size=1)
            return " ".join(s.text.strip() for s in segments).strip()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


def multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = "----saani" + uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for k, (filename, data, ctype) in files.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{filename}\"\r\n"
                  f"Content-Type: {ctype}\r\n\r\n".encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


class ApiSTT(STT):
    """Any OpenAI-compatible /audio/transcriptions endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str = "whisper-1", language: str = ""):
        self.url = base_url.rstrip("/") + "/audio/transcriptions"
        self.api_key, self.model, self.language = api_key, model, language

    def transcribe(self, pcm: bytes) -> str:
        fields = {"model": self.model, **({"language": self.language} if self.language else {})}
        body, ctype = multipart(fields, {"file": ("speech.wav", pcm_to_wav(pcm), "audio/wav")})
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={"Content-Type": ctype, "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return str(json.loads(r.read().decode("utf-8")).get("text", "")).strip()


def make_stt(config, vault=None) -> STT:
    if config.get("voice.stt", "local") == "api":
        env = config.get("voice.stt_api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(env) or (vault.get(env) if vault else None)
        if not key:
            raise RuntimeError(f"voice.stt is 'api' but {env} is not set")
        return ApiSTT(config.get("voice.stt_api_base"), key, config.get("voice.stt_api_model", "whisper-1"),
                      config.get("voice.language", ""))
    return WhisperSTT(config.get("voice.stt_model", "base"), config.get("voice.language", ""))


# ------------------------------------------------------------------ microphone
def mic_frames(stop: threading.Event) -> Iterator[bytes]:
    import sounddevice as sd  # type: ignore
    with sd.RawInputStream(samplerate=RATE, blocksize=FRAME_BYTES // 2, dtype="int16", channels=1) as stream:
        while not stop.is_set():
            data, _ = stream.read(FRAME_BYTES // 2)
            yield bytes(data)


def missing_voice_deps(config) -> list[str]:
    import importlib.util
    need = ["sounddevice", "numpy"] + (["faster_whisper"] if config.get("voice.stt", "local") == "local" else [])
    return [m for m in need if importlib.util.find_spec(m) is None]


class VoiceLoop(threading.Thread):
    """Listens continuously and calls ``on_command(text)`` when the wake word is followed by a request."""

    def __init__(self, frames: Callable[[threading.Event], Iterator[bytes]], stt: STT, tts: TTS,
                 wake_words: Iterable[str], on_command: Callable[[str], None],
                 segmenter: Segmenter | None = None, on_error: Callable[[str], None] | None = None):
        super().__init__(daemon=True, name="saani-voice")
        self.frames, self.stt, self.tts = frames, stt, tts
        self.gate = WakeGate(wake_words)
        self.on_command = on_command
        self.on_error = on_error or (lambda msg: None)
        self.segmenter = segmenter or Segmenter()
        self.stop_event = threading.Event()
        self._speaking = threading.Event()
        self.heard: list[str] = []

    def say(self, text: str) -> None:
        """Speak without hearing ourselves."""
        self._speaking.set()
        try:
            self.tts.speak(text)
        finally:
            time.sleep(0.15)
            self.segmenter.reset()
            self._speaking.clear()

    def stop(self) -> None:
        self.stop_event.set()

    def handle_utterance(self, pcm: bytes) -> None:
        try:
            text = self.stt.transcribe(pcm)
        except Exception as e:
            self.on_error(f"speech recognition failed: {e}")
            return
        if not text:
            return
        self.heard = (self.heard + [text])[-20:]
        cmd = self.gate.hear(text)
        if self.gate.just_armed:
            self.say("Yes?")
        elif cmd:
            self.on_command(cmd)

    def run(self) -> None:
        try:
            for frame in self.frames(self.stop_event):
                if self.stop_event.is_set():
                    break
                if self._speaking.is_set():
                    continue
                pcm = self.segmenter.feed(frame)
                if pcm:
                    self.handle_utterance(pcm)
        except Exception as e:
            self.on_error(f"voice loop stopped: {type(e).__name__}: {e}")
