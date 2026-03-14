from __future__ import annotations

import os
import tempfile
from typing import Optional


def record_wav(seconds: float = 6.0, sample_rate: int = 16000) -> str:
    """
    Records microphone audio to a temporary WAV file.
    Requires: sounddevice, numpy, scipy
    """
    try:
        import numpy as np
        import sounddevice as sd
        from scipy.io import wavfile
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Voice requires sounddevice+numpy+scipy installed.") from exc

    frames = int(seconds * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()

    fd, path = tempfile.mkstemp(prefix="support_voice_", suffix=".wav")
    os.close(fd)
    wavfile.write(path, sample_rate, np.squeeze(audio))
    return path


def transcribe_wav(path: str) -> str:
    """
    Transcription options:
    1) OpenAI Whisper API if OPENAI_API_KEY is set
    2) Hugging Face ASR pipeline if HF_ASR_MODEL is set (may download weights)
    """
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install openai to use Whisper API transcription.") from exc

        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        with open(path, "rb") as f:
            r = client.audio.transcriptions.create(
                model=os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1"),
                file=f,
            )
        text = (getattr(r, "text", None) or "").strip()
        if not text:
            raise RuntimeError("No transcription returned.")
        return text

    hf_model = os.getenv("HF_ASR_MODEL")
    if hf_model:
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install transformers to use HF ASR.") from exc
        asr = pipeline("automatic-speech-recognition", model=hf_model)
        r = asr(path)
        text = str(r.get("text", "")).strip()
        if not text:
            raise RuntimeError("No transcription returned.")
        return text

    raise RuntimeError("No transcription backend configured. Set OPENAI_API_KEY or HF_ASR_MODEL.")


def speak(text: str) -> None:
    """
    Optional offline TTS if pyttsx3 is installed.
    """
    try:
        import pyttsx3  # type: ignore
    except Exception:
        return
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        return
