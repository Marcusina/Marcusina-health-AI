"""
Tests for /transcribe audio sourcing (base64 or fetched audio_url).

Covers the request contract (exactly one source) and the SSRF + size guards on
URL fetching — no network: the download is exercised with a fake httpx stream.
"""

from __future__ import annotations

import base64
import os

import pytest
from pydantic import ValidationError

from app.models.schemas import TranscribeRequest
import app.tasks.consultation_tasks as ct


# ── request contract ──────────────────────────────────────────────────────────

def test_request_requires_exactly_one_source():
    # both → error
    with pytest.raises(ValidationError):
        TranscribeRequest(session_id="s", audio_base64="abc", audio_url="https://x/a.wav")
    # neither → error
    with pytest.raises(ValidationError):
        TranscribeRequest(session_id="s")
    # each alone → ok
    assert TranscribeRequest(session_id="s", audio_base64="abc").audio_base64 == "abc"
    assert TranscribeRequest(session_id="s", audio_url="https://x/a.wav").audio_url


# ── SSRF guard ────────────────────────────────────────────────────────────────

def test_validate_audio_url_scheme():
    ct._validate_audio_url("https://ok.example/a.wav")          # fine
    with pytest.raises(ValueError):
        ct._validate_audio_url("file:///etc/passwd")
    with pytest.raises(ValueError):
        ct._validate_audio_url("ftp://host/a.wav")


def test_validate_audio_url_host_allowlist(monkeypatch):
    monkeypatch.setattr(ct.settings, "AUDIO_FETCH_ALLOWED_HOSTS", ["storage.marcusina.dev"])
    ct._validate_audio_url("https://storage.marcusina.dev/a.wav")     # allowed
    with pytest.raises(ValueError):
        ct._validate_audio_url("https://evil.example/a.wav")          # blocked


# ── audio acquisition ─────────────────────────────────────────────────────────

def test_acquire_audio_from_base64(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = base64.b64encode(b"RIFFfake-wav-bytes").decode()
    path = ct._acquire_audio(data, None, "wav")
    try:
        assert os.path.exists(path)
        with open(path, "rb") as f:
            assert f.read() == b"RIFFfake-wav-bytes"
    finally:
        os.unlink(path)


def test_download_audio_enforces_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(ct.settings, "AUDIO_MAX_MB", 5)
    monkeypatch.setattr(ct.settings, "AUDIO_FETCH_ALLOWED_HOSTS", [])

    class _Resp:
        def raise_for_status(self): pass
        def iter_bytes(self):
            for _ in range(10):                       # 10 MB > 5 MB cap
                yield b"x" * (1024 * 1024)

    class _Stream:
        def __enter__(self): return _Resp()
        def __exit__(self, *a): return False

    monkeypatch.setattr(ct.httpx, "stream", lambda *a, **k: _Stream())
    dest = tmp_path / "out.wav"
    with pytest.raises(ValueError, match="AUDIO_MAX_MB"):
        ct._download_audio("https://ok.example/a.wav", str(dest))


# ── stereo diarization (real channel split, fake whisper) ─────────────────────

class _Seg:
    def __init__(self, start, end, text, nsp=0.1):
        self.start, self.end, self.text, self.no_speech_prob = start, end, text, nsp

class _Info:
    language = "en"
    language_probability = 0.99

class _FakeWhisper:
    """Returns distinct segments per call (call 1 = left, call 2 = right)."""
    def __init__(self):
        self.calls = 0
    def transcribe(self, audio, **kw):
        self.calls += 1
        if self.calls == 1:
            return iter([_Seg(0.0, 2.0, "what brings you in today")]), _Info()
        return iter([_Seg(2.0, 4.0, "i have had a cough")]), _Info()


def _write_wav(path, data, sr=16000):
    from scipy.io import wavfile
    wavfile.write(str(path), sr, data)


def test_transcribe_stereo_labels_by_channel(tmp_path):
    import numpy as np
    sr = 16000; t = np.linspace(0, 1, sr, endpoint=False)
    left = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    right = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav = tmp_path / "stereo.wav"
    _write_wav(wav, np.stack([left, right], axis=1))

    fields, _ms = ct._transcribe(_FakeWhisper(), str(wav), None,
                                 diarize_stereo=True, channel_roles=["doctor", "patient"])
    assert fields["diarized"] is True
    assert fields["speakers"] == ["doctor", "patient"]
    assert fields["transcript"].startswith("doctor: what brings you in")
    assert "patient: i have had a cough" in fields["transcript"]
    # segments are labeled and time-ordered
    assert [s["speaker"] for s in fields["segments"]] == ["doctor", "patient"]


def test_transcribe_mono_falls_back_to_flat(tmp_path):
    import numpy as np
    sr = 16000; t = np.linspace(0, 1, sr, endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    wav = tmp_path / "mono.wav"
    _write_wav(wav, mono)

    fields, _ms = ct._transcribe(_FakeWhisper(), str(wav), None,
                                 diarize_stereo=True, channel_roles=["doctor", "patient"])
    assert fields["diarized"] is False        # identical channels detected as mono
    assert "segments" not in fields


def test_request_channel_roles_must_be_pair():
    with pytest.raises(ValidationError):
        TranscribeRequest(session_id="s", audio_url="https://x/a.wav",
                          diarize_stereo=True, channel_roles=["only_one"])
