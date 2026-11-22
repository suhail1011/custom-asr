import sys
import types

import numpy as np
import pytest
import torch

from custom_asr import stream_mic
from custom_asr.stream_mic import AudioError, AudioSource, WavSource, run_stream
from test_streaming import asr, audio  # noqa: F401  (shared fixtures)

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@needs_gpu
def test_fake_realtime_source_prints_partials_then_matches_engine(asr, audio):  # noqa: F811
    asr.set_chunk(1)
    expected = asr.session()
    for i in range(0, len(audio), 1600):
        expected.feed(audio[i : i + 1600])
    expected = expected.finish()

    partials = []
    with WavSource(audio, speed=1.0) as source:
        final = run_stream(asr, source, partials.append)

    assert final == expected
    assert "slushy country roads" in final
    distinct = list(dict.fromkeys(p for p in partials if p))
    assert len(distinct) > 3
    assert all(len(b) <= len(a) + 40 for a, b in zip(distinct, distinct[1:]))  # grows, no wholesale rewrites


@needs_gpu
def test_overrun_drops_blocks_and_warns(asr, audio):  # noqa: F811
    asr.set_chunk(1)
    source = AudioSource(max_blocks=3)
    for _ in range(6):
        source.put(audio[:1600])
    source.blocks.get_nowait()
    source.blocks.put(None)
    warnings = []
    run_stream(asr, source, on_warning=warnings.append)
    assert source.dropped == 3
    assert warnings and "dropped 3" in warnings[0]


def test_missing_portaudio_exits_nonzero_with_message(monkeypatch, capsys):
    def fail():
        raise AudioError("PortAudio is not available (x). Install it with: sudo apt install libportaudio2")

    monkeypatch.setattr(stream_mic, "_sounddevice", fail)
    assert stream_mic.main([]) == 2
    assert "libportaudio2" in capsys.readouterr().err


def test_portaudio_oserror_is_wrapped(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    real_import = __import__

    def fake_import(name, *a, **k):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(AudioError, match="libportaudio2"):
        stream_mic._sounddevice()


def test_no_input_device_exits_nonzero(monkeypatch, capsys):
    fake = types.SimpleNamespace(
        check_input_settings=lambda **k: (_ for _ in ()).throw(ValueError("No input device matching 7")),
    )
    monkeypatch.setattr(stream_mic, "_sounddevice", lambda: fake)
    assert stream_mic.main(["--device", "7"]) == 2
    err = capsys.readouterr().err
    assert "no usable input device" in err and "--list-devices" in err


class _FakeSession:
    def __init__(self):
        self.fed = 0

    def feed(self, block):
        self.fed += len(block)
        return "partial"

    def finish(self):
        return f"final:{self.fed}"


class _FakeASR:
    def session(self):
        return _FakeSession()


def test_ctrl_c_feeds_queued_audio_before_final():
    source = AudioSource()
    real_get = source.blocks.get
    calls = []

    def get(block=True, timeout=None):
        if not block:
            return real_get(block=False)
        calls.append(1)
        if len(calls) == 1:
            return np.zeros(100, dtype=np.float32)
        raise KeyboardInterrupt

    source.blocks.get = get
    source.blocks.put(np.zeros(50, dtype=np.float32))
    assert run_stream(_FakeASR(), source) == "final:150"
    source.blocks.get = real_get


def test_stream_that_stops_unexpectedly_raises():
    source = AudioSource()
    source.ended = True
    with pytest.raises(AudioError, match="stopped unexpectedly"):
        run_stream(_FakeASR(), source)
