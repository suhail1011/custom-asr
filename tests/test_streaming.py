import urllib.request
from pathlib import Path

import numpy as np
import pytest
import torch

from custom_asr.engine import StreamingASR
from custom_asr.transcribe import load_audio

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample.wav"
SAMPLE_URL = "https://cdn-media.huggingface.co/speech_samples/sample1.flac"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture(scope="module")
def asr():
    return StreamingASR(chunk=1)


@pytest.fixture(scope="module")
def audio():
    if not SAMPLE.exists():
        import soundfile as sf

        SAMPLE.parent.mkdir(exist_ok=True)
        flac = SAMPLE.with_suffix(".flac")
        urllib.request.urlretrieve(SAMPLE_URL, flac)
        data, sr = sf.read(flac)
        sf.write(SAMPLE, data, sr)
    return load_audio(str(SAMPLE))


def stream_in_pieces(asr, audio, sizes):
    session = asr.session()
    i, k = 0, 0
    while i < len(audio):
        n = sizes[k % len(sizes)]
        session.feed(audio[i : i + n])
        i += n
        k += 1
    return session.finish()


@pytest.mark.parametrize("chunk", [0, 1, 6, 13])
def test_streaming_matches_offline(asr, audio, chunk):
    asr.set_chunk(chunk)
    offline = asr.transcribe_offline(audio)
    assert stream_in_pieces(asr, audio, [1600]) == offline


def test_arbitrary_piece_sizes_give_same_text(asr, audio):
    asr.set_chunk(1)
    reference = stream_in_pieces(asr, audio, [1600])
    odd = stream_in_pieces(asr, audio, [1, 37, 159, 160, 161, 5000, 333])
    assert odd == reference
    assert "slushy country roads" in reference


def test_partials_arrive_before_finish(asr, audio):
    asr.set_chunk(1)
    session = asr.session()
    partials = [session.feed(audio[i : i + 800]) for i in range(0, len(audio) // 2, 800)]
    assert partials[-1].startswith("Going along")
    assert len(partials[-1]) > len(partials[len(partials) // 4])
    final = session.finish()
    assert final.startswith(partials[-1][:20])
    with pytest.raises(RuntimeError):
        session.feed(np.zeros(160, dtype=np.float32))
