import urllib.request
from pathlib import Path

import pytest
import torch

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample.wav"
SAMPLE_URL = "https://cdn-media.huggingface.co/speech_samples/sample1.flac"


@pytest.fixture(scope="session")
def asr():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from custom_asr.engine import StreamingASR

    return StreamingASR(chunk=1)


@pytest.fixture(scope="session")
def audio():
    from custom_asr.transcribe import load_audio

    if not SAMPLE.exists():
        import soundfile as sf

        SAMPLE.parent.mkdir(exist_ok=True)
        flac = SAMPLE.with_suffix(".flac")
        try:
            urllib.request.urlretrieve(SAMPLE_URL, flac)
        except OSError as e:
            pytest.skip(f"sample download failed: {e}")
        data, sr = sf.read(flac)
        sf.write(SAMPLE, data, sr)
    return load_audio(str(SAMPLE))
