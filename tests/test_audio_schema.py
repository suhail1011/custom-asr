import numpy as np
import pytest
import soundfile as sf

from custom_asr_benchmark.audio import materialize_audio, prepare_waveform
from custom_asr_benchmark.errors import DataError
from custom_asr_benchmark.schema import validate_rows


def test_audio_is_deterministic_mono_16khz_and_clipped():
    waveform = prepare_waveform(
        {"array": np.array([[0.1, 0.1], [2.0, 2.0]]), "sampling_rate": 16000},
        16000,
        1.0,
    )
    assert waveform.dtype == np.float32
    assert waveform.shape == (2,)
    assert np.max(np.abs(waveform)) == pytest.approx(1.0)


def test_materialize_uses_content_addresses_and_manifest(tmp_path):
    row = {
        "id": "sample",
        "sampling_rate": 24000,
        "transcript": "Hello",
        "clean_audio": {"array": np.arange(2400, dtype=np.float32) / 2400, "sampling_rate": 24000},
        "noisy_audio": {"array": np.arange(2400, dtype=np.float32) / 4800, "sampling_rate": 24000},
    }
    first, path = materialize_audio([row], tmp_path, max_concurrent=2)
    second, _ = materialize_audio([row], tmp_path, max_concurrent=2)
    assert first == second
    assert [item["condition"] for item in second] == ["clean", "noisy"]
    for item in second:
        data, rate = sf.read(item["audio_filepath"], dtype="float32")
        assert rate == 16000
        assert len(data) == 1600
    assert len(list((tmp_path / "audio").glob("*.wav"))) == 2
    assert len(path.read_text().splitlines()) == 2


def test_schema_rejects_duplicate_ids():
    row = {
        "id": "x",
        "sampling_rate": 16000,
        "transcript": "x",
        "noisy_audio": {"array": [0], "sampling_rate": 16000},
        "clean_audio": {"array": [0], "sampling_rate": 16000},
    }
    with pytest.raises(DataError, match="duplicate"):
        validate_rows([row, dict(row)], blind=False)


def test_schema_rejects_unavailable_transcript_sentinel():
    row = {
        "id": "x",
        "sampling_rate": 16000,
        "transcript": "<not-available>",
        "noisy_audio": {"array": [0], "sampling_rate": 16000},
        "clean_audio": {"array": [0], "sampling_rate": 16000},
    }
    with pytest.raises(DataError, match="missing or empty"):
        validate_rows([row], blind=False)
