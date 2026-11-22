import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


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


def test_finish_without_enough_audio_returns_text(asr):
    asr.set_chunk(1)
    assert asr.session().finish() == ""
    session = asr.session()
    session.feed(np.zeros(400, dtype=np.float32))
    assert isinstance(session.finish(), str)
    assert session.finish() == session.finish()


def test_offline_transcribe_restores_eval_mode(asr, audio):
    asr.transcribe_offline(audio)
    assert not asr.model.training and not asr.model.encoder.training
