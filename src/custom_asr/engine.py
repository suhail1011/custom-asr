"""Cache-aware streaming ASR on top of NeMo's Nemotron streaming model.

Audio is fed incrementally as 16 kHz mono float32. Mel features are computed
on the fly with enough raw-audio context that they equal the offline features,
then cut into encoder chunks the same way NeMo's ``CacheAwareStreamingAudioBuffer``
does, so the streaming result matches ``model.transcribe``.
"""

import copy
import time

import numpy as np
import torch

MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
SAMPLE_RATE = 16000
HOP = 160  # samples per mel frame (10 ms)
FEATURE_MARGIN = 256  # samples either side of a frame centre that influence it (n_fft // 2)
LEFT_CONTEXT_FRAMES = 4  # extra frames recomputed on the left so segment edges stay exact
CHUNK_CHOICES = (0, 1, 6, 13)  # right-context N in att_context_size [70, N]


class StreamingASR:
    """Loads the model once; create one StreamingSession per utterance/stream."""

    def __init__(self, chunk: int = 1, device: str = "cuda", model_name: str = MODEL_NAME):
        from omegaconf import OmegaConf

        import nemo.collections.asr as nemo_asr

        self.device = torch.device(device)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location=self.device)
        model.eval()
        self.model = model
        self.set_chunk(chunk)

        cfg = copy.deepcopy(model.cfg.preprocessor)
        OmegaConf.set_struct(cfg, False)
        cfg.dither = 0.0
        cfg.pad_to = 0
        self.preprocessor = model.from_config_dict(cfg).to(self.device).eval()

    def set_chunk(self, chunk: int) -> None:
        """Select the right-context N of att_context_size [70, N]: 0/1/6/13 = 80/160/560/1120 ms.

        Applies to sessions created afterwards; do not call while a session is open.
        """
        if chunk not in CHUNK_CHOICES:
            raise ValueError(f"chunk must be one of {CHUNK_CHOICES}, got {chunk}")
        self.model.encoder.set_default_att_context_size([70, chunk])

    @property
    def chunk_ms(self) -> int:
        return self.model.encoder.streaming_cfg.shift_size[1] * 10

    def session(self) -> "StreamingSession":
        return StreamingSession(self)

    @torch.no_grad()
    def transcribe_offline(self, audio: np.ndarray) -> str:
        out = self.model.transcribe([audio.astype(np.float32)], verbose=False)
        self.model.eval()  # transcribe() leaves encoder/decoder in train mode, which enables dropout
        hyp = out[0] if isinstance(out, (list, tuple)) else out
        return hyp.text if hasattr(hyp, "text") else str(hyp)


class StreamingSession:
    """One stream. ``feed`` accepts any number of samples; ``finish`` flushes the tail."""

    def __init__(self, asr: StreamingASR):
        self.asr = asr
        self.model = asr.model
        enc = self.model.encoder
        self.cfg = enc.streaming_cfg
        self.n_mels = enc._feat_in
        get_frames = getattr(getattr(enc, "pre_encode", None), "get_sampling_frames", None)
        # minimum mel frames the subsampling needs for the first / later chunks, when the encoder reports it
        self.sampling_frames = get_frames() if get_frames else None

        self.cache = enc.get_initial_cache_state(batch_size=1)
        self.hyps = None
        self.text = ""

        self._audio = np.zeros(0, dtype=np.float32)
        self._audio_start = 0  # absolute sample index of _audio[0]
        self._total_samples = 0
        self._mel = torch.zeros(1, self.n_mels, 0, device=asr.device)
        self._mel_start = 0  # absolute frame index of _mel[..., 0]
        self._mel_frames = 0  # frames computed so far (absolute count)
        self._step = 0
        self._pos = 0  # absolute frame index where the next chunk starts
        self.finished = False
        self.step_times: list[float] = []  # seconds per encoder step, includes sync

    @torch.no_grad()
    def feed(self, samples: np.ndarray) -> str:
        """Append audio and return the current partial transcript."""
        if self.finished:
            raise RuntimeError("session already finished")
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._audio = np.concatenate([self._audio, samples])
        self._total_samples += len(samples)
        # frames whose full +-FEATURE_MARGIN sample window is already available
        stable = max(0, (self._total_samples - FEATURE_MARGIN - 1) // HOP + 1)
        self._compute_frames(stable, final=False)
        self._run_steps(final=False)
        return self.text

    @torch.no_grad()
    def finish(self) -> str:
        """Flush remaining audio, including the right-edge frames, and return the final transcript."""
        if self.finished:
            return self.text
        self._compute_frames(0, final=True)
        self._run_steps(final=True)
        self.finished = True
        return self.text

    def _compute_frames(self, stable: int, final: bool) -> None:
        if not final and stable <= self._mel_frames:
            return
        if self._total_samples == 0:
            return  # nothing was fed; the preprocessor cannot take an empty signal
        first = max(0, self._mel_frames - LEFT_CONTEXT_FRAMES)
        offset = first * HOP - self._audio_start
        seg = torch.from_numpy(self._audio[offset:]).unsqueeze(0).to(self.asr.device)
        length = torch.tensor([seg.shape[1]], device=self.asr.device)
        feats, _ = self.asr.preprocessor(input_signal=seg, length=length)
        end = first + feats.shape[-1] if final else stable
        new = feats[:, :, self._mel_frames - first : end - first]
        self._mel = torch.cat([self._mel, new], dim=-1)
        self._mel_frames = end
        # keep LEFT_CONTEXT_FRAMES of audio so the next segment recomputes its left edge exactly
        drop_audio = max(0, (self._mel_frames - LEFT_CONTEXT_FRAMES) * HOP - self._audio_start)
        self._audio = self._audio[drop_audio:]
        self._audio_start += drop_audio

    def _chunk_params(self):
        first = self._step == 0
        size = self.cfg.chunk_size
        shift = self.cfg.shift_size
        pre = self.cfg.pre_encode_cache_size
        i = 0 if first else 1
        return (
            size[i] if isinstance(size, list) else size,
            shift[i] if isinstance(shift, list) else shift,
            pre[i] if isinstance(pre, list) else pre,
        )

    def _run_steps(self, final: bool) -> None:
        while True:
            size, shift, pre_size = self._chunk_params()
            avail = self._mel_frames - self._pos
            if final:
                if avail <= 0:
                    return
                if self.sampling_frames is not None:
                    need = self.sampling_frames[0] if self._step == 0 else self.sampling_frames[1]
                    if min(avail, size) < need:
                        return
            elif avail <= size:  # keep the last chunk for finish(), which needs keep_all_outputs
                return
            self._step_once(size, shift, pre_size, last=final and avail <= size)
            if final and self._pos >= self._mel_frames:
                return

    def _step_once(self, size: int, shift: int, pre_size: int, last: bool) -> None:
        lo = self._pos - self._mel_start
        chunk = self._mel[:, :, lo : lo + size]
        cache = self._mel[:, :, max(0, lo - pre_size) : lo]
        pad = torch.zeros(1, self.n_mels, pre_size - cache.shape[-1], device=chunk.device, dtype=chunk.dtype)
        added = pre_size  # zero padding plus real cache frames
        chunk = torch.cat([pad, cache, chunk], dim=-1)
        length = min(self._mel_frames - self._pos + added, chunk.shape[-1])
        lengths = torch.tensor([length], device=chunk.device)

        is_cuda = self.asr.device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        drop = 0 if self._step == 0 else self.cfg.drop_extra_pre_encoded
        ch, tm, ln = self.cache
        (_, texts, ch, tm, ln, self.hyps) = self.model.conformer_stream_step(
            processed_signal=chunk,
            processed_signal_length=lengths,
            cache_last_channel=ch,
            cache_last_time=tm,
            cache_last_channel_len=ln,
            keep_all_outputs=last,
            previous_hypotheses=self.hyps,
            drop_extra_pre_encoded=drop,
            return_transcription=True,
        )
        if is_cuda:
            torch.cuda.synchronize()
        self.step_times.append(time.perf_counter() - t0)
        self.cache = (ch, tm, ln)
        hyp = texts[0]
        self.text = hyp.text if hasattr(hyp, "text") else str(hyp)

        self._step += 1
        self._pos += shift
        # keep the pre-encode cache plus one frame of slack for the next chunk
        keep_from = max(self._mel_start, self._pos - self.cfg.pre_encode_cache_size[1] - 1)
        if keep_from > self._mel_start:
            self._mel = self._mel[:, :, keep_from - self._mel_start :]
            self._mel_start = keep_from
