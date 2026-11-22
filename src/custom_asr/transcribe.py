"""Smoke test: stream a wav file through the engine and report latency, RTF and VRAM."""

import argparse
import sys
import time

import numpy as np
import soundfile as sf
import torch

from custom_asr.engine import CHUNK_CHOICES, SAMPLE_RATE, StreamingASR


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import torchaudio

        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()
    return audio


def stream(asr: StreamingASR, audio: np.ndarray, piece: int):
    session = asr.session()
    for i in range(0, len(audio), piece):
        session.feed(audio[i : i + piece])
    return session.finish(), session


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav")
    ap.add_argument("--chunk", type=int, nargs="+", choices=CHUNK_CHOICES, default=[1],
                    help="right-context N of att_context_size [70,N]: 0/1/6/13 = 80/160/560/1120 ms")
    ap.add_argument("--compare-offline", action="store_true", help="also run model.transcribe and compare")
    ap.add_argument("--feed-ms", type=int, default=100, help="size of the pieces fed to the engine")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    audio = load_audio(args.wav)
    duration = len(audio) / SAMPLE_RATE
    asr = StreamingASR(chunk=args.chunk[0], device=args.device)
    cuda = asr.device.type == "cuda"
    mismatch = False

    for n in args.chunk:
        asr.set_chunk(n)
        offline = asr.transcribe_offline(audio) if args.compare_offline else None
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        piece = max(1, args.feed_ms * SAMPLE_RATE // 1000)
        t0 = time.perf_counter()
        text, session = stream(asr, audio, piece)
        wall = time.perf_counter() - t0
        times = np.array(session.step_times or [0.0]) * 1000
        peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else 0.0
        print(f"\n[chunk N={n} ({asr.chunk_ms} ms)] {text}")
        print(
            f"  steps={len(times)} latency/chunk ms: mean={times.mean():.1f} p95={np.percentile(times, 95):.1f} "
            f"max={times.max():.1f} | RTF={wall / duration:.3f} | peak VRAM={peak:.0f} MiB"
        )
        if offline is not None:
            same = text == offline
            mismatch |= not same
            print(f"  matches offline: {same}")
            if not same:
                print(f"  offline: {offline}")
    return 1 if mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
