"""Live dictation: capture the microphone and stream it through the engine."""

import argparse
import queue
import shutil
import sys
import threading
import time
from typing import Callable, Iterable

import numpy as np

from custom_asr.engine import CHUNK_CHOICES, SAMPLE_RATE, StreamingASR

BLOCK_MS = 100  # audio delivered to the engine per queue item
QUEUE_BLOCKS = 100  # ~10 s of audio; beyond that the engine is too slow and blocks are dropped


class AudioError(RuntimeError):
    """The microphone cannot be used (no PortAudio, no device, stream failed)."""


class AudioSource:
    """Base source: audio blocks arrive on ``blocks``; ``None`` marks the end of the stream."""

    def __init__(self, max_blocks: int = QUEUE_BLOCKS):
        self.blocks: queue.Queue = queue.Queue(maxsize=max_blocks)
        self.dropped = 0
        self.ended = False  # the capture stopped without the end-of-stream marker

    def put(self, block: np.ndarray) -> None:
        try:
            self.blocks.put_nowait(block)
        except queue.Full:
            self.dropped += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _sounddevice():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:  # package or PortAudio shared library missing
        raise AudioError(
            f"sounddevice/PortAudio is not available ({e}). Install it with: sudo apt install libportaudio2"
        ) from e
    return sd


def list_devices() -> str:
    sd = _sounddevice()
    try:
        return str(sd.query_devices())
    except Exception as e:
        raise AudioError(f"cannot query audio devices: {e}") from e


class MicSource(AudioSource):
    """16 kHz mono float32 microphone capture through sounddevice."""

    def __init__(self, device: int | str | None = None, block_ms: int = BLOCK_MS):
        super().__init__()
        sd = _sounddevice()
        try:
            sd.check_input_settings(device=device, channels=1, samplerate=SAMPLE_RATE, dtype="float32")
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=SAMPLE_RATE * block_ms // 1000,
                device=device,
                callback=self._callback,
                finished_callback=self._finished,
            )
        except Exception as e:
            raise AudioError(f"no usable input device (device={device!r}): {e}. Try --list-devices.") from e
        self.overflows = 0

    def _callback(self, indata, frames, time_info, status):
        if status.input_overflow:
            self.overflows += 1
        self.put(indata[:, 0].copy())

    def _finished(self):
        self.ended = True

    def __enter__(self):
        try:
            self._stream.start()
        except Exception as e:
            raise AudioError(f"cannot start the microphone: {e}") from e
        return self

    def __exit__(self, *exc):
        self._stream.stop()
        self._stream.close()
        return False


class WavSource(AudioSource):
    """Replays audio in real time (or faster with ``speed``); used to test without a microphone."""

    def __init__(self, audio: np.ndarray, block_ms: int = BLOCK_MS, speed: float = 1.0, max_blocks: int = QUEUE_BLOCKS):
        super().__init__(max_blocks)
        self.audio = audio
        self.block = SAMPLE_RATE * block_ms // 1000
        self.interval = block_ms / 1000 / speed
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        next_t = time.perf_counter()
        for i in range(0, len(self.audio), self.block):
            if self._stop.is_set():
                return
            self.put(self.audio[i : i + self.block])
            next_t += self.interval
            time.sleep(max(0.0, next_t - time.perf_counter()))
        while not self._stop.is_set():
            try:
                self.blocks.put(None, timeout=0.1)
                return
            except queue.Full:
                continue

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False


def run_stream(
    asr: StreamingASR,
    source: AudioSource,
    on_partial: Callable[[str], None] = lambda text: None,
    on_warning: Callable[[str], None] = lambda msg: None,
) -> str:
    """Feed the source into one session until it ends or Ctrl+C, and return the final text."""
    session = asr.session()
    reported = 0
    try:
        while True:
            try:
                block = source.blocks.get(timeout=0.1)
            except queue.Empty:
                if source.ended:
                    raise AudioError("the audio stream stopped unexpectedly (device unplugged?)")
                continue
            if block is None:
                break
            on_partial(session.feed(block))
            if source.dropped > reported:
                on_warning(f"audio overrun: dropped {source.dropped - reported} block(s), engine is too slow")
                reported = source.dropped
    except KeyboardInterrupt:
        while True:  # audio captured before Ctrl+C still belongs in the final text
            try:
                block = source.blocks.get_nowait()
            except queue.Empty:
                break
            if block is not None:
                session.feed(block)
    return session.finish()


def _print_partial(text: str) -> None:
    width = shutil.get_terminal_size().columns - 1
    shown = text[-width:] if len(text) > width else text
    sys.stdout.write("\r" + shown.ljust(width))
    sys.stdout.flush()


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk", type=int, choices=CHUNK_CHOICES, default=1,
                    help="right-context N of att_context_size [70,N]: 0/1/6/13 = 80/160/560/1120 ms")
    ap.add_argument("--device", help="input device index or name (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    args = ap.parse_args(argv)

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    try:
        if args.list_devices:
            print(list_devices())
            return 0
        source = MicSource(device=device)  # fail before the slow model load
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        asr = StreamingASR(chunk=args.chunk)
    except KeyboardInterrupt:
        return 130
    print(f"Listening (chunk {asr.chunk_ms} ms). Speak; Ctrl+C to stop.", file=sys.stderr)
    try:
        with source:
            final = run_stream(
                asr, source, _print_partial, lambda msg: print(f"\nwarning: {msg}", file=sys.stderr)
            )
    except AudioError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 2
    sys.stdout.write("\r" + " " * (shutil.get_terminal_size().columns - 1) + "\r")
    print(final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
