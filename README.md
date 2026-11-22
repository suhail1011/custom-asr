# custom-asr

Local, GPU-accelerated streaming speech recognition with
[`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
(NeMo cache-aware streaming, fp32). Phase 1 is inference and live dictation; fine-tuning on personal
data is a later phase and not implemented yet.

## Setup

Requirements: Python 3.10-3.12, [uv](https://docs.astral.sh/uv/), an NVIDIA GPU with about 2.5 GB of VRAM and CUDA drivers.

```bash
uv sync
sudo apt install libportaudio2   # PortAudio, needed by sounddevice for the microphone
```

The first run downloads the model (~2.5 GB) from Hugging Face into `~/.cache/huggingface`.

### Microphone on WSL2

WSLg exposes the Windows microphone through PulseAudio (`PULSE_SERVER=unix:/mnt/wslg/PulseServer`).
After installing `libportaudio2`, check that an input device shows up:

```bash
uv run python -m custom_asr.stream_mic --list-devices
```

If the list has no input device, or you get a PortAudio/ALSA error, make sure Windows allows microphone access
for desktop apps, then pick a device with `--device N`. PortAudio reaches PulseAudio through ALSA, so you may
also need `sudo apt install libasound2-plugins` (not verified here).

## Usage

Get a sample clip (`data/` is git-ignored; any 16 kHz mono wav works too):

```bash
mkdir -p data && curl -L -o data/sample.flac https://cdn-media.huggingface.co/speech_samples/sample1.flac
uv run python -c "import soundfile as sf; d, sr = sf.read('data/sample.flac'); sf.write('data/sample.wav', d, sr)"
```

Transcribe a wav file in streaming mode (optionally comparing with offline `model.transcribe`):

```bash
uv run python -m custom_asr.transcribe data/sample.wav --chunk 1 --compare-offline
uv run python -m custom_asr.transcribe data/sample.wav --chunk 0 1 6 13   # latency table per chunk size
```

Dictate live from the microphone:

```bash
uv run python -m custom_asr.stream_mic [--chunk 1] [--device N] [--list-devices]
```

The current partial transcript is rewritten on one line while you speak. Press Ctrl+C to stop; the final
transcript is then printed. A missing PortAudio library or input device prints an error and exits with status 2.
If the engine cannot keep up, audio blocks are dropped with a warning on stderr.

### Chunk size and latency

`--chunk N` selects the right-context of the encoder (`att_context_size=[70,N]`). Larger chunks are more
accurate but add delay. Measured on an RTX 1000 Ada laptop GPU (6 GB), fp32, 100 ms feed pieces; peak VRAM was
2467 MiB (about 2.4 GiB) for all sizes. On the sample clip the output matched offline transcription for every size.

| `--chunk` | Chunk length | Encoder step (mean) | Real-time factor |
|-----------|--------------|---------------------|------------------|
| 0         | 80 ms        | 48-58 ms            | 0.62-0.76        |
| 1 (default) | 160 ms     | 63 ms               | 0.42             |
| 6         | 560 ms       | 78 ms               | 0.16             |
| 13        | 1120 ms      | 54-96 ms            | 0.07-0.12        |

Ranges are across runs. Real-time factor below 1 means faster than real time; `--chunk 0` is the tightest margin.

## Tests

```bash
uv run pytest
```

The tests need a CUDA GPU and download a short sample clip into `data/` on first run. The microphone itself is not
exercised by the tests; they replay a wav through a fake real-time source.
