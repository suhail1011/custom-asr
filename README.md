# Custom ASR benchmark

Local, pinned evaluation for `urgent-challenge/urgent2024_official` with NVIDIA's
approved `nemotron-speech-streaming-en-0.6b` NeMo checkpoint. The pipeline never
uploads runs or uses paid services.

## Setup

```bash
uv sync
cp benchmark.example.toml benchmark.toml
```

The example pins the currently approved full Hugging Face commit SHAs. Verify the
resolved settings, and change those SHAs only through an explicit benchmark
revision decision:

```bash
uv run urgent-asr config-check --config benchmark.toml
```

Only the dataset's **CC BY-NC-SA 4.0** terms and the model's **NVIDIA Open Model
License** are approved for this local workflow. Downloading requires you to
accept and comply with those terms.

## Split discipline

Use `validation` for development. `test_nonblind` is final, single-use scoring;
the CLI refuses another indexed run after one completes. `test_blind` is
label-free, supports predictions, and cannot report WER or CER. The current
pinned validation revision contains `<not-available>` transcript sentinels;
the pipeline rejects them explicitly rather than inventing scores. Select a
known labeled row for smoke validation, or use another approved complete
revision for full scoring.

## Run

A tiny validation smoke run is useful before spending the download budget. Use
one known labeled row or a bounded prefix:

```bash
uv run urgent-asr run --config benchmark.toml --sample-id fileid_1012
```

For the full validation set, omit `--limit`. The `prepare`, `infer`, and
`evaluate` subcommands support stopping and inspecting between stages:

```bash
uv run urgent-asr prepare --config benchmark.toml
uv run urgent-asr infer --config benchmark.toml --run-dir runs/<run-id>
uv run urgent-asr evaluate --config benchmark.toml --run-dir runs/<run-id>
```

The prepare stage materializes paired clean/noisy mono 16 kHz FLOAT WAV files in
a content-addressed cache and writes a NeMo JSONL manifest. It does not peak
normalize; samples above the configured safe peak are clipped. Network
acquisition retries three times inside the ten-minute deadline.

The inference stage resolves the approved model revision into a local `.nemo`
checkpoint, verifies its checksum, fixes `[70, 1]` attention context (160 ms),
uses fp32 and one batch at a time, halves batches on CUDA OOM, and records
predictions, word timestamps, failures, and runtime. Per-utterance watchdogs are
five minutes; row failures abort evaluation above the configured 1% ceiling.

## Outputs

Each immutable run contains resolved configuration and identities, environment
details, manifest, predictions, per-utterance scores, metrics, CSV, and Markdown.
`runs/index.jsonl` is append-only and only receives completed runs. Reports
include corpus and utterance WER/CER, word operations and alignments, paired
clean/noisy deltas, available metadata strata, and seeded paired bootstrap 95%
confidence intervals.

The cache and runs share a 10 GiB limit and are ignored by Git. Reruns use local
Hugging Face and audio caches; `HF_HUB_OFFLINE=1` makes missing artifacts fail
without network access.

## Tests

```bash
uv run pytest
```

Tests use generated tiny audio fixtures and no dataset/model downloads.
