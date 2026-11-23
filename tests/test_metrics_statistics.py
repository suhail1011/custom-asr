from custom_asr_benchmark.config import BenchmarkConfig
from custom_asr_benchmark.evaluation import evaluate_predictions
from custom_asr_benchmark.metrics import align_words, score_pair
from custom_asr_benchmark.statistics import bootstrap, paired_comparison


def config(resamples=100):
    return BenchmarkConfig.from_mapping(
        {
            "dataset": {"revision": "a" * 40},
            "model": {"revision": "b" * 40},
            "evaluation": {"bootstrap_resamples": resamples},
        }
    )


def test_metrics_and_alignment_are_normalized():
    score = score_pair("Hello, WORLD!", "hello word")
    assert score["wer"] == 0.5
    assert score["operations"] == {"correct": 1, "substitute": 1, "delete": 0, "insert": 0}
    operations = [edit["op"] for edit in align_words("a b", "a x c")]
    assert operations.count("correct") == 1
    assert operations.count("insert") == 1
    assert operations.count("substitute") == 1


def test_bootstrap_is_seeded_and_paired():
    scores = [
        {**score_pair("a b", "a"), "id": "one"},
        {**score_pair("c", "x"), "id": "two"},
    ]
    first = bootstrap(scores, "wer", resamples=20, seed=5)
    assert first == bootstrap(scores, "wer", resamples=20, seed=5)
    assert first["lower"] <= first["estimate"] <= first["upper"]
    clean = [{**score_pair("a b", "a b"), "id": "one"}]
    noisy = [{**score_pair("a b", "a"), "id": "one"}]
    paired = paired_comparison(clean, noisy, resamples=20, seed=3)
    assert paired["paired_count"] == 1
    assert paired["wer"]["mean_delta"] == 0.5


def test_evaluation_builds_condition_and_paired_summaries():
    rows = [
        {"id": "one", "condition": "clean", "text": "a b", "text_prediction": "a b", "subset": "validation"},
        {"id": "one", "condition": "noisy", "text": "a b", "text_prediction": "a", "subset": "validation"},
    ]
    scores, metrics = evaluate_predictions(rows, config())
    assert len(scores) == 2
    assert metrics["conditions"]["clean"]["corpus"]["wer"] == 0
    assert metrics["conditions"]["noisy"]["corpus"]["wer"] == 0.5
    assert metrics["paired_clean_noisy"]["paired_count"] == 1
    assert metrics["coverage"]["complete"]
