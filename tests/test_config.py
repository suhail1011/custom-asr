import pytest

from custom_asr_benchmark.config import BenchmarkConfig
from custom_asr_benchmark.errors import ConfigError


def test_config_requires_pinned_revisions_and_resolves_split(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[dataset]
revision = "a" * 40
subset = "test_blind"
[model]
revision = "b" * 40
"""
        .replace('"a" * 40', '"' + "a" * 40 + '"')
        .replace('"b" * 40', '"' + "b" * 40 + '"')
    )
    config = BenchmarkConfig.load(path)
    assert config.dataset.resolved_split == "test"
    with pytest.raises(ConfigError, match="pinned commit"):
        BenchmarkConfig.from_mapping({"dataset": {"revision": "main"}, "model": {"revision": "a" * 40}})


def test_config_rejects_unapproved_limits():
    with pytest.raises(ConfigError, match="four hours"):
        BenchmarkConfig.from_mapping(
            {
                "dataset": {"revision": "a" * 40},
                "model": {"revision": "b" * 40},
                "limits": {"cpu_hours": 5},
            }
        )
    with pytest.raises(ConfigError, match="cannot exceed three"):
        BenchmarkConfig.from_mapping(
            {
                "dataset": {"revision": "a" * 40},
                "model": {"revision": "b" * 40},
                "limits": {"network_retries": 4},
            }
        )


def test_config_rejects_unknown_keys_and_wrong_model():
    with pytest.raises(ConfigError, match="unknown top-level"):
        BenchmarkConfig.from_mapping(
            {
                "dataset": {"revision": "a" * 40},
                "model": {"revision": "b" * 40},
                "typo": True,
            }
        )
    with pytest.raises(ConfigError, match="approved Nemotron"):
        BenchmarkConfig.from_mapping(
            {
                "dataset": {"revision": "a" * 40},
                "model": {"identifier": "someone/other-model", "revision": "b" * 40},
            }
        )
