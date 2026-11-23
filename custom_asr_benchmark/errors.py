class BenchmarkError(RuntimeError):
    """An actionable benchmark failure."""


class ConfigError(BenchmarkError):
    """The resolved configuration is invalid."""


class DataError(BenchmarkError):
    """The dataset or one of its rows is invalid."""


class CapacityError(BenchmarkError):
    """A configured safety limit was exceeded."""
