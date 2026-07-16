"""Shared pytest configuration for the tetris test suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that are slower than a typical unit test but still "
        "run in the default suite (e.g. the Phase C linear-probe gate).",
    )
