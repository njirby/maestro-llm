import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "reaper: mark test as requiring a live REAPER session with reapy server"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--reaper", action="store_true", default=False,
        help="Run tests requiring a live REAPER session (reapy server must be active)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--reaper", default=False):
        skip = pytest.mark.skip(reason="requires live REAPER session (pass --reaper to run)")
        for item in items:
            if item.get_closest_marker("reaper"):
                item.add_marker(skip)
