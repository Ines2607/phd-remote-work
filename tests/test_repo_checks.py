import importlib.util
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_hardcoded_values_in_library():
    assert _load("check_hardcoded").main() == 0


def test_no_user_data_tracked():
    assert _load("check_no_user_data").main() == 0
