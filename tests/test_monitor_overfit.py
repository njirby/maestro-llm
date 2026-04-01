import importlib.util
import os
import sys


def _load_monitor_module():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "scripts", "monitor_overfit_and_relaunch.py")
    spec = importlib.util.spec_from_file_location("monitor_overfit_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_should_stop_for_overfitting_triggers_on_two_eval_rises_and_nonincreasing_train():
    module = _load_monitor_module()
    P = module.ScalarPoint

    eval_points = [P(step=100, value=2.50), P(step=200, value=2.51), P(step=300, value=2.52)]
    train_points = [P(step=90, value=2.40), P(step=190, value=2.35), P(step=290, value=2.30)]

    should_stop, _ = module.should_stop_for_overfitting(eval_points, train_points, min_eval_rise=0.005)
    assert should_stop is True


def test_should_stop_for_overfitting_ignores_small_eval_noise():
    module = _load_monitor_module()
    P = module.ScalarPoint

    eval_points = [P(step=100, value=2.50), P(step=200, value=2.503), P(step=300, value=2.507)]
    train_points = [P(step=90, value=2.40), P(step=190, value=2.35), P(step=290, value=2.30)]

    should_stop, _ = module.should_stop_for_overfitting(eval_points, train_points, min_eval_rise=0.005)
    assert should_stop is False


def test_should_stop_for_overfitting_rejects_when_train_also_rises():
    module = _load_monitor_module()
    P = module.ScalarPoint

    eval_points = [P(step=100, value=2.50), P(step=200, value=2.51), P(step=300, value=2.52)]
    train_points = [P(step=90, value=2.30), P(step=190, value=2.34), P(step=290, value=2.38)]

    should_stop, _ = module.should_stop_for_overfitting(eval_points, train_points, min_eval_rise=0.005)
    assert should_stop is False
