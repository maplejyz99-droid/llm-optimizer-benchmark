import importlib
import io
import sys
import unittest
from contextlib import nullcontext, redirect_stdout
from types import SimpleNamespace

from tests._helpers.behavior_harness import SRC_ROOT, isolated_modules


def load_weight_averaging_module():
    old_path = list(sys.path)
    sys.path.insert(0, str(SRC_ROOT))
    try:
        with isolated_modules("optim"):
            return importlib.import_module("optim.weight_averaging")
    finally:
        sys.path[:] = old_path


class FakeModel:
    def eval(self):
        return self


class FakeReader:
    def __init__(self, batch_count=11):
        self.batch_count = batch_count
        self.steps = []

    def num_batches(self):
        return self.batch_count

    def set_step(self, step):
        self.steps.append(step)


class FakeBackend:
    def is_master_process(self):
        return True


class FakeWeightAverager:
    num_saved = 1

    def get_latest_like(self, model):
        return model

    def sweep_horizon_like(self, model, max_num):
        del max_num
        yield 4, model


class FakeExponentialWeightAverager:
    def get_latest_like(self, model):
        return model


def make_config(**overrides):
    values = {
        "iterations": 10,
        "eval_batches": 2,
        "final_eval_batches": 3,
        "final_eval_tokens": None,
        "device": "cpu",
        "moe": False,
        "wandb": False,
        "wa_sweep_horizon": False,
        "max_num_wa_sweeps": None,
        "plot_router_logits": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FinalEvalBatchCapTest(unittest.TestCase):
    def setUp(self):
        self.averaging = load_weight_averaging_module()
        self.original_eval = self.averaging.eval
        self.eval_limits = []

        def fake_eval(*_args, **kwargs):
            self.eval_limits.append(
                (kwargs["max_num_batches"], kwargs.get("max_num_tokens"))
            )
            result = (0.5, 1.25, 3.5, {}, None)
            if kwargs.get("return_counts"):
                result += ({"evaluated_batches": 1, "evaluated_tokens": 8},)
            return result

        self.averaging.eval = fake_eval

    def tearDown(self):
        self.averaging.eval = self.original_eval

    def run_wa(self, cfg, *, curr_iter=10, full_eval=False):
        with redirect_stdout(io.StringIO()):
            self.averaging.eval_wa(
                curr_iter,
                FakeModel(),
                FakeWeightAverager(),
                FakeReader(),
                nullcontext(),
                FakeBackend(),
                cfg,
                full_eval=full_eval,
            )

    def run_ewa(self, cfg, *, curr_iter=10, full_eval=False):
        with redirect_stdout(io.StringIO()):
            self.averaging.eval_ewa(
                curr_iter,
                FakeModel(),
                FakeExponentialWeightAverager(),
                FakeReader(),
                nullcontext(),
                FakeBackend(),
                cfg,
                full_eval=full_eval,
            )

    def test_final_cap_is_shared_by_wa_sweep_and_ewa(self):
        self.run_wa(make_config())
        self.run_wa(make_config(wa_sweep_horizon=True))
        self.run_ewa(make_config())

        self.assertEqual(self.eval_limits, [(3, None), (3, None), (3, None)])

    def test_full_eval_uses_cap_and_periodic_eval_keeps_regular_limit(self):
        self.run_wa(make_config(), curr_iter=4, full_eval=True)
        self.run_ewa(make_config(), curr_iter=4, full_eval=False)

        self.assertEqual(self.eval_limits, [(3, None), (2, None)])

    def test_unset_or_large_cap_never_exceeds_available_batches(self):
        self.run_wa(make_config(final_eval_batches=None))
        self.run_ewa(make_config(final_eval_batches=50))

        self.assertEqual(self.eval_limits, [(11, None), (11, None)])

    def test_token_cap_is_shared_by_wa_and_ewa(self):
        cfg = make_config(final_eval_batches=None, final_eval_tokens=4096)

        self.run_wa(cfg)
        self.run_ewa(cfg)

        self.assertEqual(self.eval_limits, [(11, 4096), (11, 4096)])


if __name__ == "__main__":
    unittest.main()
