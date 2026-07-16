import importlib
import sys
import unittest
from unittest import mock

import torch

from tests._helpers.behavior_harness import SRC_ROOT, isolated_modules


def load_softeq_module():
    old_path = list(sys.path)
    sys.path.insert(0, str(SRC_ROOT))
    try:
        with isolated_modules("optim"):
            return importlib.import_module("optim.experimental.softeq_muon")
    finally:
        sys.path[:] = old_path


class SoftEqMuonBehaviorTest(unittest.TestCase):
    def test_soft_row_equilibrate_matches_row_norm_formula(self):
        softeq_muon = load_softeq_module()
        update = torch.tensor([[3.0, 4.0], [0.0, 9.0]])

        actual = softeq_muon.soft_row_equilibrate(update, alpha=0.5, eps=1e-8)
        expected = update / update.norm(dim=-1, keepdim=True).pow(0.5)

        self.assertTrue(torch.allclose(actual, expected))

    def test_matrix_step_uses_softeq_before_cutoff_and_plain_update_after(self):
        softeq_muon = load_softeq_module()
        grad = torch.tensor([[3.0, 4.0], [0.0, 2.0]])

        def one_step(global_step):
            param = torch.nn.Parameter(torch.ones_like(grad))
            param.grad = grad.clone()
            opt = softeq_muon.SoftEqK2000Muon(
                [param],
                lr=0.1,
                momentum=0.5,
                weight_decay=0.0,
                adamw_lr=0.01,
            )
            opt.global_step = global_step
            with mock.patch.object(
                softeq_muon,
                "zeropower_via_newtonschulz12",
                side_effect=lambda update, steps=12: update,
            ):
                opt.step()
            return param.detach()

        raw_update = grad.lerp(grad * 0.5, 0.5)
        softeq_update = raw_update / raw_update.norm(dim=-1, keepdim=True).pow(0.5)

        before_cutoff = one_step(global_step=0)
        after_cutoff = one_step(global_step=softeq_muon.SOFTEQ_STEPS)

        self.assertTrue(torch.allclose(before_cutoff, torch.ones_like(grad) - 0.1 * softeq_update))
        self.assertTrue(torch.allclose(after_cutoff, torch.ones_like(grad) - 0.1 * raw_update))

    def test_global_step_round_trips_through_state_dict(self):
        softeq_muon = load_softeq_module()
        param = torch.nn.Parameter(torch.ones(2, 2))
        opt = softeq_muon.SoftEqK2000Muon([param], adamw_lr=0.01)
        opt.global_step = softeq_muon.SOFTEQ_STEPS - 1

        restored_param = torch.nn.Parameter(torch.ones(2, 2))
        restored = softeq_muon.SoftEqK2000Muon([restored_param], adamw_lr=0.01)
        restored.load_state_dict(opt.state_dict())

        self.assertEqual(restored.global_step, softeq_muon.SOFTEQ_STEPS - 1)

    def test_load_state_dict_does_not_mutate_input_state(self):
        softeq_muon = load_softeq_module()
        param = torch.nn.Parameter(torch.ones(2, 2))
        opt = softeq_muon.SoftEqK2000Muon([param], adamw_lr=0.01)
        opt.global_step = softeq_muon.SOFTEQ_STEPS - 1
        state = opt.state_dict()

        first = softeq_muon.SoftEqK2000Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            adamw_lr=0.01,
        )
        second = softeq_muon.SoftEqK2000Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            adamw_lr=0.01,
        )
        first.load_state_dict(state)
        second.load_state_dict(state)

        self.assertIn("softeq_global_step", state)
        self.assertEqual(first.global_step, softeq_muon.SOFTEQ_STEPS - 1)
        self.assertEqual(second.global_step, softeq_muon.SOFTEQ_STEPS - 1)


if __name__ == "__main__":
    unittest.main()
