import sys
import types
import unittest

from tests._helpers.behavior_harness import SRC_ROOT, isolated_modules


try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    old_path = list(sys.path)
    sys.path.insert(0, str(SRC_ROOT))
    try:
        with isolated_modules("models"):
            from models.llama import LlamaAttention, LlamaMLP
    finally:
        sys.path[:] = old_path


@unittest.skipUnless(torch is not None, "requires torch and project runtime dependencies")
class NewtonBufferAllocationBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.base_config = {
            "n_embd": 16,
            "multiple_of": 8,
            "bias": False,
            "n_head": 4,
            "dropout": 0.0,
            "sequence_length": 8,
        }

    def make_modules(self, optimizer):
        config = types.SimpleNamespace(**self.base_config, opt=optimizer)
        return LlamaMLP(config), LlamaAttention(config)

    def test_newton_buffers_exist_only_for_newton_muon(self):
        for optimizer in ("adamw", "muon", "sophiag", "soap"):
            modules = self.make_modules(optimizer)
            names = {
                name
                for module in modules
                for name, _buffer in module.named_buffers()
            }
            self.assertFalse(
                any(name.startswith("newton_muon_") for name in names),
                (optimizer, names),
            )

        mlp, attention = self.make_modules("newton-muon")
        self.assertEqual(
            {name for name, _buffer in mlp.named_buffers()},
            {
                "newton_muon_fc_accum",
                "newton_muon_fc_count",
                "newton_muon_proj_accum",
                "newton_muon_proj_count",
            },
        )
        self.assertEqual(
            {name for name, _buffer in attention.named_buffers()},
            {
                "newton_muon_qkv_accum",
                "newton_muon_qkv_count",
                "newton_muon_o_accum",
                "newton_muon_o_count",
            },
        )

    def test_non_newton_modules_reject_preconditioning(self):
        mlp, attention = self.make_modules("adamw")
        x = torch.randn(1, 2, self.base_config["n_embd"])
        freqs_cis = torch.zeros(
            2,
            self.base_config["n_embd"] // self.base_config["n_head"] // 2,
            2,
        )

        with self.assertRaisesRegex(RuntimeError, "without buffers"):
            mlp(x, precond_flag=True)
        with self.assertRaisesRegex(RuntimeError, "without buffers"):
            attention(x, freqs_cis, precond_flag=True)


if __name__ == "__main__":
    unittest.main()
