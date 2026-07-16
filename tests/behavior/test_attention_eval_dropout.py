import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tests._helpers.behavior_harness import SRC_ROOT, isolated_modules


def load_model_modules():
    old_path = list(sys.path)
    sys.path.insert(0, str(SRC_ROOT))
    try:
        with isolated_modules("models"):
            return {
                name: importlib.import_module(f"models.{name}")
                for name in ("base", "llama", "mup", "mup_llama")
            }
    finally:
        sys.path[:] = old_path


class AttentionEvalDropoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = load_model_modules()

    def test_sdpa_dropout_follows_module_training_mode_for_all_models(self):
        config = SimpleNamespace(
            n_embd=8,
            n_head=2,
            bias=False,
            dropout=0.625,
            sequence_length=4,
            opt="adamw",
        )
        x = torch.randn(2, 4, 8)
        freqs_cis = self.modules["llama"].precompute_freqs_cis(4, 4)
        cases = (
            (
                "base",
                self.modules["base"].CausalSelfAttention(config),
                lambda attention: attention(x),
            ),
            (
                "llama",
                self.modules["llama"].LlamaAttention(config),
                lambda attention: attention(x, freqs_cis),
            ),
            (
                "mup",
                self.modules["mup"].CausalSelfAttention(config),
                lambda attention: attention(x),
            ),
            (
                "mup_llama",
                self.modules["mup_llama"].LlamaAttention(config),
                lambda attention: attention(x, freqs_cis),
            ),
        )

        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        for name, attention, forward in cases:
            with self.subTest(model=name):
                dropout_probabilities = []

                def capture_sdpa(*args, **kwargs):
                    dropout_probabilities.append(kwargs["dropout_p"])
                    return original_sdpa(*args, **kwargs)

                with mock.patch.object(
                    torch.nn.functional,
                    "scaled_dot_product_attention",
                    side_effect=capture_sdpa,
                ):
                    attention.train()
                    forward(attention)
                    attention.eval()
                    first_eval = forward(attention)
                    second_eval = forward(attention)

                self.assertEqual(dropout_probabilities, [0.625, 0.0, 0.0])
                torch.testing.assert_close(
                    first_eval,
                    second_eval,
                    rtol=0.0,
                    atol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
