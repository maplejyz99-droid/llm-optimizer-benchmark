import importlib
import sys
import unittest

import torch

from tests._helpers.behavior_harness import SRC_ROOT, isolated_modules


def load_utils_module():
    old_path = list(sys.path)
    sys.path.insert(0, str(SRC_ROOT))
    try:
        with isolated_modules("optim"):
            return importlib.import_module("optim.utils")
    finally:
        sys.path[:] = old_path


class FixedReader:
    def __init__(self, targets):
        self.targets = targets

    def sample_batch(self):
        return torch.zeros_like(self.targets), self.targets.clone()


class FixedModel(torch.nn.Module):
    def __init__(self, predictions, vocab_size=3):
        super().__init__()
        self.predictions = predictions
        self.vocab_size = vocab_size

    def forward(self, x, **_kwargs):
        logits = torch.zeros(*x.shape, self.vocab_size)
        logits.scatter_(-1, self.predictions.unsqueeze(-1), 1.0)
        return {
            "loss": torch.tensor(0.0),
            "logits": logits,
            "aux_losses": {},
        }


class EvalMaskBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.utils = load_utils_module()

    def test_token_accuracy_ignores_minus_one_targets(self):
        targets = torch.tensor([[1, -1]])
        model = FixedModel(torch.tensor([[1, 2]])).eval()

        accuracy, _loss, _perplexity, _aux, _router = self.utils.eval(
            model,
            FixedReader(targets),
            max_num_batches=1,
        )

        self.assertEqual(accuracy, 1.0)

    def test_token_accuracy_rejects_validation_without_effective_targets(self):
        targets = torch.tensor([[-1, -1]])
        model = FixedModel(torch.tensor([[0, 0]])).eval()

        with self.assertRaisesRegex(ValueError, "no effective targets"):
            self.utils.eval(
                model,
                FixedReader(targets),
                max_num_batches=1,
            )


if __name__ == "__main__":
    unittest.main()
