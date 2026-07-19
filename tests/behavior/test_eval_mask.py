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


class SequenceReader:
    def __init__(self, targets):
        self.targets = iter(targets)

    def sample_batch(self):
        targets = next(self.targets)
        return torch.zeros_like(targets), targets.clone()


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


class RecordingLossModel(FixedModel):
    def __init__(self, predictions, vocab_size=3):
        super().__init__(predictions, vocab_size=vocab_size)
        self.seen_targets = []

    def forward(self, x, targets, **_kwargs):
        self.seen_targets.append(targets.clone())
        outputs = super().forward(x)
        valid_targets = targets != -1
        outputs["loss"] = targets[valid_targets].float().mean()
        return outputs


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

    def test_token_budget_masks_only_excess_targets_and_reports_actual_work(self):
        targets = torch.tensor([[0, 1, 2, 0]])
        model = RecordingLossModel(targets).eval()

        result = self.utils.eval(
            model,
            FixedReader(targets),
            max_num_batches=10,
            max_num_tokens=5,
            return_counts=True,
        )

        accuracy, loss, _perplexity, _aux, _router, counts = result
        self.assertEqual(accuracy, 1.0)
        self.assertAlmostEqual(loss, 0.6)
        self.assertEqual(len(model.seen_targets), 2)
        self.assertTrue(torch.equal(model.seen_targets[0], targets))
        self.assertTrue(
            torch.equal(
                model.seen_targets[1],
                torch.tensor([[0, -1, -1, -1]]),
            )
        )
        self.assertEqual(
            counts,
            {"evaluated_batches": 2, "evaluated_tokens": 5},
        )

    def test_empty_target_batch_is_skipped_without_hiding_later_valid_data(self):
        batches = [
            torch.tensor([[0, -1]]),
            torch.tensor([[-1, -1]]),
            torch.tensor([[0, 1]]),
        ]
        model = RecordingLossModel(torch.tensor([[0, 1]])).eval()

        result = self.utils.eval(
            model,
            SequenceReader(batches),
            max_num_batches=3,
            return_counts=True,
        )

        accuracy, loss, _perplexity, _aux, _router, counts = result
        self.assertEqual(accuracy, 1.0)
        self.assertAlmostEqual(loss, 1.0 / 3.0)
        self.assertEqual(len(model.seen_targets), 2)
        self.assertEqual(
            counts,
            {"evaluated_batches": 2, "evaluated_tokens": 3},
        )


if __name__ == "__main__":
    unittest.main()
