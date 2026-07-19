import importlib
import sys
import unittest
from types import SimpleNamespace

from tests._helpers.behavior_harness import SRC_ROOT, isolated_modules


def load_models_utils():
    old_path = list(sys.path)
    sys.path.insert(0, str(SRC_ROOT))
    try:
        with isolated_modules("models"):
            return importlib.import_module("models.utils")
    finally:
        sys.path[:] = old_path


class ModelLoaderContractTest(unittest.TestCase):
    def test_base_pretrained_loader_receives_from_dense(self):
        models_utils = load_models_utils()
        calls = []

        class FakeModel:
            def from_pretrained(self, path, *, from_dense):
                calls.append((path, from_dense))

        models_utils.GPTBase = lambda _args: FakeModel()
        args = SimpleNamespace(
            model="base",
            use_pretrained="checkpoint.pt",
            from_dense=False,
        )

        model = models_utils.get_model(args)

        self.assertIsInstance(model, FakeModel)
        self.assertEqual(calls, [("checkpoint.pt", False)])


if __name__ == "__main__":
    unittest.main()
