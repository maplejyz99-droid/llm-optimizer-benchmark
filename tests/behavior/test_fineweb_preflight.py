import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "data" / "check_fineweb_30b.py"


def load_preflight_module():
    spec = importlib.util.spec_from_file_location("check_fineweb_30b", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FineWebPreflightBehaviorTest(unittest.TestCase):
    def _make_artifact(self, root: Path, *, tokenizer="gpt2") -> Path:
        module = load_preflight_module()
        artifact = root / "fineweb-30B"
        artifact.mkdir()
        with (artifact / "train.bin").open("wb") as handle:
            handle.truncate(module.TRAIN_TOKENS * module.ITEM_SIZE)
        with (artifact / "val.bin").open("wb") as handle:
            handle.truncate(module.VAL_TOKENS * module.ITEM_SIZE)
        (artifact / "meta.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "dtype": "uint16",
                    "tokenizer": tokenizer,
                    "train_tokens_target": module.TRAIN_TOKENS,
                    "train_tokens_written": module.TRAIN_TOKENS,
                    "val_tokens_target": module.VAL_TOKENS,
                    "val_tokens_written": module.VAL_TOKENS,
                }
            ),
            encoding="utf-8",
        )
        return artifact

    def test_validates_direct_and_parent_paths(self):
        module = load_preflight_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact = self._make_artifact(root)

            from_parent = module.validate_fineweb_30b(root)
            from_direct = module.validate_fineweb_30b(artifact)

        self.assertEqual(from_parent["profile"], "fineweb-30B-gpt2-uint16-v1")
        self.assertEqual(from_parent["fineweb_dir"], str(artifact.resolve()))
        self.assertEqual(from_direct["fineweb_dir"], str(artifact.resolve()))

    def test_rejects_wrong_identity_even_when_bins_exist(self):
        module = load_preflight_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._make_artifact(root, tokenizer="not-gpt2")

            with self.assertRaisesRegex(ValueError, "tokenizer"):
                module.validate_fineweb_30b(root)

    def test_default_directory_uses_environment_then_portable_fallback(self):
        module = load_preflight_module()
        with patch.dict("os.environ", {"LLMOPT_DATASETS_DIR": "/data/llmopt"}):
            self.assertEqual(module.default_datasets_dir(), Path("/data/llmopt"))
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                module.default_datasets_dir(), Path("./src/data/datasets/")
            )


if __name__ == "__main__":
    unittest.main()
