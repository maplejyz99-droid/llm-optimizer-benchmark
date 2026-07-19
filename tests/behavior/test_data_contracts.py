import importlib
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._helpers.behavior_harness import isolated_modules, patched_modules


def fake_module(name, **attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def load_data_utils(slimpajama_module, fineweb_module=None):
    torch_distributed = fake_module(
        "torch.distributed",
        is_initialized=lambda: False,
    )
    torch_module = fake_module("torch", distributed=torch_distributed)
    replacements = {
        "torch": torch_module,
        "torch.distributed": torch_distributed,
        "src.data.arxiv": fake_module(
            "src.data.arxiv",
            get_arxiv_2000=lambda _: None,
            get_arxiv_full=lambda _: None,
        ),
        "src.data.benchmarks": fake_module(
            "src.data.benchmarks",
            SUPPORTED_TASK_MAP={},
        ),
        "src.data.c4": fake_module("src.data.c4", get_c4_data=lambda _: None),
        "src.data.fineweb": fineweb_module
        or fake_module(
            "src.data.fineweb",
            get_fineweb_data=lambda *_args, **_kwargs: None,
        ),
        "src.data.fineweb_edu": fake_module(
            "src.data.fineweb_edu", get_fineweb_edu_data=lambda _: None
        ),
        "src.data.openwebtext2": fake_module(
            "src.data.openwebtext2", get_openwebtext2_data=lambda _: None
        ),
        "src.data.redpajama": fake_module(
            "src.data.redpajama",
            get_redpajama_data=lambda _: None,
            get_redpajamav2_data=lambda _: None,
        ),
        "src.data.shakespeare": fake_module(
            "src.data.shakespeare", get_shakespeare_data=lambda _: None
        ),
        "src.data.slimpajama": slimpajama_module,
        "src.data.wikitext": fake_module(
            "src.data.wikitext", get_wikitext_data=lambda _: None
        ),
    }
    with isolated_modules("src.data"), patched_modules(replacements):
        return importlib.import_module("src.data.utils")


def load_fineweb_module(load_dataset):
    tokenizer = types.SimpleNamespace(eot_token=50256, encode_ordinary=lambda text: [])
    replacements = {
        "datasets": fake_module("datasets", load_dataset=load_dataset),
        "tiktoken": fake_module("tiktoken", get_encoding=lambda _name: tokenizer),
        "tqdm": fake_module("tqdm", tqdm=lambda iterable, **_kwargs: iterable),
    }
    with isolated_modules("src.data"), patched_modules(replacements):
        return importlib.import_module("src.data.fineweb")


class FakeTokenizedDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        return [row[key] for row in self.rows]

    def shard(self, num_shards, index, contiguous):
        self.last_shard = (num_shards, index, contiguous)
        return self

    def with_format(self, _format):
        return self


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def train_test_split(self, **kwargs):
        self.split_kwargs = kwargs
        return FakeDatasetDict(
            train=FakeDataset([self.rows[0]]),
            test=FakeDataset([self.rows[-1]]),
        )

    def map(self, process, **_kwargs):
        return FakeTokenizedDataset([process(row) for row in self.rows])


class FakeDatasetDict(dict):
    def map(self, process, **kwargs):
        return FakeDatasetDict(
            **{split: dataset.map(process, **kwargs) for split, dataset in self.items()}
        )


class DataContractBehaviorTest(unittest.TestCase):
    def test_fineweb_accepts_direct_dataset_directory(self):
        load_calls = []
        fineweb = load_fineweb_module(
            lambda *args, **kwargs: load_calls.append((args, kwargs))
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "fineweb-30B"
            dataset_dir.mkdir()
            (dataset_dir / "train.bin").write_bytes(b"\x00\x00")
            (dataset_dir / "val.bin").write_bytes(b"\x01\x00")

            result = fineweb.get_fineweb_data(str(dataset_dir))

        self.assertEqual(
            result,
            {
                "train": str(Path(os.path.abspath(dataset_dir)) / "train.bin"),
                "val": str(Path(os.path.abspath(dataset_dir)) / "val.bin"),
            },
        )
        self.assertEqual(load_calls, [])

    def test_fineweb_resolves_30b_and_100bt_children(self):
        fineweb = load_fineweb_module(
            lambda *_args, **_kwargs: self.fail("existing data must not download")
        )

        for child in ("fineweb-30B", "fineweb-100BT"):
            with self.subTest(child=child), tempfile.TemporaryDirectory() as tmp_dir:
                dataset_dir = Path(tmp_dir) / child
                dataset_dir.mkdir()
                (dataset_dir / "train.bin").write_bytes(b"\x00\x00")
                (dataset_dir / "val.bin").write_bytes(b"\x01\x00")

                result = fineweb.get_fineweb_data(tmp_dir)

                expected_dir = Path(os.path.abspath(dataset_dir))
                self.assertEqual(result["train"], str(expected_dir / "train.bin"))
                self.assertEqual(result["val"], str(expected_dir / "val.bin"))

    def test_fineweb_no_download_fails_before_creating_or_fetching(self):
        load_calls = []
        fineweb = load_fineweb_module(
            lambda *args, **kwargs: load_calls.append((args, kwargs))
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_dir = Path(tmp_dir) / "missing"
            with patch.dict("os.environ", {"LLMOPT_FINEWEB_NO_DOWNLOAD": "1"}):
                with self.assertRaisesRegex(FileNotFoundError, "fineweb-30B"):
                    fineweb.get_fineweb_data(
                        str(datasets_dir), allow_download=True
                    )
            self.assertFalse(datasets_dir.exists())

        self.assertEqual(load_calls, [])

    def test_fineweb_missing_data_fails_closed_by_default(self):
        fineweb = load_fineweb_module(
            lambda *_args, **_kwargs: self.fail(
                "training must not download data without explicit opt-in"
            )
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_dir = Path(tmp_dir) / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "allow_dataset_download"):
                fineweb.get_fineweb_data(str(datasets_dir))
            self.assertFalse(datasets_dir.exists())

    def test_fineweb_explicit_download_opt_in_preserves_legacy_builder(self):
        load_calls = []
        source_dataset = FakeDataset([{"text": "train"}, {"text": "val"}])

        def load_dataset(*args, **kwargs):
            load_calls.append((args, kwargs))
            return source_dataset

        fineweb = load_fineweb_module(load_dataset)
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = fineweb.get_fineweb_data(
                tmp_dir, num_proc=1, allow_download=True
            )

            expected_dir = Path(tmp_dir) / "fineweb-100BT"
            self.assertEqual(
                result,
                {
                    "train": str(expected_dir / "train.bin"),
                    "val": str(expected_dir / "val.bin"),
                },
            )
            self.assertTrue((expected_dir / "train.bin").exists())
            self.assertTrue((expected_dir / "val.bin").exists())

        self.assertEqual(load_calls[0][0], ("HuggingFaceFW/fineweb",))
        self.assertEqual(load_calls[0][1]["name"], "sample-100BT")

    def test_fineweb_partial_pair_is_not_treated_as_complete(self):
        load_calls = []
        fineweb = load_fineweb_module(
            lambda *args, **kwargs: load_calls.append((args, kwargs))
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "fineweb-100BT"
            dataset_dir.mkdir()
            (dataset_dir / "train.bin").write_bytes(b"incomplete")

            with patch.dict("os.environ", {"LLMOPT_FINEWEB_NO_DOWNLOAD": "1"}):
                with self.assertRaisesRegex(FileNotFoundError, "train.bin/val.bin"):
                    fineweb.get_fineweb_data(tmp_dir)

        self.assertEqual(load_calls, [])

    def test_fineweb_rejects_empty_or_non_uint16_aligned_bins(self):
        fineweb = load_fineweb_module(
            lambda *_args, **_kwargs: self.fail("invalid data must not download")
        )

        for invalid_bytes in (b"", b"x"):
            with self.subTest(size=len(invalid_bytes)), tempfile.TemporaryDirectory() as tmp_dir:
                dataset_dir = Path(tmp_dir) / "fineweb-30B"
                dataset_dir.mkdir()
                (dataset_dir / "train.bin").write_bytes(invalid_bytes)
                (dataset_dir / "val.bin").write_bytes(b"\x00\x00")
                with patch.dict("os.environ", {"LLMOPT_FINEWEB_NO_DOWNLOAD": "1"}):
                    with self.assertRaises(FileNotFoundError):
                        fineweb.get_fineweb_data(tmp_dir)

    def test_fineweb_prefers_30b_when_both_named_children_exist(self):
        fineweb = load_fineweb_module(
            lambda *_args, **_kwargs: self.fail("existing data must not download")
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            for child in ("fineweb-30B", "fineweb-100BT"):
                dataset_dir = Path(tmp_dir) / child
                dataset_dir.mkdir()
                (dataset_dir / "train.bin").write_bytes(b"\x00\x00")
                (dataset_dir / "val.bin").write_bytes(b"\x01\x00")

            result = fineweb.get_fineweb_data(tmp_dir)

        self.assertIn("fineweb-30B", result["train"])
        self.assertIn("fineweb-30B", result["val"])

    def test_fineweb_dispatch_passes_download_policy(self):
        calls = []
        fineweb_module = fake_module(
            "src.data.fineweb",
            get_fineweb_data=lambda path, allow_download=False: calls.append(
                (path, allow_download)
            )
            or {"train": "train.bin", "val": "val.bin"},
        )
        slimpajama_module = fake_module(
            "src.data.slimpajama",
            get_slimpajama_data=lambda _path: None,
            get_slimpajama_chunk1=lambda _path: None,
        )
        data_utils = load_data_utils(slimpajama_module, fineweb_module)
        args = types.SimpleNamespace(
            dataset="fineweb",
            datasets_dir="/datasets",
            allow_dataset_download=True,
        )

        result = data_utils.get_dataset(args)

        self.assertEqual(result, {"train": "train.bin", "val": "val.bin"})
        self.assertEqual(calls, [("/datasets", True)])

    def test_slimpajama_chunk1_dispatches_to_its_builder(self):
        calls = []
        slimpajama_module = fake_module(
            "src.data.slimpajama",
            get_slimpajama_data=lambda path: calls.append(("full", path)),
            get_slimpajama_chunk1=lambda path: calls.append(("chunk1", path))
            or {"train": "train.bin", "val": "val.bin"},
        )
        data_utils = load_data_utils(slimpajama_module)
        args = types.SimpleNamespace(
            dataset="slimpajama_chunk1", datasets_dir="/datasets"
        )

        result = data_utils.get_dataset(args)

        self.assertEqual(result, {"train": "train.bin", "val": "val.bin"})
        self.assertEqual(calls, [("chunk1", "/datasets")])

    def test_slimpajama_chunk1_handles_dataset_split_and_uses_isolated_paths(self):
        import numpy as np

        load_calls = []
        source_dataset = FakeDataset([{"text": "train"}, {"text": "val"}])

        def load_dataset(name, **kwargs):
            load_calls.append((name, kwargs))
            return source_dataset

        tokenizer = types.SimpleNamespace(
            eot_token=50256,
            encode_ordinary=lambda text: [len(text)],
        )
        replacements = {
            "datasets": fake_module("datasets", load_dataset=load_dataset),
            "tiktoken": fake_module(
                "tiktoken", get_encoding=lambda name: tokenizer
            ),
            "tqdm": fake_module("tqdm", tqdm=lambda iterable, **_kwargs: iterable),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            regular_dataset_path = Path(tmp_dir) / "slimpajama6B" / "train.bin"
            regular_dataset_path.parent.mkdir(parents=True)
            regular_dataset_path.write_bytes(b"keep-me")

            with isolated_modules("src.data"), patched_modules(replacements):
                slimpajama = importlib.import_module("src.data.slimpajama")
                result = slimpajama.get_slimpajama_chunk1(tmp_dir, num_proc=1)

            expected_dir = Path(tmp_dir) / "slimpajama627B" / "chunk1"
            self.assertEqual(
                result,
                {
                    "train": str(expected_dir / "train.bin"),
                    "val": str(expected_dir / "val.bin"),
                },
            )
            self.assertTrue((expected_dir / "train.bin").exists())
            self.assertTrue((expected_dir / "val.bin").exists())
            self.assertEqual(regular_dataset_path.read_bytes(), b"keep-me")

        self.assertEqual(
            load_calls,
            [
                (
                    "cerebras/SlimPajama-627B",
                    {"data_dir": "train/chunk1", "split": "train"},
                )
            ],
        )
        self.assertEqual(
            source_dataset.split_kwargs,
            {"test_size": 0.0005, "seed": 2357, "shuffle": True},
        )


if __name__ == "__main__":
    unittest.main()
