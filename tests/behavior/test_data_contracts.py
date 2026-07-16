import importlib
import tempfile
import types
import unittest
from pathlib import Path

from tests._helpers.behavior_harness import isolated_modules, patched_modules


def fake_module(name, **attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def load_data_utils(slimpajama_module):
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
        "src.data.fineweb": fake_module(
            "src.data.fineweb", get_fineweb_data=lambda _: None
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
