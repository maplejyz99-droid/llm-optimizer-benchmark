import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

tknzr = tiktoken.get_encoding("gpt2")


def _is_valid_token_bin(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0 and os.path.getsize(path) % 2 == 0


def _has_fineweb_bins(path):
    return _is_valid_token_bin(os.path.join(path, "train.bin")) and _is_valid_token_bin(
        os.path.join(path, "val.bin")
    )


def _resolve_existing_fineweb_path(datasets_dir):
    candidates = [
        datasets_dir,
        os.path.join(datasets_dir, "fineweb-30B"),
        os.path.join(datasets_dir, "fineweb-100BT"),
    ]
    seen = set()
    for path in candidates:
        path = os.path.abspath(os.path.expanduser(path))
        if path in seen:
            continue
        seen.add(path)
        if _has_fineweb_bins(path):
            return path
    return None


def get_fineweb_data(datasets_dir, num_proc=40, *, allow_download=False):
    """Return existing FineWeb binaries or explicitly build the legacy sample.

    Dataset discovery is read-only. Missing data fails closed unless the caller
    explicitly opts in with ``allow_download=True``. The legacy
    ``LLMOPT_FINEWEB_NO_DOWNLOAD=1`` guard remains a final safety override for
    existing deployment scripts.
    """
    existing_path = _resolve_existing_fineweb_path(datasets_dir)
    if existing_path is not None:
        return {
            "train": os.path.join(existing_path, "train.bin"),
            "val": os.path.join(existing_path, "val.bin"),
        }

    downloads_disabled = os.environ.get("LLMOPT_FINEWEB_NO_DOWNLOAD") == "1"
    if not allow_download or downloads_disabled:
        expected = [
            os.path.join(datasets_dir, "train.bin"),
            os.path.join(datasets_dir, "fineweb-30B", "train.bin"),
            os.path.join(datasets_dir, "fineweb-100BT", "train.bin"),
        ]
        raise FileNotFoundError(
            "FineWeb train.bin/val.bin not found. Checked the direct directory, "
            "fineweb-30B/, and fineweb-100BT/. Training does not download data "
            "by default; set --datasets_dir or LLMOPT_DATASETS_DIR to existing "
            "data. To intentionally build the legacy sample-100BT, pass "
            "--allow_dataset_download. LLMOPT_FINEWEB_NO_DOWNLOAD=1 always "
            f"disables downloads. Example train paths: {expected}"
        )

    FWEB_DATA_PATH = os.path.join(datasets_dir, "fineweb-100BT")
    if not _has_fineweb_bins(FWEB_DATA_PATH):
        os.makedirs(FWEB_DATA_PATH, exist_ok=True)

        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-100BT",
            split="train",
            streaming=False,
            verification_mode="no_checks",
        )

        split_dataset = dataset.train_test_split(
            test_size=0.0001, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(FWEB_DATA_PATH, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    return {
        "train": os.path.join(FWEB_DATA_PATH, "train.bin"),
        "val": os.path.join(FWEB_DATA_PATH, "val.bin"),
    }


if __name__ == "__main__":
    get_fineweb_data("./datasets/", allow_download=True)
