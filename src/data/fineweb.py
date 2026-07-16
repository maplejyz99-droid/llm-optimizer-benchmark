import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

tknzr = tiktoken.get_encoding("gpt2")


def _has_fineweb_bins(path):
    return os.path.exists(os.path.join(path, "train.bin")) and os.path.exists(
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


def get_fineweb_data(datasets_dir, num_proc=40):
    """Return existing FineWeb memmap files, or build legacy fineweb-100BT when allowed.

    Supported existing layouts:
    - --datasets_dir /path/to/fineweb-30B
    - --datasets_dir /path/to/datasets containing fineweb-30B/
    - --datasets_dir /path/to/datasets containing fineweb-100BT/

    A800 scripts set LLMOPT_FINEWEB_NO_DOWNLOAD=1 so missing train.bin/val.bin
    fails fast instead of starting a HuggingFace download.
    """
    existing_path = _resolve_existing_fineweb_path(datasets_dir)
    if existing_path is not None:
        return {
            "train": os.path.join(existing_path, "train.bin"),
            "val": os.path.join(existing_path, "val.bin"),
        }

    if os.environ.get("LLMOPT_FINEWEB_NO_DOWNLOAD") == "1":
        expected = [
            os.path.join(datasets_dir, "train.bin"),
            os.path.join(datasets_dir, "fineweb-30B", "train.bin"),
            os.path.join(datasets_dir, "fineweb-100BT", "train.bin"),
        ]
        raise FileNotFoundError(
            "FineWeb train.bin/val.bin not found. Checked direct fineweb dir, "
            "fineweb-30B/, and fineweb-100BT/. A800 runs disable automatic "
            f"download. Example checked train paths: {expected}"
        )

    FWEB_DATA_PATH = os.path.join(datasets_dir, "fineweb-100BT")
    if not os.path.exists(os.path.join(FWEB_DATA_PATH, "train.bin")):
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
            ids = tknzr.encode_ordinary(example["text"])
            ids.append(tknzr.eot_token)
            out = {"ids": ids, "len": len(ids)}
            return out

        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(FWEB_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    return {
        "train": os.path.join(FWEB_DATA_PATH, "train.bin"),
        "val": os.path.join(FWEB_DATA_PATH, "val.bin"),
    }


if __name__ == "__main__":
    get_fineweb_data("./datasets/")
