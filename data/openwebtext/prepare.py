# Code partially from Andrej karpathy's prepare.py from nanoGPT

import os
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import Dataset
import tiktoken
import numpy as np
from tqdm import tqdm

from tools.config import DataConfig

cfg = DataConfig()
num_proc = cfg.num_proc
num_proc_load_dataset = cfg.num_proc_load_dataset
enc = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)
    split_dataset = dataset["train"].train_test_split(test_size=0.00005, seed=908, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")
    
    def preprocess(sample):
        ids = enc.encode_ordinary(sample["text"])
        ids.append(enc.eot_token)
        out = {"ids": ids, "len": len(ids)}
        return out

    tokenized_dataset = split_dataset.map(
        preprocess,
        remove_columns=["text"],
        desc="tokenizing the dataset",
        num_proc=num_proc
    )
    print("Finished tokenizing.")
    for split, dset in tokenized_dataset.items():
        split_len = np.sum(dset["len"], dtype=np.int64)
        filename = Path(os.path.dirname(__file__)) / f"{split}.bin"
        dtype = np.int64
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(split_len,))
        total_batches = cfg.total_batches

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()




    

