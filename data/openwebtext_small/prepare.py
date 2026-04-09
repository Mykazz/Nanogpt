# data/openwebtext_small/prepare.py
import os
import numpy as np
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")

# Scriptas, kuris padaro OpenWebText duomenų rinkinį mažesnį, kad būtų galima greičiau testuoti modelio treniravimą. Jis paims pirmuosius 2000 dokumentų iš mokymo rinkinio ir 100 dokumentų iš validacijos rinkinio, juos apdoros ir išsaugos
max_train_docs = 2000
max_val_docs = 100

if __name__ == "__main__":
    dataset = load_dataset("openwebtext")

    train_ds = dataset["train"].select(range(max_train_docs))
    val_ds = dataset["train"].select(range(max_train_docs, max_train_docs + max_val_docs))

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    train_tok = train_ds.map(process, remove_columns=["text"])
    val_tok = val_ds.map(process, remove_columns=["text"])

    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)

    for split, dset in [("train", train_tok), ("val", val_tok)]:
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(out_dir, f"{split}.bin")
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
        idx = 0
        for ids in dset["ids"]:
            ids = np.array(ids, dtype=np.uint16)
            arr[idx:idx+len(ids)] = ids
            idx += len(ids)
        arr.flush()
        print(f"{split}.bin written with {arr_len:,} tokens")