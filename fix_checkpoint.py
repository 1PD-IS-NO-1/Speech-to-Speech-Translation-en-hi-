
"""
fix_checkpoint.py
-----------------
;-if you are getting missing Lm_head warnings during inference after training then this is the best solution. please saw the usage.
Re-saves checkpoints made by qwen_s2st.py so that `lm_head.weight` is stored
as its own tensor. After this, a plain AutoModelForCausalLM.from_pretrained()
loads the correct trained output layer (no 'lm_head.weight MISSING', no
random head), in any code: vLLM, notebooks, other scripts.

The head's values are identical to the embedding table (that is how it was
trained), so nothing about the model's behaviour changes.

Usage:
  python fix_checkpoint.py ./runs/qwen05b_s2st ./runs/qwen05b_s2st/stage_mt

For every DIR given, a fixed copy is written to DIR_fixed (originals are
never modified). tokenizer files and vocab_spec.json are copied along.
Then, e.g.:
  mv ./runs/qwen05b_s2st ./runs/qwen05b_s2st_old
  mv ./runs/qwen05b_s2st_fixed ./runs/qwen05b_s2st
"""
import argparse
import glob
import os
import shutil

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def stored_keys(path):
    from safetensors import safe_open
    keys = {}
    for fn in glob.glob(os.path.join(path, "*.safetensors")):
        with safe_open(fn, "pt") as f:
            for k in f.keys():
                if k.endswith("embed_tokens.weight") or k == "lm_head.weight":
                    keys[k] = tuple(f.get_slice(k).get_shape())
    return keys


def load(path, dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--suffix", default="_fixed")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="fp32 keeps the weights exactly as trained (default)")
    args = ap.parse_args()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    for d in args.dirs:
        src = d.rstrip("/")
        dst = src + args.suffix
        print(f"\n=== {src} -> {dst}")
        have = stored_keys(src)
        print(f"stored tensors before: {have}")
        if "lm_head.weight" in have:
            print("lm_head.weight is already stored; nothing to do.")
            continue
        if os.path.exists(dst):
            print(f"{dst} already exists; skipping (delete it to redo).")
            continue

        lm = load(src, dtype)
        emb_w = lm.get_input_embeddings().weight
        # the head was trained tied to the embedding table: give it its own copy
        lm.lm_head.weight = nn.Parameter(emb_w.detach().clone())
        lm.config.tie_word_embeddings = False
        lm.save_pretrained(dst)
        AutoTokenizer.from_pretrained(src).save_pretrained(dst)
        shutil.copy(os.path.join(src, "vocab_spec.json"), dst)
        del lm

        # ---- verify with a plain load (no workaround) ----
        after = stored_keys(dst)
        print(f"stored tensors after:  {after}")
        assert "lm_head.weight" in after, "lm_head.weight still not stored!"
        m = load(dst, dtype)
        same = torch.equal(m.lm_head.weight, m.get_input_embeddings().weight)
        print(f"plain reload OK; lm_head == embedding values: {same}")
        assert same, "head/embedding values differ after reload"
        del m
    print("\nDone. Use the *_fixed folders (or rename them over the originals).")


if __name__ == "__main__":
    main()