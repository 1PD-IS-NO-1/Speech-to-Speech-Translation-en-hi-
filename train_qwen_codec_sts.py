# """
# train_qwen_codec_s2s.py
# ------------------------
# Fine-tunes Qwen2.5-0.5B, repurposed as a discrete-token seq2seq model, to
# translate speech between English and Hindi via MioCodec content tokens.

# Two training modes:

#   CODEC-ONLY (original): en_codec -> hi_codec directly. Works with either
#     --vocab_mode shrink or extend. No text needed in the jsonl.

#   TEXT-BRIDGE (new, requires --vocab_mode extend): uses the real Qwen
#     tokenizer + Qwen's pretrained text vocab (never touched in shrink
#     mode, which is why this requires extend) to train:
#         src_text + src_codec  ->  tgt_text + tgt_codec
#     i.e. the model generates the target language's TEXT first, then its
#     codec tokens, conditioned on that text. This gives the model a
#     linguistic "thought" step it can lean on Qwen's pretrained knowledge
#     for, rather than mapping codec ids to codec ids blind -- the latter
#     is a much harder problem to learn from ~100k pairs with randomly
#     initialized codec embeddings, which is likely why your shrink/extend
#     codec-only runs underperformed.

#     Enabled automatically when --vocab_mode extend and the jsonl has
#     "en_text"/"hi_text" fields; disable with --no_text_bridge to fall back
#     to codec-only even in extend mode.

# Reverse pairs: --include_reverse (on by default) also trains hi->en from
# the same rows, roughly doubling effective data. Works in both codec-only
# and text-bridge modes; uses the same LANG_EN/LANG_HI tags, just swapped.

# Sequence layout:

#   Text-bridge:
#     [BOS][LANG_SRC] src_text [T2C] src_codec [SEP][LANG_TGT] tgt_text [T2C] tgt_codec [EOS]
#     labels: -100 through [SEP][LANG_TGT], then real ids for tgt_text [T2C] tgt_codec [EOS]

#   Codec-only (unchanged from before):
#     [BOS][LANG_SRC] src_codec [SEP][LANG_TGT] tgt_codec [EOS]
#     labels: -100 through [SEP][LANG_TGT], then real ids for tgt_codec [EOS]

# Input file format (one line = one utterance pair), produced by
# extract_codec_tokens.py, optionally with en_text/hi_text added:
#   {"id": "...", "en_tokens": [...], "hi_tokens": [...],
#    "en_text": "...", "hi_text": "..."}   <- text fields optional

# A vocab_spec.json is written next to the checkpoint recording exactly
# which ids mean what and whether text-bridge was used -- inference code
# should load that rather than hardcoding offsets/mode.

# Usage:
#   # text-bridge (recommended given your codec-only results), with reverse pairs:
#   python train_qwen_codec_s2s.py \
#       --train_files data/mahendra_train_with_text.jsonl \
#       --eval_file data/mahendra_dev_with_text.jsonl \
#       --output_dir runs/qwen05b_textbridge_extend \
#       --vocab_mode extend \
#       --base_model Qwen/Qwen2.5-0.5B \
#       --epochs 10 --batch_size 16 --grad_accum 4

#   # codec-only, no text, no reverse (old behavior):
#   python train_qwen_codec_s2s.py \
#       --train_files data/mahendra_train.jsonl --eval_file data/mahendra_dev.jsonl \
#       --output_dir runs/qwen05b_codec_only --vocab_mode shrink \
#       --no_text_bridge --no_reverse
# """
# import argparse
# import inspect
# import json
# import os
# import random
# from dataclasses import dataclass, asdict, field

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     AutoConfig,
#     Trainer,
#     TrainingArguments,
# )

# CODEC_VOCAB_SIZE = 12800  # MioCodec single-codebook vocab
# N_SPECIAL = 7  # PAD, BOS, EOS, SEP, T2C, LANG_EN, LANG_HI


# @dataclass
# class VocabSpec:
#     mode: str
#     codec_offset: int  # add this to raw MioCodec ids [0,12799] to get model ids
#     pad: int
#     bos: int
#     eos: int
#     sep: int
#     lang_en: int
#     lang_hi: int
#     total_vocab: int
#     # added for text-bridge; default so OLD vocab_spec.json files (saved
#     # before this feature existed) still load without error
#     t2c: int = -1
#     text_bridge: bool = False

#     def save(self, output_dir: str):
#         with open(os.path.join(output_dir, "vocab_spec.json"), "w") as f:
#             json.dump(asdict(self), f, indent=2)

#     @staticmethod
#     def load(output_dir: str) -> "VocabSpec":
#         with open(os.path.join(output_dir, "vocab_spec.json")) as f:
#             return VocabSpec(**json.load(f))


# def make_vocab_spec(mode: str, base_vocab_size: int, text_bridge: bool) -> VocabSpec:
#     if mode == "shrink":
#         codec_offset = 0
#         special_start = CODEC_VOCAB_SIZE
#     elif mode == "extend":
#         codec_offset = base_vocab_size
#         special_start = base_vocab_size + CODEC_VOCAB_SIZE
#     else:
#         raise ValueError(f"unknown vocab_mode {mode}")
#     pad, bos, eos, sep, t2c, lang_en, lang_hi = range(special_start, special_start + N_SPECIAL)
#     return VocabSpec(
#         mode=mode, codec_offset=codec_offset, pad=pad, bos=bos, eos=eos,
#         sep=sep, lang_en=lang_en, lang_hi=lang_hi,
#         total_vocab=special_start + N_SPECIAL,
#         t2c=t2c, text_bridge=text_bridge,
#     )


# def build_model(base_model_name: str, mode: str, text_bridge: bool):
#     config = AutoConfig.from_pretrained(base_model_name)
#     model = AutoModelForCausalLM.from_pretrained(base_model_name)
#     tokenizer = AutoTokenizer.from_pretrained(base_model_name)
#     base_vocab_size = model.get_input_embeddings().weight.shape[0]
#     spec = make_vocab_spec(mode, base_vocab_size, text_bridge)

#     model.resize_token_embeddings(spec.total_vocab)

#     if mode == "shrink":
#         std = getattr(config, "initializer_range", 0.02)
#         nn.init.normal_(model.get_input_embeddings().weight, mean=0.0, std=std)
#         out_emb = model.get_output_embeddings()
#         if out_emb is not None and not model.config.tie_word_embeddings:
#             nn.init.normal_(out_emb.weight, mean=0.0, std=std)
#     # in "extend" mode rows [0, base_vocab_size) are Qwen's pretrained text
#     # embeddings and are deliberately left untouched -- that's the whole
#     # point of text-bridge mode.

#     model.config.vocab_size = spec.total_vocab
#     model.config.pad_token_id = spec.pad
#     model.config.bos_token_id = spec.bos
#     model.config.eos_token_id = spec.eos
#     return model, tokenizer, spec


# # --------------------------------------------------------------------------
# # Dataset
# # --------------------------------------------------------------------------
# class CodecPairDataset(Dataset):
#     def __init__(self, paths, spec: VocabSpec, tokenizer=None, include_reverse=True,
#                  max_src_len=400, max_tgt_len=400, max_text_len=64):
#         self.spec = spec
#         self.tokenizer = tokenizer
#         self.examples = []  # each: (src_lang, src_text_ids, src_codec_raw, tgt_lang, tgt_text_ids, tgt_codec_raw)

#         if spec.text_bridge and tokenizer is None:
#             raise ValueError("text_bridge=True requires a tokenizer")

#         n_rows = 0
#         for path in paths:
#             with open(path) as f:
#                 for line in f:
#                     line = line.strip()
#                     if not line:
#                         continue
#                     rec = json.loads(line)
#                     en_codec = rec["en_tokens"][:max_src_len]
#                     hi_codec = rec["hi_tokens"][:max_tgt_len]
#                     if len(en_codec) == 0 or len(hi_codec) == 0:
#                         continue

#                     if spec.text_bridge:
#                         if "en_text" not in rec or "hi_text" not in rec:
#                             raise ValueError(
#                                 f"text_bridge=True but record {rec.get('id')} in {path} is missing "
#                                 f"en_text/hi_text. Re-run without --no_text_bridge disabled, or use a "
#                                 f"jsonl that has both text fields (e.g. mahendra_train_with_text.jsonl)."
#                             )
#                         en_text_ids = tokenizer.encode(rec["en_text"], add_special_tokens=False)[:max_text_len]
#                         hi_text_ids = tokenizer.encode(rec["hi_text"], add_special_tokens=False)[:max_text_len]
#                     else:
#                         en_text_ids, hi_text_ids = [], []

#                     self.examples.append(("en", en_text_ids, en_codec, "hi", hi_text_ids, hi_codec))
#                     if include_reverse:
#                         self.examples.append(("hi", hi_text_ids, hi_codec, "en", en_text_ids, en_codec))
#                     n_rows += 1

#         random.shuffle(self.examples)
#         print(f"Loaded {n_rows} source rows -> {len(self.examples)} training examples "
#               f"from {paths} (vocab_mode={spec.mode}, text_bridge={spec.text_bridge}, "
#               f"reverse={include_reverse})")

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         spec = self.spec
#         off = spec.codec_offset
#         src_lang, src_text_ids, src_codec_raw, tgt_lang, tgt_text_ids, tgt_codec_raw = self.examples[idx]

#         src_lang_tok = spec.lang_en if src_lang == "en" else spec.lang_hi
#         tgt_lang_tok = spec.lang_en if tgt_lang == "en" else spec.lang_hi
#         src_codec = [t + off for t in src_codec_raw]
#         tgt_codec = [t + off for t in tgt_codec_raw]

#         if spec.text_bridge:
#             src_block = [spec.bos, src_lang_tok] + src_text_ids + [spec.t2c] + src_codec
#             tgt_block = tgt_text_ids + [spec.t2c] + tgt_codec
#         else:
#             src_block = [spec.bos, src_lang_tok] + src_codec
#             tgt_block = tgt_codec

#         input_ids = src_block + [spec.sep, tgt_lang_tok] + tgt_block + [spec.eos]
#         prefix_len = len(src_block) + 2  # + [SEP, tgt_lang_tok]
#         labels = [-100] * prefix_len + tgt_block + [spec.eos]
#         assert len(input_ids) == len(labels)
#         return {"input_ids": input_ids, "labels": labels}


# @dataclass
# class PadCollator:
#     pad_id: int

#     def __call__(self, batch):
#         max_len = max(len(ex["input_ids"]) for ex in batch)
#         input_ids, labels, attn = [], [], []
#         for ex in batch:
#             ids, lab = ex["input_ids"], ex["labels"]
#             pad_n = max_len - len(ids)
#             input_ids.append(ids + [self.pad_id] * pad_n)
#             labels.append(lab + [-100] * pad_n)
#             attn.append([1] * len(ids) + [0] * pad_n)
#         return {
#             "input_ids": torch.tensor(input_ids, dtype=torch.long),
#             "labels": torch.tensor(labels, dtype=torch.long),
#             "attention_mask": torch.tensor(attn, dtype=torch.long),
#         }


# # --------------------------------------------------------------------------
# # Inference helpers (greedy AR generation, for sanity-checking a checkpoint).
# # Always take/return RAW MioCodec ids in [0, 12799] regardless of
# # vocab_mode -- offset bookkeeping is handled internally.
# # --------------------------------------------------------------------------
# @torch.inference_mode()
# def translate_codec_only(model, spec: VocabSpec, src_lang: str, src_codec_raw, tgt_lang: str,
#                           device="cuda", max_new_tokens=400):
#     """Codec-only inference: src_codec -> tgt_codec. Works for shrink or extend
#     checkpoints trained with spec.text_bridge=False."""
#     model.eval()
#     off = spec.codec_offset
#     src_lang_tok = spec.lang_en if src_lang == "en" else spec.lang_hi
#     tgt_lang_tok = spec.lang_en if tgt_lang == "en" else spec.lang_hi
#     src_codec = [t + off for t in src_codec_raw]
#     prompt = torch.tensor([[spec.bos, src_lang_tok] + src_codec + [spec.sep, tgt_lang_tok]], device=device)
#     out = model.generate(
#         prompt, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
#         eos_token_id=spec.eos, pad_token_id=spec.pad,
#     )
#     generated = out[0, prompt.shape[1]:].tolist()
#     if spec.eos in generated:
#         generated = generated[: generated.index(spec.eos)]
#     return [t - off for t in generated]


# @torch.inference_mode()
# def translate_with_text_bridge(model, tokenizer, spec: VocabSpec, src_lang: str, src_text: str,
#                                 src_codec_raw, tgt_lang: str, device="cuda", max_new_tokens=400):
#     """Text-bridge inference: (src_text, src_codec) -> (tgt_text, tgt_codec).
#     Requires src_text at inference time (e.g. from an ASR pass), matching
#     how the model was trained. Returns (tgt_text: str, tgt_codec_raw: list[int])."""
#     assert spec.text_bridge, "this checkpoint was not trained with text_bridge=True"
#     model.eval()
#     off = spec.codec_offset
#     src_lang_tok = spec.lang_en if src_lang == "en" else spec.lang_hi
#     tgt_lang_tok = spec.lang_en if tgt_lang == "en" else spec.lang_hi
#     src_text_ids = tokenizer.encode(src_text, add_special_tokens=False)
#     src_codec = [t + off for t in src_codec_raw]

#     prompt_ids = [spec.bos, src_lang_tok] + src_text_ids + [spec.t2c] + src_codec + [spec.sep, tgt_lang_tok]
#     prompt = torch.tensor([prompt_ids], device=device)
#     out = model.generate(
#         prompt, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
#         eos_token_id=spec.eos, pad_token_id=spec.pad,
#     )
#     generated = out[0, prompt.shape[1]:].tolist()
#     if spec.eos in generated:
#         generated = generated[: generated.index(spec.eos)]

#     if spec.t2c in generated:
#         split = generated.index(spec.t2c)
#         tgt_text_ids, tgt_codec_ids = generated[:split], generated[split + 1:]
#     else:
#         # model failed to emit the T2C marker -- treat everything as text,
#         # no codec tokens produced (caller should treat this as a failure case)
#         tgt_text_ids, tgt_codec_ids = generated, []

#     tgt_text = tokenizer.decode(tgt_text_ids, skip_special_tokens=True)
#     tgt_codec_raw = [t - off for t in tgt_codec_ids]
#     return tgt_text, tgt_codec_raw


# # --------------------------------------------------------------------------
# # transformers version drift compatibility layer (unchanged from before)
# # --------------------------------------------------------------------------
# def build_training_args(args, do_eval: bool) -> TrainingArguments:
#     strategy = "epoch" if do_eval else "no"
#     desired = dict(
#         output_dir=args.output_dir,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.grad_accum,
#         num_train_epochs=args.epochs,
#         learning_rate=args.lr,
#         lr_scheduler_type="cosine",
#         warmup_ratio=args.warmup_ratio,
#         bf16=args.bf16,
#         logging_steps=20,
#         save_strategy="epoch",
#         save_total_limit=3,
#         report_to=[],
#         remove_unused_columns=False,
#         eval_strategy=strategy,
#         evaluation_strategy=strategy,
#         load_best_model_at_end=do_eval,
#         metric_for_best_model="eval_loss" if do_eval else None,
#         greater_is_better=False if do_eval else None,
#     )
#     desired = {k: v for k, v in desired.items() if v is not None}

#     valid_params = set(inspect.signature(TrainingArguments.__init__).parameters)
#     filtered = {k: v for k, v in desired.items() if k in valid_params}
#     dropped = sorted(set(desired) - set(filtered))
#     if dropped:
#         print(f"[compat] this transformers version doesn't support {dropped} -- skipping them.")
#     if not do_eval:
#         filtered.pop("load_best_model_at_end", None)
#     return TrainingArguments(**filtered)


# # --------------------------------------------------------------------------
# # Main
# # --------------------------------------------------------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--train_files", nargs="+", required=True)
#     ap.add_argument("--eval_file", default=None)
#     ap.add_argument("--output_dir", required=True)
#     ap.add_argument("--vocab_mode", choices=["extend", "shrink"], default="extend")
#     ap.add_argument("--no_text_bridge", action="store_true",
#                      help="disable text-bridge even in extend mode (fall back to codec-only)")
#     ap.add_argument("--no_reverse", action="store_true",
#                      help="disable automatic hi->en reverse pairs (default: reverse pairs included)")
#     ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B")
#     ap.add_argument("--epochs", type=float, default=10)
#     ap.add_argument("--batch_size", type=int, default=16)
#     ap.add_argument("--grad_accum", type=int, default=4)
#     ap.add_argument("--lr", type=float, default=2e-4)
#     ap.add_argument("--max_src_len", type=int, default=400)
#     ap.add_argument("--max_tgt_len", type=int, default=400)
#     ap.add_argument("--max_text_len", type=int, default=64)
#     ap.add_argument("--warmup_ratio", type=float, default=0.03)
#     ap.add_argument("--bf16", action="store_true", default=True)
#     args = ap.parse_args()

#     text_bridge = (args.vocab_mode == "extend") and (not args.no_text_bridge)
#     if args.no_text_bridge and args.vocab_mode == "shrink":
#         pass  # no-op, shrink never supports text_bridge anyway
#     include_reverse = not args.no_reverse

#     os.makedirs(args.output_dir, exist_ok=True)
#     model, tokenizer, spec = build_model(args.base_model, args.vocab_mode, text_bridge)
#     spec.save(args.output_dir)
#     tokenizer.save_pretrained(args.output_dir)
#     print(f"vocab_mode={spec.mode} text_bridge={spec.text_bridge} total_vocab={spec.total_vocab} "
#           f"codec_offset={spec.codec_offset}")
#     print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

#     train_ds = CodecPairDataset(
#         args.train_files, spec, tokenizer=tokenizer if spec.text_bridge else None,
#         include_reverse=include_reverse, max_src_len=args.max_src_len,
#         max_tgt_len=args.max_tgt_len, max_text_len=args.max_text_len,
#     )
#     eval_ds = (
#         CodecPairDataset(
#             [args.eval_file], spec, tokenizer=tokenizer if spec.text_bridge else None,
#             include_reverse=include_reverse, max_src_len=args.max_src_len,
#             max_tgt_len=args.max_tgt_len, max_text_len=args.max_text_len,
#         )
#         if args.eval_file else None
#     )

#     training_args = build_training_args(args, do_eval=eval_ds is not None)

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=eval_ds,
#         data_collator=PadCollator(pad_id=spec.pad),
#     )
#     trainer.train()
#     trainer.save_model(args.output_dir)
#     spec.save(args.output_dir)
#     tokenizer.save_pretrained(args.output_dir)
#     print(f"Saved to {args.output_dir}")

#     if len(train_ds) > 0:
#         src_lang, src_text_ids, src_codec_raw, tgt_lang, tgt_text_ids, tgt_codec_raw = train_ds.examples[0]
#         if spec.text_bridge:
#             src_text = tokenizer.decode(src_text_ids, skip_special_tokens=True)
#             pred_text, pred_codec = translate_with_text_bridge(
#                 model, tokenizer, spec, src_lang, src_text, src_codec_raw, tgt_lang, device=model.device
#             )
#             true_text = tokenizer.decode(tgt_text_ids, skip_special_tokens=True)
#             print(f"[sanity check] {src_lang}->{tgt_lang} true_text={true_text!r} pred_text={pred_text!r} "
#                   f"true_codec_len={len(tgt_codec_raw)} pred_codec_len={len(pred_codec)}")
#         else:
#             pred = translate_codec_only(model, spec, src_lang, src_codec_raw, tgt_lang, device=model.device)
#             print(f"[sanity check] {src_lang}->{tgt_lang} true_len={len(tgt_codec_raw)} pred_len={len(pred)}")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python
#!/usr/bin/env python
"""
qwen_s2st.py  (v2: curriculum training)
---------------------------------------
Qwen2.5 as a discrete-token speech-to-speech translation model (EN <-> HI)
via MioCodec content tokens, with a TEXT BRIDGE:

  speech:  [BOS][LANG_SRC] src_text [T2C] src_codec [SEP][LANG_TGT] tgt_text [T2C] tgt_codec [EOS]
  text MT: [BOS][LANG_SRC] src_text [SEP][LANG_TGT] tgt_text [EOS]

CURRICULUM
  Stage 1  "mt"    text-only en<->hi pairs (--mt_files). Teaches/refreshes Hindi<->English
                   translation BEFORE any audio is involved. LoRA (default) or full FT,
                   low LR. The result is merged into the base weights and exported to
                   <output_dir>/stage_mt.
  Stage 2  "s2st"  a NEW LoRA on top of the merged stage-1 model, trained on
                   en_text + en_audio -> hi_text + hi_audio (both directions), with
                   text-only MT examples replayed (--mt_fraction) so translation is
                   not forgotten, plus source-codec dropout.
                   The codec/special embedding rows are trained too, with a higher
                   LR (--emb_lr_mult).

MEMORY TRICK (important for small GPUs)
  The embedding table is split into  old (Qwen's 151,936 text rows, frozen, never
  touched)  and  new (12,807 codec+special rows, trainable, only ~46 MB). This removes
  the 564 MiB dense embedding gradient + 1.2 GB of Adam state that caused the OOM,
  and guarantees Qwen's text embeddings stay bit-identical. Everything is merged back
  into a normal HF checkpoint when saving.

OTHER
  * Weighted loss: text tokens x --text_weight, [T2C]/[EOS] x --special_weight.
    Train/eval text loss and codec loss are logged separately.
  * Loss is computed in checkpointed chunks (--loss_chunk) so the 164k-vocab logits
    never all sit in memory.
  * Trainables (LoRA + new rows) are saved every --save_steps and each epoch to
    <output_dir>/trainables_*.pt (small). Resume stage 2 with --resume_trainables.
  * Generation eval at the end of each stage (chrF if sacrebleu is installed).

Input formats (jsonl)
  speech rows (--train_files / --eval_file):
     {"id":..., "en_tokens":[...], "hi_tokens":[...], "en_text":"...", "hi_text":"..."}
     rows with empty text are skipped.
  MT rows (--mt_files): {"en_text","hi_text"} | {"en","hi"} | {"english","hindi"} |
     {"src","tgt"} | {"translation":{"en","hi"}}

Examples
  # smoke test (a few minutes): verify everything runs
  python qwen_s2st.py --train_files data/train.jsonl --eval_file data/dev.jsonl \
      --mt_files indictrans2.jsonl --output_dir runs3/smoke --limit_rows 300 \
      --mt_epochs 1 --epochs 1 --gen_eval_n 4

  # full run (reusing the already warmed-up codec rows from the old stage0):
  python qwen_s2st.py \
      --init_from ./runs2/qwen05b_s2st/stage0 \
      --train_files ./data/mahendra_train_with_text.jsonl \
      --eval_file ./data/mahendra_test_with_text.jsonl \
      --mt_files ./indictrans2.jsonl --output_dir ./runs3/qwen05b_s2st \
      --stages both --mt_epochs 1 --epochs 3 \
      --mt_batch_size 16 --mt_grad_accum 2 --batch_size 3 --grad_accum 12 --grad_ckpt

  # only stage 1, or only stage 2 from a stage-1 export:
  python qwen_s2st.py ... --stages mt
  python qwen_s2st.py ... --stages s2st --init_from ./runs3/qwen05b_s2st/stage_mt
"""
#!/usr/bin/env python
"""
qwen_s2st.py  (v2: curriculum training)
---------------------------------------
Qwen2.5 as a discrete-token speech-to-speech translation model (EN <-> HI)
via MioCodec content tokens, with a TEXT BRIDGE:

  speech:  [BOS][LANG_SRC] src_text [T2C] src_codec [SEP][LANG_TGT] tgt_text [T2C] tgt_codec [EOS]
  text MT: [BOS][LANG_SRC] src_text [SEP][LANG_TGT] tgt_text [EOS]

CURRICULUM
  Stage 1  "mt"    text-only en<->hi pairs (--mt_files). Teaches/refreshes Hindi<->English
                   translation BEFORE any audio is involved. LoRA (default) or full FT,
                   low LR. The result is merged into the base weights and exported to
                   <output_dir>/stage_mt.
  Stage 2  "s2st"  a NEW LoRA on top of the merged stage-1 model, trained on
                   en_text + en_audio -> hi_text + hi_audio (both directions), with
                   text-only MT examples replayed (--mt_fraction) so translation is
                   not forgotten, plus source-codec dropout.
                   The codec/special embedding rows are trained too, with a higher
                   LR (--emb_lr_mult).

MEMORY TRICK (important for small GPUs)
  The embedding table is split into  old (Qwen's 151,936 text rows, frozen, never
  touched)  and  new (12,807 codec+special rows, trainable, only ~46 MB). This removes
  the 564 MiB dense embedding gradient + 1.2 GB of Adam state that caused the OOM,
  and guarantees Qwen's text embeddings stay bit-identical. Everything is merged back
  into a normal HF checkpoint when saving.

OTHER
  * Weighted loss: text tokens x --text_weight, [T2C]/[EOS] x --special_weight.
    Train/eval text loss and codec loss are logged separately.
  * Loss is computed in checkpointed chunks (--loss_chunk) so the 164k-vocab logits
    never all sit in memory.
  * Trainables (LoRA + new rows) are saved every --save_steps and each epoch to
    <output_dir>/trainables_*.pt (small). Resume stage 2 with --resume_trainables.
  * Generation eval at the end of each stage (chrF if sacrebleu is installed).

Input formats (jsonl)
  speech rows (--train_files / --eval_file):
     {"id":..., "en_tokens":[...], "hi_tokens":[...], "en_text":"...", "hi_text":"..."}
     rows with empty text are skipped.
  MT rows (--mt_files): {"en_text","hi_text"} | {"en","hi"} | {"english","hindi"} |
     {"src","tgt"} | {"translation":{"en","hi"}}

Examples
  # smoke test (a few minutes): verify everything runs
  python qwen_s2st.py --train_files data/train.jsonl --eval_file data/dev.jsonl \
      --mt_files indictrans2.jsonl --output_dir runs3/smoke --limit_rows 300 \
      --mt_epochs 1 --epochs 1 --gen_eval_n 4

  # full run (reusing the already warmed-up codec rows from the old stage0):
  python qwen_s2st.py \
      --init_from ./runs2/qwen05b_s2st/stage0 \
      --train_files ./data/mahendra_train_with_text.jsonl \
      --eval_file ./data/mahendra_test_with_text.jsonl \
      --mt_files ./indictrans2.jsonl --output_dir ./runs3/qwen05b_s2st \
      --stages both --mt_epochs 1 --epochs 3 \
      --mt_batch_size 16 --mt_grad_accum 2 --batch_size 3 --grad_accum 12 --grad_ckpt

  # only stage 1, or only stage 2 from a stage-1 export:
  python qwen_s2st.py ... --stages mt
  python qwen_s2st.py ... --stages s2st --init_from ./runs3/qwen05b_s2st/stage_mt
"""
import argparse
import gc
import inspect
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

CODEC_VOCAB_SIZE = 12800  # MioCodec single-codebook vocab
N_SPECIAL = 7  # PAD, BOS, EOS, SEP, T2C, LANG_EN, LANG_HI


# --------------------------------------------------------------------------
# Vocab spec
# --------------------------------------------------------------------------
@dataclass
class VocabSpec:
    mode: str
    codec_offset: int  # add to raw MioCodec ids [0,12799] to get model ids
    pad: int
    bos: int
    eos: int
    sep: int
    lang_en: int
    lang_hi: int
    total_vocab: int
    t2c: int = -1
    text_bridge: bool = True

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "vocab_spec.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(output_dir: str) -> "VocabSpec":
        with open(os.path.join(output_dir, "vocab_spec.json")) as f:
            return VocabSpec(**json.load(f))


def make_vocab_spec(base_vocab_size: int) -> VocabSpec:
    start = base_vocab_size + CODEC_VOCAB_SIZE
    pad, bos, eos, sep, t2c, lang_en, lang_hi = range(start, start + N_SPECIAL)
    return VocabSpec(
        mode="extend", codec_offset=base_vocab_size, pad=pad, bos=bos, eos=eos,
        sep=sep, lang_en=lang_en, lang_hi=lang_hi,
        total_vocab=start + N_SPECIAL, t2c=t2c, text_bridge=True,
    )


# --------------------------------------------------------------------------
# Split embeddings: frozen text rows + small trainable codec/special rows
# --------------------------------------------------------------------------
class SplitEmbedding(nn.Module):
    def __init__(self, old, new):
        super().__init__()
        self.old = old if isinstance(old, nn.Parameter) else nn.Parameter(old, requires_grad=False)
        self.new = new if isinstance(new, nn.Parameter) else nn.Parameter(new.float())
        self.num_embeddings = self.old.shape[0] + self.new.shape[0]
        self.embedding_dim = self.old.shape[1]
        self.padding_idx = None

    def forward(self, ids):
        n = self.old.shape[0]
        is_new = (ids >= n).unsqueeze(-1)
        a = F.embedding(ids.clamp(max=n - 1), self.old)
        b = F.embedding((ids - n).clamp(min=0), self.new).to(a.dtype)
        return torch.where(is_new, b, a)

    @property
    def weight(self):  # only for compatibility; builds a copy
        return torch.cat([self.old, self.new.to(self.old.dtype)], 0)


class SplitHead(nn.Module):
    def __init__(self, old, new):
        super().__init__()
        self.old = old if isinstance(old, nn.Parameter) else nn.Parameter(old, requires_grad=False)
        self.new = new if isinstance(new, nn.Parameter) else nn.Parameter(new.float())
        self.in_features = self.old.shape[1]
        self.out_features = self.old.shape[0] + self.new.shape[0]

    def forward(self, x):
        return torch.cat([F.linear(x, self.old.to(x.dtype)),
                          F.linear(x, self.new.to(x.dtype))], dim=-1)

    @property
    def weight(self):
        return torch.cat([self.old, self.new.to(self.old.dtype)], 0)


@torch.no_grad()
def init_new_rows(model, n_old: int):
    """New rows = mean of text embeddings + per-dim-std noise."""
    seen = set()
    for m in (model.get_input_embeddings(), model.get_output_embeddings()):
        if m is None or id(m.weight) in seen:
            continue
        seen.add(id(m.weight))
        w = m.weight
        old = w[:n_old].float()
        mean, std = old.mean(0, keepdim=True), old.std(0, keepdim=True)
        w[n_old:] = (mean + torch.randn_like(w[n_old:].float()) * std).to(w.dtype)


@torch.no_grad()
def split_embeddings(model, n_old: int):
    emb = model.get_input_embeddings()
    if isinstance(emb, SplitEmbedding):
        return
    head = model.get_output_embeddings()
    W = emb.weight
    tied = head is not None and head.weight.data_ptr() == W.data_ptr()
    se = SplitEmbedding(W[:n_old].detach().clone(), W[n_old:].detach().clone())
    if head is not None and not tied:
        H = head.weight
        new_head = SplitHead(H[:n_old].detach().clone(), H[n_old:].detach().clone())
    elif head is not None:
        new_head = SplitHead(se.old, se.new)  # share the same Parameters
    model.set_input_embeddings(se)
    if head is not None:
        model.set_output_embeddings(new_head)
    del W
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def unsplit_embeddings(model):
    """Merge split rows back into normal modules. Returns True if head is tied to embeddings."""
    emb = model.get_input_embeddings()
    if not isinstance(emb, SplitEmbedding):
        return None
    head = model.get_output_embeddings()
    tied = isinstance(head, SplitHead) and head.old is emb.old
    W = torch.cat([emb.old.data, emb.new.data.to(emb.old.dtype)], 0)
    new_emb = nn.Embedding(W.shape[0], W.shape[1], _weight=W)
    model.set_input_embeddings(new_emb)
    if isinstance(head, SplitHead):
        lin = nn.Linear(W.shape[1], W.shape[0], bias=False, device="meta")
        if tied:
            lin.weight = new_emb.weight
        else:
            lin.weight = nn.Parameter(torch.cat([head.old.data, head.new.data.to(head.old.dtype)], 0))
        model.set_output_embeddings(lin)
    return tied


def _load_lm(path, dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)


def build_fresh(base_model_name: str, dtype):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = _load_lm(base_model_name, dtype)
    base_vocab = model.get_input_embeddings().weight.shape[0]
    spec = make_vocab_spec(base_vocab)
    model.resize_token_embeddings(spec.total_vocab)
    init_new_rows(model, base_vocab)
    model.config.vocab_size = spec.total_vocab
    for cfg in (model.config, model.generation_config):
        cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id = spec.pad, spec.bos, spec.eos
    return model, tokenizer, spec


def export_model(model, spec, tokenizer, out_dir, resplit=True):
    """Save a normal HF checkpoint (embeddings merged back). Model must not be PEFT-wrapped."""
    tied = unsplit_embeddings(model)
    if tied:  # without this flag some transformers versions reload a RANDOM lm_head
        model.config.tie_word_embeddings = True
    model.config.use_cache = True
    model.save_pretrained(out_dir)
    spec.save(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[export] saved merged model -> {out_dir}")
    if resplit:
        split_embeddings(model, spec.codec_offset)


# --------------------------------------------------------------------------
# Trainable-parameter configuration
# --------------------------------------------------------------------------
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def wrap_lora(model, args):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise SystemExit("LoRA needs peft:  pip install peft")
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                     target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def strip_peft_attrs(model):
    for a in ("peft_config", "_hf_peft_config_loaded"):
        if hasattr(model, a):
            try:
                delattr(model, a)
            except Exception:
                pass


def report_trainable(model, tag):
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] trainable params: {n_tr/1e6:.1f}M / {n_all/1e6:.1f}M")


def configure_mt(model, args):
    for p in model.parameters():
        p.requires_grad = False
    if args.mt_mode == "lora":
        model = wrap_lora(model, args)
    else:  # full FT of the transformer; embeddings (old and new) stay frozen
        for n, p in model.named_parameters():
            if not (n.endswith(".old") or n.endswith(".new")):
                p.requires_grad = True
    report_trainable(model, f"stage mt/{args.mt_mode}")
    return model


def configure_s2st(model, args):
    for p in model.parameters():
        p.requires_grad = False
    model = wrap_lora(model, args)  # fresh adapter on the merged stage-1 weights
    for n, p in model.named_parameters():
        if n.endswith(".new"):
            p.requires_grad = True
    report_trainable(model, "stage s2st/lora+codec rows")
    return model


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def iter_jsonl(path, limit=0):
    """Yield dict rows. Invalid lines are skipped (first few are printed)."""
    n, bad = 0, 0
    with open(path, encoding="utf-8-sig") as f:  # utf-8-sig also strips a BOM
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                rec = None
            if not isinstance(rec, dict):
                bad += 1
                if bad <= 5:
                    print(f"[warn] {path}:{ln}: not a JSON object, skipping: {line[:80]!r}")
                continue
            yield rec
            n += 1
            if limit and n >= limit:
                break
    if n == 0 and bad > 0:
        raise ValueError(f"{path}: no valid JSON-lines rows found ({bad} bad lines). "
                         f"Is this the right file (a JSON array, LFS pointer, or log output)?")
    if bad:
        print(f"[warn] {path}: skipped {bad} invalid lines")


def encode_many(tok, texts, bs=2048):
    out = []
    for i in range(0, len(texts), bs):
        out.extend(tok(texts[i:i + bs], add_special_tokens=False)["input_ids"])
    return out


def _ok_text(s):
    return isinstance(s, str) and s.strip() != ""


def load_speech(paths, tok, *, include_reverse, max_codec_len, max_text_len, limit=0):
    """-> (speech_items, text_only_items_derived_from_the_same_rows)"""
    rows, missing, skipped = [], 0, 0
    for path in paths:
        for rec in iter_jsonl(path, limit):
            en_c, hi_c = rec.get("en_tokens"), rec.get("hi_tokens")
            if not en_c or not hi_c or len(en_c) > max_codec_len or len(hi_c) > max_codec_len:
                skipped += 1
                continue
            if not (_ok_text(rec.get("en_text")) and _ok_text(rec.get("hi_text"))):
                missing += 1
                if missing <= 5:
                    print(f"[warn] skipping {rec.get('id')} in {path}: empty/missing en_text/hi_text")
                continue
            rows.append((en_c, hi_c, rec["en_text"], rec["hi_text"]))
    en_ids = encode_many(tok, [r[2] for r in rows])
    hi_ids = encode_many(tok, [r[3] for r in rows])
    speech, derived = [], []
    for (en_c, hi_c, _, _), en_t, hi_t in zip(rows, en_ids, hi_ids):
        if not en_t or not hi_t or len(en_t) > max_text_len or len(hi_t) > max_text_len:
            skipped += 1  # skip rather than truncate (truncated text vs full audio is harmful)
            continue
        speech.append(("speech", "en", en_t, en_c, "hi", hi_t, hi_c))
        derived.append(("mt", "en", en_t, None, "hi", hi_t, None))
        if include_reverse:
            speech.append(("speech", "hi", hi_t, hi_c, "en", en_t, en_c))
            derived.append(("mt", "hi", hi_t, None, "en", en_t, None))
    print(f"[data] {paths}: {len(speech)} speech examples "
          f"(skipped {skipped} over-length, {missing} missing-text rows)")
    return speech, derived


def _get_pair(rec):
    if isinstance(rec.get("translation"), dict):
        rec = rec["translation"]
    for ek, hk in (("en_text", "hi_text"), ("en", "hi"), ("english", "hindi"), ("src", "tgt")):
        e, h = rec.get(ek), rec.get(hk)
        if _ok_text(e) and _ok_text(h):
            return e, h
    return None, None


def load_mt(paths, tok, *, include_reverse, max_text_len, limit=0, max_examples=0):
    pairs, bad = [], 0
    for path in paths:
        first = True
        for rec in iter_jsonl(path, limit):
            if first:
                print(f"[mt] {path}: first row keys = {list(rec.keys())}")
                first = False
            e, h = _get_pair(rec)
            if e is None:
                bad += 1
            else:
                pairs.append((e, h))
    if max_examples and len(pairs) > max_examples:
        pairs = random.sample(pairs, max_examples)
    en_ids = encode_many(tok, [p[0] for p in pairs])
    hi_ids = encode_many(tok, [p[1] for p in pairs])
    items = []
    for en_t, hi_t in zip(en_ids, hi_ids):
        if not en_t or not hi_t or len(en_t) > max_text_len or len(hi_t) > max_text_len:
            continue
        items.append(("mt", "en", en_t, None, "hi", hi_t, None))
        if include_reverse:
            items.append(("mt", "hi", hi_t, None, "en", en_t, None))
    print(f"[mt] loaded {len(items)} MT examples ({bad} unusable rows)")
    if not items:
        print("[warn] NO usable rows in --mt_files; check the key names printed above")
    return items


class ItemsDataset(Dataset):
    """item = (kind, src_lang, src_text_ids, src_codec_raw, tgt_lang, tgt_text_ids, tgt_codec_raw)"""

    def __init__(self, items, spec, drop_src_codec_p=0.0):
        self.items, self.spec, self.drop_p = items, spec, drop_src_codec_p

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s, off = self.spec, self.spec.codec_offset
        kind, sl, st, sc, tl, tt, tc = self.items[idx]
        sl_tok = s.lang_en if sl == "en" else s.lang_hi
        tl_tok = s.lang_en if tl == "en" else s.lang_hi
        if kind == "mt":
            src = [s.bos, sl_tok] + st
            tgt = tt
        else:
            keep = not (self.drop_p > 0 and random.random() < self.drop_p)
            src = [s.bos, sl_tok] + st + [s.t2c] + ([t + off for t in sc] if keep else [])
            tgt = tt + [s.t2c] + [t + off for t in tc]
        input_ids = src + [s.sep, tl_tok] + tgt + [s.eos]
        labels = [-100] * (len(src) + 2) + tgt + [s.eos]
        return {"input_ids": input_ids, "labels": labels}


@dataclass
class PadCollator:
    pad_id: int

    def __call__(self, batch):
        n = max(len(ex["input_ids"]) for ex in batch)
        ids, labs, att = [], [], []
        for ex in batch:
            pad = n - len(ex["input_ids"])
            ids.append(ex["input_ids"] + [self.pad_id] * pad)
            labs.append(ex["labels"] + [-100] * pad)
            att.append([1] * len(ex["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labs),
                "attention_mask": torch.tensor(att)}


# --------------------------------------------------------------------------
# Trainer: weighted + chunked loss, split metrics, separate LR for new rows
# --------------------------------------------------------------------------
def _get_lm(model):
    m = getattr(model, "module", model)
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    return m


def _new_acc():
    return defaultdict(float)


class S2STTrainer(Trainer):
    def __init__(self, *a, spec, text_w=5.0, codec_w=1.0, special_w=3.0,
                 emb_lr_mult=1.0, loss_chunk=512, **kw):
        super().__init__(*a, **kw)
        self.spec, self.text_w, self.codec_w, self.special_w = spec, text_w, codec_w, special_w
        self.emb_lr_mult, self.loss_chunk = emb_lr_mult, loss_chunk
        self.model_accepts_loss_kwargs = False  # we normalise the loss ourselves
        self._train_acc, self._eval_acc = _new_acc(), _new_acc()

    # ---- optimizer: new codec rows get lr * emb_lr_mult, no weight decay ----
    def create_optimizer(self, *a, **kw):
        if self.optimizer is None:
            new_p, other = [], []
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    (new_p if n.endswith(".new") else other).append(p)
            lr = self.args.learning_rate
            groups = []
            if other:
                groups.append({"params": other, "lr": lr, "weight_decay": self.args.weight_decay})
            if new_p:
                groups.append({"params": new_p, "lr": lr * self.emb_lr_mult, "weight_decay": 0.0})
            cls, okw = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = cls(groups, **okw)
        return self.optimizer

    def _token_ce(self, lm, h, t):
        def f(hc, tc):
            return F.cross_entropy(lm.lm_head(hc).float(), tc, reduction="none")
        C = self.loss_chunk
        if C <= 0 or h.shape[0] <= C:
            return f(h, t)
        use_ckpt = torch.is_grad_enabled() and h.requires_grad
        outs = []
        for i in range(0, h.shape[0], C):
            hc, tc = h[i:i + C], t[i:i + C]
            outs.append(checkpoint(f, hc, tc, use_reentrant=False) if use_ckpt else f(hc, tc))
        return torch.cat(outs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        lm = _get_lm(model)
        labels = inputs["labels"]
        hidden = lm.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                          use_cache=False).last_hidden_state
        h, tgt = hidden[:, :-1], labels[:, 1:]
        keep = tgt != -100
        if keep.sum() == 0:
            return hidden.sum() * 0.0
        h_sel, t = h[keep], tgt[keep]
        per = self._token_ce(lm, h_sel, t)

        off = self.spec.codec_offset
        is_text = t < off
        is_codec = (t >= off) & (t < off + CODEC_VOCAB_SIZE)
        is_spec = t >= off + CODEC_VOCAB_SIZE
        w = is_text.float() * self.text_w + is_codec.float() * self.codec_w \
            + is_spec.float() * self.special_w
        loss = (per * w).sum() / w.sum().clamp(min=1e-6)

        with torch.no_grad():
            acc = self._train_acc if model.training else self._eval_acc
            d = per.detach()
            acc["text"] += d[is_text].sum().item();   acc["n_text"] += is_text.sum().item()
            acc["codec"] += d[is_codec].sum().item(); acc["n_codec"] += is_codec.sum().item()
            acc["spec"] += d[is_spec].sum().item();   acc["n_spec"] += is_spec.sum().item()
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def evaluate(self, *args, **kwargs):
        self._eval_acc = _new_acc()
        metrics = super().evaluate(*args, **kwargs)
        a, pre = self._eval_acc, kwargs.get("metric_key_prefix", "eval")
        extra = {
            f"{pre}_text_loss": a["text"] / max(a["n_text"], 1),
            f"{pre}_codec_loss": a["codec"] / max(a["n_codec"], 1),
            f"{pre}_special_loss": a["spec"] / max(a["n_spec"], 1),
        }
        metrics.update(extra)
        self.log(extra)
        return metrics

    def log(self, logs, *args, **kwargs):
        a = self._train_acc
        if "loss" in logs and (a["n_text"] + a["n_codec"]) > 0:
            logs = dict(logs)
            logs["train_text_loss"] = round(a["text"] / max(a["n_text"], 1), 4)
            logs["train_codec_loss"] = round(a["codec"] / max(a["n_codec"], 1), 4)
            self._train_acc = _new_acc()
        super().log(logs, *args, **kwargs)


class TrainablesSaver(TrainerCallback):
    """Saves only the trainable tensors (LoRA + new rows): small and quick."""

    def __init__(self, out_dir, tag, save_steps):
        self.out_dir, self.tag, self.save_steps = out_dir, tag, save_steps

    def _save(self, model, name):
        sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(sd, os.path.join(self.out_dir, name))

    def on_step_end(self, args, state, control, model=None, **kw):
        if model is not None and self.save_steps and state.global_step % self.save_steps == 0:
            self._save(model, f"trainables_{self.tag}_latest.pt")

    def on_epoch_end(self, args, state, control, model=None, **kw):
        if model is not None:
            ep = int(round(state.epoch)) if state.epoch is not None else 0
            self._save(model, f"trainables_{self.tag}_epoch{ep}.pt")


def build_training_args(args, out_dir, lr, epochs, bs, ga, do_eval):
    ev = "epoch" if do_eval else "no"
    desired = dict(
        output_dir=out_dir,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=ga,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        optim=args.optim,
        bf16=args.bf16,
        logging_steps=20,
        save_strategy="no",  # we save trainables / merged models ourselves
        report_to=[],
        remove_unused_columns=False,
        prediction_loss_only=True,
        eval_strategy=ev,
        evaluation_strategy=ev,
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_ckpt else None,
        seed=args.seed,
    )
    desired = {k: v for k, v in desired.items() if v is not None}
    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    filtered = {k: v for k, v in desired.items() if k in valid}
    dropped = sorted(set(desired) - set(filtered))
    if dropped:
        print(f"[compat] this transformers version doesn't support {dropped} -- skipping.")
    return TrainingArguments(**filtered)


def make_trainer(model, args, spec, train_ds, eval_ds, *, lr, epochs, bs, ga, emb_lr_mult, tag):
    targs = build_training_args(args, os.path.join(args.output_dir, f"_trainer_{tag}"),
                                lr, epochs, bs, ga, eval_ds is not None)
    return S2STTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=PadCollator(spec.pad),
        callbacks=[TrainablesSaver(args.output_dir, tag, args.save_steps)],
        spec=spec, text_w=args.text_weight, codec_w=args.codec_weight,
        special_w=args.special_weight, emb_lr_mult=emb_lr_mult, loss_chunk=args.loss_chunk,
    )


def finish_trainer(trainer):
    """Return the (unwrapped) model and release the trainer's GPU memory."""
    model = getattr(trainer.model, "module", trainer.model)
    trainer.optimizer, trainer.lr_scheduler = None, None
    try:
        trainer.accelerator.free_memory()
    except Exception:
        pass
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return model


def merge_if_peft(model):
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
        strip_peft_attrs(model)
    for p in model.parameters():
        p.requires_grad = False
    return model


# --------------------------------------------------------------------------
# Inference / evaluation helpers (greedy). Take/return RAW MioCodec ids.
# --------------------------------------------------------------------------
def prepare_for_inference(model):
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    model.config.use_cache = True
    model.eval()


@torch.inference_mode()
def translate(model, tokenizer, spec: VocabSpec, src_lang, tgt_lang, src_text,
              src_codec_raw=None, text_only=False, max_new_tokens=700):
    """speech mode: (src_text [+ src_codec]) -> (tgt_text, tgt_codec_raw)
       text_only=True: plain MT -> (tgt_text, [])"""
    model.eval()
    dev, off = next(model.parameters()).device, spec.codec_offset
    lang = lambda l: spec.lang_en if l == "en" else spec.lang_hi
    ids = [spec.bos, lang(src_lang)] + tokenizer.encode(src_text, add_special_tokens=False)
    if not text_only:
        ids.append(spec.t2c)
        if src_codec_raw:
            ids += [t + off for t in src_codec_raw]
    ids += [spec.sep, lang(tgt_lang)]
    prompt = torch.tensor([ids], device=dev)
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        out = model.generate(
            prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1, eos_token_id=spec.eos, pad_token_id=spec.pad,
        )
    gen = out[0, prompt.shape[1]:].tolist()
    if spec.eos in gen:
        gen = gen[: gen.index(spec.eos)]
    if spec.t2c in gen:
        k = gen.index(spec.t2c)
        text_ids, codec_ids = gen[:k], gen[k + 1:]
    else:
        text_ids, codec_ids = gen, []
    text_ids = [t for t in text_ids if t < min(len(tokenizer), off)]
    text = tokenizer.decode(text_ids, skip_special_tokens=True)
    codec = [t - off for t in codec_ids if off <= t < off + CODEC_VOCAB_SIZE]
    return text, ([] if text_only else codec)


def generation_eval(model, tokenizer, spec, items, n, mode, tag):
    """mode 'mt': text-only translation; mode 'speech': text-bridge speech translation."""
    want = "mt" if mode == "mt" else "speech"
    items = [it for it in items if it[0] == want]
    if not items or n <= 0:
        return
    prepare_for_inference(model)
    random.Random(0).shuffle(items)
    items = items[:n]
    try:
        from sacrebleu import corpus_chrf
    except ImportError:
        corpus_chrf = None
        print("[gen-eval] sacrebleu not installed (pip install sacrebleu) -> samples only")
    res = defaultdict(lambda: {"hyp": [], "ref": [], "has_codec": 0, "ratio": []})
    for i, (_, sl, st, sc, tl, tt, tc) in enumerate(items):
        src_text, ref = tokenizer.decode(st), tokenizer.decode(tt)
        if mode == "mt":
            hyp, _ = translate(model, tokenizer, spec, sl, tl, src_text, text_only=True,
                               max_new_tokens=220)
            codec, tc = [], None
        else:
            hyp, codec = translate(model, tokenizer, spec, sl, tl, src_text, sc,
                                   max_new_tokens=700)
        r = res[f"{sl}->{tl}"]
        r["hyp"].append(hyp)
        r["ref"].append(ref)
        if tc is not None:
            r["has_codec"] += int(len(codec) > 0)
            r["ratio"].append(len(codec) / max(len(tc), 1))
        if i < 3:
            print(f"[{tag} sample {sl}->{tl}]\n  src: {src_text}\n  ref: {ref}\n  hyp: {hyp}")
    for d, r in res.items():
        n_d = len(r["hyp"])
        msg = f"[{tag} gen-eval] {d}: n={n_d}"
        if r["ratio"]:
            msg += f" codec_emitted={r['has_codec']}/{n_d} mean_len_ratio={sum(r['ratio'])/len(r['ratio']):.2f}"
        if corpus_chrf:
            msg += f" chrF={corpus_chrf(r['hyp'], [r['ref']]).score:.1f}"
        print(msg)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--train_files", nargs="+", required=True)
    ap.add_argument("--eval_file", default=None)
    ap.add_argument("--mt_files", nargs="*", default=[])
    ap.add_argument("--mt_max_examples", type=int, default=0, help="cap MT rows (0 = all)")
    ap.add_argument("--mt_add_speech_text", action="store_true",
                    help="stage 1 also uses the text pairs of the speech rows (in-domain sentences)")
    ap.add_argument("--mt_fraction", type=float, default=0.25,
                    help="stage 2: fraction of examples that are text-only MT (replay)")
    ap.add_argument("--drop_src_codec_p", type=float, default=0.15)
    ap.add_argument("--no_reverse", action="store_true")
    ap.add_argument("--max_codec_len", type=int, default=400)
    ap.add_argument("--max_text_len", type=int, default=160)
    ap.add_argument("--limit_rows", type=int, default=0, help="read only N rows per file (smoke test)")
    # model / stages
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--init_from", default=None,
                    help="start from a saved model dir (e.g. old stage0, or <out>/stage_mt)")
    ap.add_argument("--stages", choices=["both", "mt", "s2st"], default="both")
    ap.add_argument("--base_dtype", choices=["fp32", "bf16"], default="fp32")
    # stage 1 (MT)
    ap.add_argument("--mt_mode", choices=["lora", "full"], default="lora")
    ap.add_argument("--mt_epochs", type=float, default=1)
    ap.add_argument("--mt_lr", type=float, default=None, help="default 1e-4 (lora) / 1e-5 (full)")
    ap.add_argument("--mt_batch_size", type=int, default=16)
    ap.add_argument("--mt_grad_accum", type=int, default=2)
    # stage 2 (S2ST)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--emb_lr_mult", type=float, default=10.0,
                    help="LR multiplier for the new codec rows in stage 2")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--resume_trainables", default=None,
                    help="trainables_s2st_*.pt from an interrupted stage 2 (weights only)")
    # lora
    ap.add_argument("--lora_r", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=256)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    # loss
    ap.add_argument("--text_weight", type=float, default=5.0)
    ap.add_argument("--codec_weight", type=float, default=1.0)
    ap.add_argument("--special_weight", type=float, default=3.0)
    ap.add_argument("--loss_chunk", type=int, default=512)
    # misc
    ap.add_argument("--optim", default="adamw_torch",
                    help="e.g. adamw_bnb_8bit (needs bitsandbytes) to save optimizer memory")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--save_steps", type=int, default=1000, help="trainables snapshot interval")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--no_bf16", dest="bf16", action="store_false", default=True)
    ap.add_argument("--gen_eval_n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = torch.bfloat16 if args.base_dtype == "bf16" else torch.float32
    reverse = not args.no_reverse

    # ---- model ----
    if args.init_from:
        model = _load_lm(args.init_from, dtype)
        tokenizer = AutoTokenizer.from_pretrained(args.init_from)
        spec = VocabSpec.load(args.init_from)
    else:
        model, tokenizer, spec = build_fresh(args.base_model, dtype)
    n_old = spec.codec_offset
    split_embeddings(model, n_old)
    spec.save(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"total_vocab={spec.total_vocab} codec_offset={n_old} stages={args.stages}")

    # ---- data ----
    kw = dict(include_reverse=reverse, max_text_len=args.max_text_len, limit=args.limit_rows)
    speech_tr, derived_tr = [], []
    if args.stages != "mt" or args.mt_add_speech_text or not args.mt_files:
        speech_tr, derived_tr = load_speech(args.train_files, tokenizer,
                                            max_codec_len=args.max_codec_len, **kw)
    mt_pool = load_mt(args.mt_files, tokenizer, max_examples=args.mt_max_examples, **kw) \
        if args.mt_files else []
    speech_ev, derived_ev = ([], [])
    if args.eval_file:
        speech_ev, derived_ev = load_speech([args.eval_file], tokenizer,
                                            max_codec_len=args.max_codec_len, **kw)

    # =================== stage 1: text-only MT ===================
    if args.stages in ("both", "mt"):
        items = list(mt_pool)
        if args.mt_add_speech_text or not items:
            if not mt_pool:
                print("[warn] no MT pairs from --mt_files: stage 1 uses text pairs of the speech rows")
            items += derived_tr
        if not items:
            raise SystemExit("stage 1 has no data (give --mt_files or --train_files)")
        random.shuffle(items)
        train_ds = ItemsDataset(items, spec)
        eval_ds = ItemsDataset(derived_ev, spec) if derived_ev else None
        print(f"\n===== STAGE 1 (MT, {args.mt_mode}): {len(train_ds)} examples =====")

        model = configure_mt(model, args)
        lr = args.mt_lr if args.mt_lr is not None else (1e-4 if args.mt_mode == "lora" else 1e-5)
        trainer = make_trainer(model, args, spec, train_ds, eval_ds, lr=lr, epochs=args.mt_epochs,
                               bs=args.mt_batch_size, ga=args.mt_grad_accum,
                               emb_lr_mult=1.0, tag="mt")
        trainer.train()
        model = merge_if_peft(finish_trainer(trainer))
        trainer = None  # drop the last reference so GPU memory is freed
        generation_eval(model, tokenizer, spec, derived_ev, args.gen_eval_n, "mt", "stage1")
        export_model(model, spec, tokenizer, os.path.join(args.output_dir, "stage_mt"), resplit=True)

    # =================== stage 2: speech translation ===================
    if args.stages in ("both", "s2st"):
        if not speech_tr:
            raise SystemExit("stage 2 has no speech data")
        pool = mt_pool or derived_tr
        if args.mt_fraction > 0 and not mt_pool:
            print("[warn] no --mt_files: replaying text pairs derived from the speech rows")
        n_mt = int(args.mt_fraction / (1.0 - args.mt_fraction) * len(speech_tr)) if args.mt_fraction > 0 else 0
        replay = []
        if n_mt and pool:
            p = list(pool)
            random.shuffle(p)
            replay = (p * math.ceil(n_mt / len(p)))[:n_mt]
        items = speech_tr + replay
        random.shuffle(items)
        train_ds = ItemsDataset(items, spec, drop_src_codec_p=args.drop_src_codec_p)
        eval_ds = ItemsDataset(speech_ev, spec) if speech_ev else None
        print(f"\n===== STAGE 2 (S2ST, LoRA): {len(speech_tr)} speech + {len(replay)} MT replay =====")

        model = configure_s2st(model, args)
        if args.resume_trainables:
            sd = torch.load(args.resume_trainables, map_location="cpu")
            res = model.load_state_dict(sd, strict=False)
            print(f"[resume] loaded {len(sd) - len(res.unexpected_keys)}/{len(sd)} tensors "
                  f"from {args.resume_trainables}")
        trainer = make_trainer(model, args, spec, train_ds, eval_ds, lr=args.lr, epochs=args.epochs,
                               bs=args.batch_size, ga=args.grad_accum,
                               emb_lr_mult=args.emb_lr_mult, tag="s2st")
        trainer.train()
        model = merge_if_peft(finish_trainer(trainer))
        trainer = None

        export_model(model, spec, tokenizer, args.output_dir, resplit=False)
        generation_eval(model, tokenizer, spec, speech_ev, args.gen_eval_n, "speech", "stage2")
        generation_eval(model, tokenizer, spec, derived_ev, args.gen_eval_n, "mt", "stage2-mt-retention")
        print(f"Done. Final model: {args.output_dir}")


if __name__ == "__main__":
    main()