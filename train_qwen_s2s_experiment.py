# # """
# # train_qwen_codec_s2s.py
# # ------------------------
# # Fine-tunes Qwen2.5-0.5B, repurposed as a discrete-token seq2seq model, to
# # map English MioCodec content tokens -> Hindi MioCodec content tokens.

# # Two vocab strategies, selectable with --vocab_mode, so you can run a
# # controlled comparison:

# #   extend (recommended default):
# #     Keep Qwen2.5-0.5B's full ~151,936-token text vocab untouched and
# #     append 12,800 codec ids + 6 control tokens on top. Embedding table
# #     grows from ~136M to ~148M params (~+2.3% of the 0.5B total) -- cheap,
# #     because embedding gradients are sparse: pretrained text rows never
# #     looked up in a codec-only batch simply never get updated. You keep
# #     the model closer to "actually Qwen2.5-0.5B" and leave room to later
# #     mix in text conditioning without another resize.

# #   shrink:
# #     Replace the embedding + LM head with a small ~12,806-row table and
# #     reinitialize it. Drops the unused ~124M text-embedding params
# #     (model shrinks to ~369M, ~25% smaller) at the cost of literally
# #     discarding the pretrained embedding/head weights (the transformer
# #     blocks in between are still pretrained and reused either way).

# # Sequence layout (single decoder-only stream, loss masked on the source):
# #   [BOS] [LANG_EN] en_tok_1 ... en_tok_N [SEP] [LANG_HI] hi_tok_1 ... hi_tok_M [EOS]
# #   labels:  -100     -100     -100 ...  -100   -100     -100     hi_tok_1 ... hi_tok_M [EOS]

# # Input file format (one line = one utterance pair), produced by
# # extract_codec_tokens.py:
# #   {"id": "...", "en_tokens": [...], "hi_tokens": [...]}

# # A vocab_spec.json is written next to the checkpoint recording exactly
# # which ids mean what (codec_offset, special ids, mode) -- inference code
# # should load that rather than hardcoding offsets, since they differ
# # between the two modes.

# # Usage:
# #   python train_qwen_codec_s2s.py \
# #       --train_files data/mahendra_train.jsonl data/sakthi_train.jsonl \
# #       --eval_file data/mahendra_dev.jsonl \
# #       --output_dir runs/qwen05b_en2hi_codec_extend \
# #       --vocab_mode extend \
# #       --base_model Qwen/Qwen2.5-0.5B \
# #       --epochs 10 --batch_size 32 --grad_accum 2

# #   # same data, shrink mode, for the comparison:
# #   python train_qwen_codec_s2s.py ... --output_dir runs/..._shrink --vocab_mode shrink
# # """
# # import argparse
# # import inspect
# # import json
# # import os
# # import random
# # from dataclasses import dataclass, asdict

# # import torch
# # import torch.nn as nn
# # from torch.utils.data import Dataset
# # from transformers import (
# #     AutoModelForCausalLM,
# #     AutoConfig,
# #     Trainer,
# #     TrainingArguments,
# # )

# # CODEC_VOCAB_SIZE = 12800  # MioCodec single-codebook vocab
# # N_SPECIAL = 6  # PAD, BOS, EOS, SEP, LANG_EN, LANG_HI


# # @dataclass
# # class VocabSpec:
# #     mode: str
# #     codec_offset: int  # add this to raw MioCodec ids [0,12799] to get model ids
# #     pad: int
# #     bos: int
# #     eos: int
# #     sep: int
# #     lang_en: int
# #     lang_hi: int
# #     total_vocab: int

# #     def save(self, output_dir: str):
# #         with open(os.path.join(output_dir, "vocab_spec.json"), "w") as f:
# #             json.dump(asdict(self), f, indent=2)

# #     @staticmethod
# #     def load(output_dir: str) -> "VocabSpec":
# #         with open(os.path.join(output_dir, "vocab_spec.json")) as f:
# #             return VocabSpec(**json.load(f))


# # def make_vocab_spec(mode: str, base_vocab_size: int) -> VocabSpec:
# #     if mode == "shrink":
# #         codec_offset = 0
# #         special_start = CODEC_VOCAB_SIZE
# #     elif mode == "extend":
# #         codec_offset = base_vocab_size
# #         special_start = base_vocab_size + CODEC_VOCAB_SIZE
# #     else:
# #         raise ValueError(f"unknown vocab_mode {mode}")
# #     pad, bos, eos, sep, lang_en, lang_hi = range(special_start, special_start + N_SPECIAL)
# #     return VocabSpec(
# #         mode=mode, codec_offset=codec_offset, pad=pad, bos=bos, eos=eos,
# #         sep=sep, lang_en=lang_en, lang_hi=lang_hi,
# #         total_vocab=special_start + N_SPECIAL,
# #     )


# # def build_model(base_model_name: str, mode: str):
# #     config = AutoConfig.from_pretrained(base_model_name)
# #     model = AutoModelForCausalLM.from_pretrained(base_model_name)
# #     base_vocab_size = model.get_input_embeddings().weight.shape[0]
# #     spec = make_vocab_spec(mode, base_vocab_size)

# #     # resize_token_embeddings truncates/extends at the END of the table and
# #     # random-inits any newly added rows for us; it never touches rows that
# #     # survive the resize.
# #     model.resize_token_embeddings(spec.total_vocab)

# #     if mode == "shrink":
# #         # every surviving row is a leftover text-token embedding with no
# #         # relationship to our codec ids -> reinit the whole table cleanly.
# #         std = getattr(config, "initializer_range", 0.02)
# #         nn.init.normal_(model.get_input_embeddings().weight, mean=0.0, std=std)
# #         out_emb = model.get_output_embeddings()
# #         if out_emb is not None and not model.config.tie_word_embeddings:
# #             nn.init.normal_(out_emb.weight, mean=0.0, std=std)
# #     # in "extend" mode we deliberately leave rows [0, base_vocab_size) alone.

# #     model.config.vocab_size = spec.total_vocab
# #     model.config.pad_token_id = spec.pad
# #     model.config.bos_token_id = spec.bos
# #     model.config.eos_token_id = spec.eos
# #     return model, spec


# # # --------------------------------------------------------------------------
# # # Dataset
# # # --------------------------------------------------------------------------
# # class CodecPairDataset(Dataset):
# #     def __init__(self, paths, spec: VocabSpec, max_src_len=400, max_tgt_len=400):
# #         self.spec = spec
# #         self.examples = []
# #         for path in paths:
# #             with open(path) as f:
# #                 for line in f:
# #                     line = line.strip()
# #                     if not line:
# #                         continue
# #                     rec = json.loads(line)
# #                     en = rec["en_tokens"][:max_src_len]
# #                     hi = rec["hi_tokens"][:max_tgt_len]
# #                     if len(en) == 0 or len(hi) == 0:
# #                         continue
# #                     self.examples.append((en, hi))
# #         random.shuffle(self.examples)
# #         print(f"Loaded {len(self.examples)} pairs from {paths} (vocab_mode={spec.mode})")

# #     def __len__(self):
# #         return len(self.examples)

# #     def __getitem__(self, idx):
# #         spec = self.spec
# #         off = spec.codec_offset
# #         en_raw, hi_raw = self.examples[idx]
# #         en = [t + off for t in en_raw]
# #         hi = [t + off for t in hi_raw]

# #         input_ids = [spec.bos, spec.lang_en] + en + [spec.sep, spec.lang_hi] + hi + [spec.eos]
# #         prefix_len = 2 + len(en) + 2  # BOS,LANG_EN,...,SEP,LANG_HI
# #         labels = [-100] * prefix_len + hi + [spec.eos]
# #         assert len(input_ids) == len(labels)
# #         return {"input_ids": input_ids, "labels": labels}


# # @dataclass
# # class PadCollator:
# #     pad_id: int

# #     def __call__(self, batch):
# #         max_len = max(len(ex["input_ids"]) for ex in batch)
# #         input_ids, labels, attn = [], [], []
# #         for ex in batch:
# #             ids, lab = ex["input_ids"], ex["labels"]
# #             pad_n = max_len - len(ids)
# #             input_ids.append(ids + [self.pad_id] * pad_n)
# #             labels.append(lab + [-100] * pad_n)
# #             attn.append([1] * len(ids) + [0] * pad_n)
# #         return {
# #             "input_ids": torch.tensor(input_ids, dtype=torch.long),
# #             "labels": torch.tensor(labels, dtype=torch.long),
# #             "attention_mask": torch.tensor(attn, dtype=torch.long),
# #         }


# # # --------------------------------------------------------------------------
# # # Inference helper (greedy AR generation, for sanity-checking a checkpoint).
# # # Always takes/returns RAW MioCodec ids in [0, 12799] regardless of
# # # vocab_mode -- offset bookkeeping is handled internally.
# # # --------------------------------------------------------------------------
# # @torch.inference_mode()
# # def translate_en_codec_to_hi_codec(model, spec: VocabSpec, en_tokens_raw, device="cuda", max_new_tokens=400):
# #     model.eval()
# #     off = spec.codec_offset
# #     en = [t + off for t in en_tokens_raw]
# #     prompt = torch.tensor([[spec.bos, spec.lang_en] + en + [spec.sep, spec.lang_hi]], device=device)
# #     out = model.generate(
# #         prompt, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
# #         eos_token_id=spec.eos, pad_token_id=spec.pad,
# #     )
# #     generated = out[0, prompt.shape[1]:].tolist()
# #     if spec.eos in generated:
# #         generated = generated[: generated.index(spec.eos)]
# #     return [t - off for t in generated]  # back to raw MioCodec ids for the decoder


# # # --------------------------------------------------------------------------
# # # transformers version drift is a real hazard here (warmup_ratio was
# # # missing entirely in some installs; eval_strategy vs evaluation_strategy
# # # renamed across versions). Rather than hardcode one set of kwarg names,
# # # introspect whatever TrainingArguments is actually installed and only
# # # pass what it supports, so this script keeps working across versions.
# # # --------------------------------------------------------------------------
# # def build_training_args(args, do_eval: bool) -> TrainingArguments:
# #     strategy = "epoch" if do_eval else "no"
# #     desired = dict(
# #         output_dir=args.output_dir,
# #         per_device_train_batch_size=args.batch_size,
# #         per_device_eval_batch_size=args.batch_size,
# #         gradient_accumulation_steps=args.grad_accum,
# #         num_train_epochs=args.epochs,
# #         learning_rate=args.lr,
# #         lr_scheduler_type="cosine",
# #         warmup_ratio=args.warmup_ratio,
# #         bf16=args.bf16,
# #         logging_steps=20,
# #         save_strategy="epoch",
# #         save_total_limit=3,
# #         report_to=[],
# #         remove_unused_columns=False,
# #         # both spellings offered; whichever this transformers version
# #         # doesn't recognize gets filtered out below
# #         eval_strategy=strategy,
# #         evaluation_strategy=strategy,
# #         load_best_model_at_end=do_eval,
# #         metric_for_best_model="eval_loss" if do_eval else None,
# #         greater_is_better=False if do_eval else None,
# #     )
# #     desired = {k: v for k, v in desired.items() if v is not None}

# #     valid_params = set(inspect.signature(TrainingArguments.__init__).parameters)
# #     filtered = {k: v for k, v in desired.items() if k in valid_params}
# #     dropped = sorted(set(desired) - set(filtered))
# #     if dropped:
# #         print(f"[compat] this transformers version doesn't support {dropped} -- skipping them "
# #               f"(check `pip show transformers`; consider `pip install -U transformers` if that matters to you).")
# #     if not do_eval:
# #         # load_best_model_at_end needs matching save/eval strategies; we
# #         # already excluded it via do_eval=False above, this is just a guard
# #         filtered.pop("load_best_model_at_end", None)
# #     return TrainingArguments(**filtered)


# # # --------------------------------------------------------------------------
# # # Main
# # # --------------------------------------------------------------------------
# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--train_files", nargs="+", required=True)
# #     ap.add_argument("--eval_file", default=None)
# #     ap.add_argument("--output_dir", required=True)
# #     ap.add_argument("--vocab_mode", choices=["extend", "shrink"], default="extend")
# #     ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B")
# #     ap.add_argument("--epochs", type=float, default=4)
# #     ap.add_argument("--batch_size", type=int, default=32)
# #     ap.add_argument("--grad_accum", type=int, default=2)
# #     ap.add_argument("--lr", type=float, default=2e-4)
# #     ap.add_argument("--max_src_len", type=int, default=400)
# #     ap.add_argument("--max_tgt_len", type=int, default=400)
# #     ap.add_argument("--warmup_ratio", type=float, default=0.03)
# #     ap.add_argument("--bf16", action="store_true", default=True)
# #     args = ap.parse_args()

# #     os.makedirs(args.output_dir, exist_ok=True)
# #     model, spec = build_model(args.base_model, args.vocab_mode)
# #     spec.save(args.output_dir)
# #     print(f"vocab_mode={spec.mode} total_vocab={spec.total_vocab} codec_offset={spec.codec_offset}")
# #     print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# #     train_ds = CodecPairDataset(args.train_files, spec, args.max_src_len, args.max_tgt_len)
# #     eval_ds = (
# #         CodecPairDataset([args.eval_file], spec, args.max_src_len, args.max_tgt_len)
# #         if args.eval_file else None
# #     )

# #     training_args = build_training_args(args, do_eval=eval_ds is not None)

# #     trainer = Trainer(
# #         model=model,
# #         args=training_args,
# #         train_dataset=train_ds,
# #         eval_dataset=eval_ds,
# #         data_collator=PadCollator(pad_id=spec.pad),
# #     )
# #     trainer.train()
# #     trainer.save_model(args.output_dir)
# #     spec.save(args.output_dir)  # re-save alongside final checkpoint
# #     print(f"Saved to {args.output_dir}")

# #     if len(train_ds) > 0:
# #         en_raw, hi_raw = train_ds.examples[0]
# #         pred = translate_en_codec_to_hi_codec(model, spec, en_raw, device=model.device)
# #         print(f"[sanity check] en_len={len(en_raw)} true_hi_len={len(hi_raw)} pred_hi_len={len(pred)}")


# # if __name__ == "__main__":
# #     main()

# # # !python train_qwen_codec_s2s.py \
# # # --train_files ./data/mahendra_train.jsonl \
# # # --eval_file ./data/mahendra_validation.jsonl \
# # # --output_dir runs/en2hi_extend --vocab_mode extend \
# # # --epochs 1 --batch_size 4 --grad_accum 4 --lr 2e-4

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

"""
train_qwen_codec_s2s.py
------------------------
Fine-tunes Qwen2.5-0.5B, repurposed as a discrete-token seq2seq model, to
translate speech between English and Hindi via MioCodec content tokens.

Two training modes:

  CODEC-ONLY (original): en_codec -> hi_codec directly. Works with either
    --vocab_mode shrink or extend. No text needed in the jsonl.

  TEXT-BRIDGE (new, requires --vocab_mode extend): uses the real Qwen
    tokenizer + Qwen's pretrained text vocab (never touched in shrink
    mode, which is why this requires extend) to train:
        src_text + src_codec  ->  tgt_text + tgt_codec
    i.e. the model generates the target language's TEXT first, then its
    codec tokens, conditioned on that text. This gives the model a
    linguistic "thought" step it can lean on Qwen's pretrained knowledge
    for, rather than mapping codec ids to codec ids blind -- the latter
    is a much harder problem to learn from ~100k pairs with randomly
    initialized codec embeddings, which is likely why your shrink/extend
    codec-only runs underperformed.

    Enabled automatically when --vocab_mode extend and the jsonl has
    "en_text"/"hi_text" fields; disable with --no_text_bridge to fall back
    to codec-only even in extend mode.

Reverse pairs: --include_reverse (on by default) also trains hi->en from
the same rows, roughly doubling effective data. Works in both codec-only
and text-bridge modes; uses the same LANG_EN/LANG_HI tags, just swapped.

Sequence layout:

  Text-bridge:
    [BOS][LANG_SRC] src_text [T2C] src_codec [SEP][LANG_TGT] tgt_text [T2C] tgt_codec [EOS]
    labels: -100 through [SEP][LANG_TGT], then real ids for tgt_text [T2C] tgt_codec [EOS]

  Codec-only (unchanged from before):
    [BOS][LANG_SRC] src_codec [SEP][LANG_TGT] tgt_codec [EOS]
    labels: -100 through [SEP][LANG_TGT], then real ids for tgt_codec [EOS]

Input file format (one line = one utterance pair), produced by
extract_codec_tokens.py, optionally with en_text/hi_text added:
  {"id": "...", "en_tokens": [...], "hi_tokens": [...],
   "en_text": "...", "hi_text": "..."}   <- text fields optional

A vocab_spec.json is written next to the checkpoint recording exactly
which ids mean what and whether text-bridge was used -- inference code
should load that rather than hardcoding offsets/mode.

Usage:
  # text-bridge (recommended given your codec-only results), with reverse pairs:
  python train_qwen_codec_s2s.py \
      --train_files data/mahendra_train_with_text.jsonl \
      --eval_file data/mahendra_dev_with_text.jsonl \
      --output_dir runs/qwen05b_textbridge_extend \
      --vocab_mode extend \
      --base_model Qwen/Qwen2.5-0.5B \
      --epochs 10 --batch_size 16 --grad_accum 4

  # codec-only, no text, no reverse (old behavior):
  python train_qwen_codec_s2s.py \
      --train_files data/mahendra_train.jsonl --eval_file data/mahendra_dev.jsonl \
      --output_dir runs/qwen05b_codec_only --vocab_mode shrink \
      --no_text_bridge --no_reverse
"""
import argparse
import inspect
import json
import os
import random
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
)

CODEC_VOCAB_SIZE = 12800  # MioCodec single-codebook vocab
N_SPECIAL = 7  # PAD, BOS, EOS, SEP, T2C, LANG_EN, LANG_HI


@dataclass
class VocabSpec:
    mode: str
    codec_offset: int  # add this to raw MioCodec ids [0,12799] to get model ids
    pad: int
    bos: int
    eos: int
    sep: int
    lang_en: int
    lang_hi: int
    total_vocab: int
    # added for text-bridge; default so OLD vocab_spec.json files (saved
    # before this feature existed) still load without error
    t2c: int = -1
    text_bridge: bool = False

    def save(self, output_dir: str):
        with open(os.path.join(output_dir, "vocab_spec.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(output_dir: str) -> "VocabSpec":
        with open(os.path.join(output_dir, "vocab_spec.json")) as f:
            return VocabSpec(**json.load(f))


def make_vocab_spec(mode: str, base_vocab_size: int, text_bridge: bool) -> VocabSpec:
    if mode == "shrink":
        codec_offset = 0
        special_start = CODEC_VOCAB_SIZE
    elif mode == "extend":
        codec_offset = base_vocab_size
        special_start = base_vocab_size + CODEC_VOCAB_SIZE
    else:
        raise ValueError(f"unknown vocab_mode {mode}")
    pad, bos, eos, sep, t2c, lang_en, lang_hi = range(special_start, special_start + N_SPECIAL)
    return VocabSpec(
        mode=mode, codec_offset=codec_offset, pad=pad, bos=bos, eos=eos,
        sep=sep, lang_en=lang_en, lang_hi=lang_hi,
        total_vocab=special_start + N_SPECIAL,
        t2c=t2c, text_bridge=text_bridge,
    )


def build_model(base_model_name: str, mode: str, text_bridge: bool):
    config = AutoConfig.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_vocab_size = model.get_input_embeddings().weight.shape[0]
    spec = make_vocab_spec(mode, base_vocab_size, text_bridge)

    model.resize_token_embeddings(spec.total_vocab)

    if mode == "shrink":
        std = getattr(config, "initializer_range", 0.02)
        nn.init.normal_(model.get_input_embeddings().weight, mean=0.0, std=std)
        out_emb = model.get_output_embeddings()
        if out_emb is not None and not model.config.tie_word_embeddings:
            nn.init.normal_(out_emb.weight, mean=0.0, std=std)
    # in "extend" mode rows [0, base_vocab_size) are Qwen's pretrained text
    # embeddings and are deliberately left untouched -- that's the whole
    # point of text-bridge mode.

    model.config.vocab_size = spec.total_vocab
    model.config.pad_token_id = spec.pad
    model.config.bos_token_id = spec.bos
    model.config.eos_token_id = spec.eos
    return model, tokenizer, spec


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class CodecPairDataset(Dataset):
    def __init__(self, paths, spec: VocabSpec, tokenizer=None, include_reverse=True,
                 max_src_len=400, max_tgt_len=400, max_text_len=64):
        self.spec = spec
        self.tokenizer = tokenizer
        self.examples = []  # each: (src_lang, src_text_ids, src_codec_raw, tgt_lang, tgt_text_ids, tgt_codec_raw)

        if spec.text_bridge and tokenizer is None:
            raise ValueError("text_bridge=True requires a tokenizer")

        n_rows = 0
        for path in paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    en_codec = rec["en_tokens"][:max_src_len]
                    hi_codec = rec["hi_tokens"][:max_tgt_len]
                    if len(en_codec) == 0 or len(hi_codec) == 0:
                        continue

                    if spec.text_bridge:
                        if "en_text" not in rec or "hi_text" not in rec:
                            raise ValueError(
                                f"text_bridge=True but record {rec.get('id')} in {path} is missing "
                                f"en_text/hi_text. Re-run without --no_text_bridge disabled, or use a "
                                f"jsonl that has both text fields (e.g. mahendra_train_with_text.jsonl)."
                            )
                        en_text_ids = tokenizer.encode(rec["en_text"], add_special_tokens=False)[:max_text_len]
                        hi_text_ids = tokenizer.encode(rec["hi_text"], add_special_tokens=False)[:max_text_len]
                    else:
                        en_text_ids, hi_text_ids = [], []

                    self.examples.append(("en", en_text_ids, en_codec, "hi", hi_text_ids, hi_codec))
                    if include_reverse:
                        self.examples.append(("hi", hi_text_ids, hi_codec, "en", en_text_ids, en_codec))
                    n_rows += 1

        random.shuffle(self.examples)
        print(f"Loaded {n_rows} source rows -> {len(self.examples)} training examples "
              f"from {paths} (vocab_mode={spec.mode}, text_bridge={spec.text_bridge}, "
              f"reverse={include_reverse})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        spec = self.spec
        off = spec.codec_offset
        src_lang, src_text_ids, src_codec_raw, tgt_lang, tgt_text_ids, tgt_codec_raw = self.examples[idx]

        src_lang_tok = spec.lang_en if src_lang == "en" else spec.lang_hi
        tgt_lang_tok = spec.lang_en if tgt_lang == "en" else spec.lang_hi
        src_codec = [t + off for t in src_codec_raw]
        tgt_codec = [t + off for t in tgt_codec_raw]

        if spec.text_bridge:
            src_block = [spec.bos, src_lang_tok] + src_text_ids + [spec.t2c] + src_codec
            tgt_block = tgt_text_ids + [spec.t2c] + tgt_codec
        else:
            src_block = [spec.bos, src_lang_tok] + src_codec
            tgt_block = tgt_codec

        input_ids = src_block + [spec.sep, tgt_lang_tok] + tgt_block + [spec.eos]
        prefix_len = len(src_block) + 2  # + [SEP, tgt_lang_tok]
        labels = [-100] * prefix_len + tgt_block + [spec.eos]
        assert len(input_ids) == len(labels)
        return {"input_ids": input_ids, "labels": labels}


# --------------------------------------------------------------------------
# Text-only MT data (no audio). Used to inject translation knowledge that
# 115k speech pairs alone can't provide.
#
# Your speech transcripts are normalized (lowercase English, no apostrophes
# or punctuation: "weve", "havent"). External MT corpora are not, so we
# normalize them the same way or the model sees a train/test mismatch.
#
# NOTE: we strip an explicit punctuation set instead of using regex \w,
# because Python's \w would also delete Devanagari vowel signs (matras),
# which are combining marks, and destroy the Hindi text.
# --------------------------------------------------------------------------
_PUNCT = set(".,?!;:\"'`()[]{}<>-–—…।॥|/\\*_#@$%^&+=~“”‘’«»")


def normalize_text(text: str, lang: str) -> str:
    text = "".join(ch for ch in text if ch not in _PUNCT)
    if lang == "en":
        text = text.lower()
    return " ".join(text.split())


class MTTextDataset(Dataset):
    """Text-only pairs: [BOS][LANG_SRC] src_text [SEP][LANG_TGT] tgt_text [EOS]
    (no [T2C], no codec span). Loss only on the target text + EOS.
    Both directions are added if include_reverse."""

    def __init__(self, paths, spec: VocabSpec, tokenizer, include_reverse=True,
                 max_text_len=64, max_examples=None, seed=0):
        self.spec = spec
        self.examples = []
        rng = random.Random(seed)
        for path in paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    en = normalize_text(rec["en_text"], "en")
                    hi = normalize_text(rec["hi_text"], "hi")
                    if not en or not hi:
                        continue
                    en_ids = tokenizer.encode(en, add_special_tokens=False)
                    hi_ids = tokenizer.encode(hi, add_special_tokens=False)
                    # skip over-long pairs instead of truncating (a truncated
                    # translation teaches the model to stop mid-sentence)
                    if len(en_ids) > max_text_len or len(hi_ids) > max_text_len:
                        continue
                    self.examples.append(("en", en_ids, "hi", hi_ids))
                    if include_reverse:
                        self.examples.append(("hi", hi_ids, "en", en_ids))
        rng.shuffle(self.examples)
        if max_examples is not None:
            self.examples = self.examples[:max_examples]
        print(f"MT text dataset: {len(self.examples)} examples from {paths}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        spec = self.spec
        src_lang, src_ids, tgt_lang, tgt_ids = self.examples[idx]
        src_tok = spec.lang_en if src_lang == "en" else spec.lang_hi
        tgt_tok = spec.lang_en if tgt_lang == "en" else spec.lang_hi
        prefix = [spec.bos, src_tok] + src_ids + [spec.sep, tgt_tok]
        input_ids = prefix + tgt_ids + [spec.eos]
        labels = [-100] * len(prefix) + tgt_ids + [spec.eos]
        return {"input_ids": input_ids, "labels": labels}


@dataclass
class PadCollator:
    pad_id: int

    def __call__(self, batch):
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, labels, attn = [], [], []
        for ex in batch:
            ids, lab = ex["input_ids"], ex["labels"]
            pad_n = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_n)
            labels.append(lab + [-100] * pad_n)
            attn.append([1] * len(ids) + [0] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# --------------------------------------------------------------------------
# Inference helpers (greedy AR generation, for sanity-checking a checkpoint).
# Always take/return RAW MioCodec ids in [0, 12799] regardless of
# vocab_mode -- offset bookkeeping is handled internally.
# --------------------------------------------------------------------------
@torch.inference_mode()
def translate_codec_only(model, spec: VocabSpec, src_lang: str, src_codec_raw, tgt_lang: str,
                          device="cuda", max_new_tokens=400):
    """Codec-only inference: src_codec -> tgt_codec. Works for shrink or extend
    checkpoints trained with spec.text_bridge=False."""
    model.eval()
    off = spec.codec_offset
    src_lang_tok = spec.lang_en if src_lang == "en" else spec.lang_hi
    tgt_lang_tok = spec.lang_en if tgt_lang == "en" else spec.lang_hi
    src_codec = [t + off for t in src_codec_raw]
    prompt = torch.tensor([[spec.bos, src_lang_tok] + src_codec + [spec.sep, tgt_lang_tok]], device=device)
    out = model.generate(
        prompt, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        eos_token_id=spec.eos, pad_token_id=spec.pad,
    )
    generated = out[0, prompt.shape[1]:].tolist()
    if spec.eos in generated:
        generated = generated[: generated.index(spec.eos)]
    return [t - off for t in generated]


@torch.inference_mode()
def translate_with_text_bridge(model, tokenizer, spec: VocabSpec, src_lang: str, src_text: str,
                                src_codec_raw, tgt_lang: str, device="cuda", max_new_tokens=400):
    """Text-bridge inference: (src_text, src_codec) -> (tgt_text, tgt_codec).
    Requires src_text at inference time (e.g. from an ASR pass), matching
    how the model was trained. Returns (tgt_text: str, tgt_codec_raw: list[int])."""
    assert spec.text_bridge, "this checkpoint was not trained with text_bridge=True"
    model.eval()
    off = spec.codec_offset
    src_lang_tok = spec.lang_en if src_lang == "en" else spec.lang_hi
    tgt_lang_tok = spec.lang_en if tgt_lang == "en" else spec.lang_hi
    src_text_ids = tokenizer.encode(src_text, add_special_tokens=False)
    src_codec = [t + off for t in src_codec_raw]

    prompt_ids = [spec.bos, src_lang_tok] + src_text_ids + [spec.t2c] + src_codec + [spec.sep, tgt_lang_tok]
    prompt = torch.tensor([prompt_ids], device=device)
    out = model.generate(
        prompt, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        eos_token_id=spec.eos, pad_token_id=spec.pad,
    )
    generated = out[0, prompt.shape[1]:].tolist()
    if spec.eos in generated:
        generated = generated[: generated.index(spec.eos)]

    if spec.t2c in generated:
        split = generated.index(spec.t2c)
        tgt_text_ids, tgt_codec_ids = generated[:split], generated[split + 1:]
    else:
        # model failed to emit the T2C marker -- treat everything as text,
        # no codec tokens produced (caller should treat this as a failure case)
        tgt_text_ids, tgt_codec_ids = generated, []

    tgt_text = tokenizer.decode(tgt_text_ids, skip_special_tokens=True)
    tgt_codec_raw = [t - off for t in tgt_codec_ids]
    return tgt_text, tgt_codec_raw


@torch.inference_mode()
def generate_tgt_text(model, tokenizer, spec: VocabSpec, src_lang: str, src_text: str,
                       src_codec_raw, tgt_lang: str, num_beams: int = 4,
                       max_text_tokens: int = 64, device="cuda"):
    """PHASE 1 of two-phase decoding: generate only the target TEXT span,
    stopping at [T2C] (or EOS). Beam search is cheap here because the text
    span is short (~15-30 tokens), and it is where translation quality is
    decided. Returns (text_ids: list[int] in model space, prompt_ids)."""
    assert spec.text_bridge
    model.eval()
    off = spec.codec_offset
    src_lang_tok = spec.lang_en if src_lang == "en" else spec.lang_hi
    tgt_lang_tok = spec.lang_en if tgt_lang == "en" else spec.lang_hi
    src_text_ids = tokenizer.encode(src_text, add_special_tokens=False)
    src_codec = [t + off for t in src_codec_raw]
    prompt_ids = [spec.bos, src_lang_tok] + src_text_ids + [spec.t2c] + src_codec + [spec.sep, tgt_lang_tok]
    prompt = torch.tensor([prompt_ids], device=device)

    out = model.generate(
        prompt, max_new_tokens=max_text_tokens, do_sample=False,
        num_beams=num_beams, early_stopping=True if num_beams > 1 else False,
        eos_token_id=[spec.t2c, spec.eos], pad_token_id=spec.pad,
    )
    gen = out[0, prompt.shape[1]:].tolist()
    # cut at the first stop marker
    for i, t in enumerate(gen):
        if t in (spec.t2c, spec.eos, spec.pad):
            gen = gen[:i]
            break
    return gen, prompt_ids


@torch.inference_mode()
def translate_with_text_bridge_beam(model, tokenizer, spec: VocabSpec, src_lang: str, src_text: str,
                                     src_codec_raw, tgt_lang: str, num_beams: int = 4,
                                     max_text_tokens: int = 64, max_codec_tokens: int = 400,
                                     forced_tgt_text: str = None, device="cuda"):
    """Two-phase decode:
       phase 1: beam search for the target text (skipped if forced_tgt_text is
                given -- e.g. text from an external MT system like IndicTrans2)
       phase 2: greedy codec generation conditioned on that fixed text.
    Returns (tgt_text: str, tgt_codec_raw: list[int])."""
    assert spec.text_bridge
    off = spec.codec_offset

    if forced_tgt_text is not None:
        text_ids = tokenizer.encode(forced_tgt_text, add_special_tokens=False)
        src_lang_tok = spec.lang_en if src_lang == "en" else spec.lang_hi
        tgt_lang_tok = spec.lang_en if tgt_lang == "en" else spec.lang_hi
        src_text_ids = tokenizer.encode(src_text, add_special_tokens=False)
        src_codec = [t + off for t in src_codec_raw]
        prompt_ids = [spec.bos, src_lang_tok] + src_text_ids + [spec.t2c] + src_codec + [spec.sep, tgt_lang_tok]
    else:
        text_ids, prompt_ids = generate_tgt_text(
            model, tokenizer, spec, src_lang, src_text, src_codec_raw, tgt_lang,
            num_beams=num_beams, max_text_tokens=max_text_tokens, device=device,
        )

    phase2_prompt = torch.tensor([prompt_ids + text_ids + [spec.t2c]], device=device)
    out = model.generate(
        phase2_prompt, max_new_tokens=max_codec_tokens, do_sample=False, num_beams=1,
        eos_token_id=spec.eos, pad_token_id=spec.pad,
    )
    codec_ids = out[0, phase2_prompt.shape[1]:].tolist()
    if spec.eos in codec_ids:
        codec_ids = codec_ids[: codec_ids.index(spec.eos)]
    # drop anything that isn't a codec id (defensive; model should only emit codec ids here)
    codec_raw = [t - off for t in codec_ids if off <= t < off + CODEC_VOCAB_SIZE]

    tgt_text = tokenizer.decode(text_ids, skip_special_tokens=True)
    return tgt_text, codec_raw


# --------------------------------------------------------------------------
# transformers version drift compatibility layer (unchanged from before)
# --------------------------------------------------------------------------
def build_training_args(args, do_eval: bool) -> TrainingArguments:
    strategy = "epoch" if do_eval else "no"
    desired = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=3,
        report_to=[],
        remove_unused_columns=False,
        eval_strategy=strategy,
        evaluation_strategy=strategy,
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss" if do_eval else None,
        greater_is_better=False if do_eval else None,
    )
    desired = {k: v for k, v in desired.items() if v is not None}

    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    filtered = {k: v for k, v in desired.items() if k in valid_params}
    dropped = sorted(set(desired) - set(filtered))
    if dropped:
        print(f"[compat] this transformers version doesn't support {dropped} -- skipping them.")
    if not do_eval:
        filtered.pop("load_best_model_at_end", None)
    return TrainingArguments(**filtered)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_files", nargs="+", required=True)
    ap.add_argument("--eval_file", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--vocab_mode", choices=["extend", "shrink"], default="extend")
    ap.add_argument("--no_text_bridge", action="store_true",
                     help="disable text-bridge even in extend mode (fall back to codec-only)")
    ap.add_argument("--no_reverse", action="store_true",
                     help="disable automatic hi->en reverse pairs (default: reverse pairs included)")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_src_len", type=int, default=400)
    ap.add_argument("--max_tgt_len", type=int, default=400)
    ap.add_argument("--max_text_len", type=int, default=64)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--gradient_checkpointing", action="store_true",
                     help="trade compute for memory so you can raise --batch_size instead of "
                          "relying purely on --grad_accum; recomputes activations in the backward "
                          "pass instead of storing them")
    ap.add_argument("--init_from", default=None,
                     help="Continue training from an existing text-bridge checkpoint dir "
                          "(model + tokenizer + vocab_spec.json) instead of starting from "
                          "--base_model. Keeps the learned codec embeddings. Recommended "
                          "lower --lr (e.g. 1e-4) when continuing.")
    ap.add_argument("--text_mt_files", nargs="*", default=None,
                     help="Text-only parallel jsonl files with en_text/hi_text (no audio), "
                          "e.g. IITB / Samanantar exported to jsonl. Mixed into training.")
    ap.add_argument("--mt_ratio", type=float, default=1.0,
                     help="Number of text-only MT examples per speech example "
                          "(1.0 = roughly 50/50 mix). Keep > 0 speech examples in the mix so "
                          "the shared (tied) embeddings of the codec rows are not drifted away.")
    ap.add_argument("--max_mt_examples", type=int, default=None,
                     help="Hard cap on MT examples (after mt_ratio), if you have a huge corpus")
    args = ap.parse_args()

    include_reverse = not args.no_reverse
    os.makedirs(args.output_dir, exist_ok=True)

    if args.init_from:
        spec = VocabSpec.load(args.init_from)
        if not spec.text_bridge:
            raise ValueError(
                f"--init_from {args.init_from} was trained with text_bridge=False; "
                f"this continuation path needs a text-bridge checkpoint."
            )
        model = AutoModelForCausalLM.from_pretrained(args.init_from)
        tokenizer = AutoTokenizer.from_pretrained(args.init_from)
        model.config.use_cache = True
        print(f"Continuing from {args.init_from} (vocab_mode={spec.mode}, text_bridge=True)")
    else:
        text_bridge = (args.vocab_mode == "extend") and (not args.no_text_bridge)
        model, tokenizer, spec = build_model(args.base_model, args.vocab_mode, text_bridge)
    if args.text_mt_files and not spec.text_bridge:
        raise ValueError("--text_mt_files requires a text-bridge model (--vocab_mode extend, no --no_text_bridge)")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # required alongside grad checkpointing during training;
        # re-enabled automatically at inference since infer_offline.py/streaming_infer.py load a
        # fresh AutoModelForCausalLM.from_pretrained(checkpoint_dir), not this in-memory object
    spec.save(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"vocab_mode={spec.mode} text_bridge={spec.text_bridge} total_vocab={spec.total_vocab} "
          f"codec_offset={spec.codec_offset}")
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    train_ds = CodecPairDataset(
        args.train_files, spec, tokenizer=tokenizer if spec.text_bridge else None,
        include_reverse=include_reverse, max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len, max_text_len=args.max_text_len,
    )
    eval_ds = (
        CodecPairDataset(
            [args.eval_file], spec, tokenizer=tokenizer if spec.text_bridge else None,
            include_reverse=include_reverse, max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len, max_text_len=args.max_text_len,
        )
        if args.eval_file else None
    )

    if args.text_mt_files:
        n_mt = int(len(train_ds) * args.mt_ratio)
        if args.max_mt_examples is not None:
            n_mt = min(n_mt, args.max_mt_examples)
        mt_ds = MTTextDataset(
            args.text_mt_files, spec, tokenizer, include_reverse=include_reverse,
            max_text_len=args.max_text_len, max_examples=n_mt,
        )
        # Fixed mix for the whole run (speech examples + a one-time sample of MT
        # examples). Eval stays speech-only so eval_loss tracks the real task.
        train_ds_for_trainer = ConcatDataset([train_ds, mt_ds])
        print(f"Training mix: {len(train_ds)} speech + {len(mt_ds)} text-only MT "
              f"= {len(train_ds_for_trainer)} examples/epoch")
    else:
        train_ds_for_trainer = train_ds

    training_args = build_training_args(args, do_eval=eval_ds is not None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_for_trainer,
        eval_dataset=eval_ds,
        data_collator=PadCollator(pad_id=spec.pad),
    )
    trainer.train()
    # gradient checkpointing needs use_cache=False during training, but a saved
    # config with use_cache=False makes generate() recompute everything (very slow)
    model.config.use_cache = True
    trainer.save_model(args.output_dir)
    spec.save(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")

    if len(train_ds) > 0:
        src_lang, src_text_ids, src_codec_raw, tgt_lang, tgt_text_ids, tgt_codec_raw = train_ds.examples[0]
        if spec.text_bridge:
            src_text = tokenizer.decode(src_text_ids, skip_special_tokens=True)
            pred_text, pred_codec = translate_with_text_bridge(
                model, tokenizer, spec, src_lang, src_text, src_codec_raw, tgt_lang, device=model.device
            )
            true_text = tokenizer.decode(tgt_text_ids, skip_special_tokens=True)
            print(f"[sanity check] {src_lang}->{tgt_lang} true_text={true_text!r} pred_text={pred_text!r} "
                  f"true_codec_len={len(tgt_codec_raw)} pred_codec_len={len(pred_codec)}")
        else:
            pred = translate_codec_only(model, spec, src_lang, src_codec_raw, tgt_lang, device=model.device)
            print(f"[sanity check] {src_lang}->{tgt_lang} true_len={len(tgt_codec_raw)} pred_len={len(pred)}")


if __name__ == "__main__":
    main()