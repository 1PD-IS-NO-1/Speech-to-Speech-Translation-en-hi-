# Speech-to-Speech Translation with Qwen2.5 over MioCodec tokens (English ↔ Hindi)

Qwen2.5-0.5B is fine-tuned as a discrete-token sequence model. It reads the source
utterance as [MioCodec](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz) content tokens and
writes the translation as **target text followed by target speech tokens**. The speech tokens are
turned back into a waveform by the MioCodec decoder.

> **Important:** this is *not* a fully end-to-end model. At inference you provide the **source
> transcript** (for example from Whisper). The model uses it as extra input (see "Text bridge").

TODO: add a short audio demo (3–4 EN→HI and HI→EN examples) and your evaluation numbers.

---

## How it works

### Text bridge
The model first generates the target-language **text**, then the target **speech tokens conditioned on
that text**. This lets it use Qwen's pretrained linguistic knowledge instead of mapping speech tokens
to speech tokens blindly.

```
speech example (stage 2):
 [BOS][LANG_SRC] src_text [T2C] src_codec_tokens [SEP][LANG_TGT] tgt_text [T2C] tgt_codec_tokens [EOS]

text-only example (stage 1, and replayed in stage 2):
 [BOS][LANG_SRC] src_text [SEP][LANG_TGT] tgt_text [EOS]
```

`[T2C]` ("text to codec") marks where the text ends and the speech tokens begin. The loss is only on
the target side. Special ids and the speech-token offset are written to `vocab_spec.json`, which **must
stay next to the model files**.

### Inference pipeline
```
 source wav ──MioCodec.encode──► source speech tokens ─┐
 source transcript (ASR) ──────────────────────────────┤
                                                       ▼
                                  Qwen: phase 1 → target TEXT (beam search)
                                        phase 2 → target SPEECH tokens (greedy)
                                                       ▼
 target wav ◄──MioCodec.decode(tokens, voice embedding)┘
```
By default the output uses the *source speaker's* voice embedding; use `--reference_wav` to speak in
another voice.

### Two-stage training (why)
| Stage | Data | What is trained |
|---|---|---|
| **1 – MT** | text-only EN↔HI sentence pairs | LoRA (default) or full fine-tune on Qwen's weights, low LR. Merged into the base weights. |
| **2 – S2ST** | EN speech+text → HI speech+text (and reverse), plus replayed text-only pairs | A **new** LoRA on the stage-1 model, plus the new speech-token embedding rows (higher LR). |

What we learned the hard way (details in [`experiments/`](experiments/README.md)): training everything
at once with a high LR made the audio match its text perfectly but destroyed the *translation*;
teaching translation first and then adding speech with a small LoRA keeps both.

Other design choices in `qwen_s2st.py`:
* **Split embeddings:** Qwen's 151,936 text embedding rows are frozen (never touched); only the 12,807
  new speech/special rows are trained. This also cuts memory a lot, so training fits on an 8 GB GPU.
* **Weighted loss:** text tokens are weighted ×5 (`--text_weight`) so ~30 text tokens are not drowned
  out by ~300 speech tokens. Text-loss and speech-loss are logged separately.
* **Text-only replay** (`--mt_fraction`) and **source-speech dropout** (`--drop_src_codec_p`) keep
  translation alive and let the model also translate from text alone.

---

## Repository layout
```
qwen_s2st.py        training (stage 1 + stage 2), export, generation eval
infer_offline.py    speech-to-speech inference for one utterance
check_text_mt.py    quick text-only translation check of one or more checkpoints
fix_checkpoint.py   one-time fix for checkpoints that don't store lm_head (see Troubleshooting)
experiments/        history of earlier approaches and what we learned
requirements.txt
```

## Installation
```bash
python -m venv env && source env/bin/activate
pip install -r requirements.txt
# + install MioCodec (see requirements.txt)
```
Tested on Python 3.12 with a single 8 GB NVIDIA GPU. TODO: add your exact torch / transformers versions.

---

## Inference

### 1. Download the weights
Weights are hosted on the Hugging Face Hub (TODO: replace with your repo id):
```python
from huggingface_hub import snapshot_download
snapshot_download("YOUR_HF_USERNAME/YOUR_MODEL_REPO", local_dir="checkpoints/qwen05b_s2st")
```
The folder must contain `model.safetensors`, `config.json`, tokenizer files **and `vocab_spec.json`**.

### 2. Get the source transcript
The model needs the transcript of the source audio. Any ASR works, for example Whisper:
```python
import whisper
print(whisper.load_model("small").transcribe("english.wav", language="en")["text"])   # use "hi" for Hindi
```

### 3. Translate
```bash
# English -> Hindi
python infer_offline.py \
  --checkpoint_dir checkpoints/qwen05b_s2st \
  --direction en2hi \
  --input_wav english.wav \
  --src_text "and while we've made some advancements" \
  --output_wav hindi_out.wav

# Hindi -> English
python infer_offline.py \
  --checkpoint_dir checkpoints/qwen05b_s2st \
  --direction hi2en \
  --input_wav hindi.wav \
  --src_text "और हालाँकि हमने कुछ प्रगति की है" \
  --output_wav english_out.wav
```
The script prints the generated target text, then writes the waveform.

Useful options:

| Option | Meaning |
|---|---|
| `--num_beams 4` | beam width for the target-**text** phase (speech tokens are always greedy) |
| `--forced_tgt_text "..."` | skip text generation and speak this text (e.g. from a stronger MT system) |
| `--no_src_codec` | translate from `--src_text` only; the input wav is then used just for the voice |
| `--reference_wav ref.wav` | speak in the voice of another speaker |
| `--max_new_tokens 400` | max speech tokens (25 per second of audio → 400 ≈ 16 s) |
| `--max_text_tokens 200` | max target text tokens |
| `--fp32` | load the LM in fp32 instead of bf16 |

Check translation quality alone (no audio):
```bash
python check_text_mt.py checkpoints/qwen05b_s2st/stage_mt checkpoints/qwen05b_s2st
```

---

## Training

### Data format
Everything is JSON Lines (one object per line).

**Speech pairs** (`--train_files`, `--eval_file`): the *same sentence* spoken in both languages, as
MioCodec content tokens, plus both transcripts.
```json
{"id": "utt_0001",
 "en_tokens": [123, 4051, ...],
 "hi_tokens": [88, 9012, ...],
 "en_text": "and while we've made some advancements",
 "hi_text": "और हालाँकि हमने कुछ प्रगति की है"}
```
* tokens are the raw MioCodec ids (0–12799) from `codec.encode(wav).content_token_indices`;
* `*_text` is required (rows with empty text are skipped);
* rows longer than `--max_codec_len` (default 400 tokens ≈ 16 s) or `--max_text_len` (default 160 Qwen
  tokens) are skipped, not truncated.

**Text-only pairs** (`--mt_files`, strongly recommended): any parallel corpus. Accepted keys:
`{"en_text","hi_text"}`, `{"en","hi"}`, `{"english","hindi"}`, `{"src","tgt"}` or `{"translation": {"en","hi"}}`.

Reverse direction (HI→EN) is generated automatically from the same rows (`--no_reverse` disables it).

To create the speech-pair file from audio you need a small extraction script that runs MioCodec on each
wav (see the inference code in `infer_offline.py` for the exact encode call). TODO: add your
`extract_codec_tokens.py` to this repo and describe its input manifest.

### Run
```bash
python qwen_s2st.py \
  --train_files data/train.jsonl \
  --eval_file data/dev.jsonl \
  --mt_files data/mt_en_hi.jsonl \
  --output_dir checkpoints/qwen05b_s2st \
  --stages both \
  --mt_epochs 1 --epochs 3 \
  --mt_batch_size 16 --mt_grad_accum 2 \
  --batch_size 3 --grad_accum 12 --grad_ckpt
```
Quick smoke test first (a few minutes): add `--limit_rows 300 --epochs 1 --gen_eval_n 4`.

Outputs in `--output_dir`: the final merged model, `vocab_spec.json`, tokenizer files, `stage_mt/` (the
stage-1 model), and small `trainables_*.pt` snapshots (LoRA + new rows) for resuming stage 2 with
`--resume_trainables`.

Main knobs:

| Flag | Default | Notes |
|---|---|---|
| `--stages` | `both` | `mt` or `s2st` to run one stage (`--init_from <dir>` to continue from a saved model) |
| `--mt_mode` | `lora` | `full` adapts Hindi better but needs far more memory (try `--optim adamw_bnb_8bit`) |
| `--lr` / `--mt_lr` | 1e-4 / 1e-4 (lora) | 1e-5 for `--mt_mode full` |
| `--emb_lr_mult` | 10 | LR multiplier for the new speech-token rows in stage 2 |
| `--mt_fraction` | 0.25 | share of text-only pairs replayed in stage 2 (raise it if translation degrades) |
| `--text_weight` | 5 | loss weight of text tokens |
| `--lora_r` / `--lora_alpha` | 128 / 256 | lower them to save memory |
| `--mt_max_examples` | 0 (all) | cap the text-only corpus to shorten stage 1 |

Both stages print generation-eval at the end (chrF if `sacrebleu` is installed), including a
text-only check after stage 2 (`stage2-mt-retention`) that shows whether translation survived.

### Troubleshooting
**`lm_head.weight | MISSING` and garbage output.** Qwen2.5-0.5B ties its output layer to the embedding
table, and some `transformers` versions then load a *random* output layer. The current `qwen_s2st.py`
stores `lm_head.weight` explicitly (and prints `[verify] OK` after export). Checkpoints trained with
an older version can be repaired once:
```bash
python fix_checkpoint.py checkpoints/qwen05b_s2st checkpoints/qwen05b_s2st/stage_mt
```
`infer_offline.py` also handles old checkpoints automatically.

**CUDA out of memory.** Use `--grad_ckpt`, lower `--batch_size` (raise `--grad_accum`), lower
`--lora_r`, `--max_codec_len`, or set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

---

## Adding more languages

> **Current status:** the code supports exactly one pair, English ↔ Hindi. The language tags and the
> field names (`en_*`, `hi_*`) are hard-coded. Adding a language needs the data below **and** a code
> change (see "Code changes"). Nothing here has been tested on a third language yet.

### Data you need per new language pair
| Data | Used for | Required? |
|---|---|---|
| **Parallel speech pairs with transcripts**: the same sentence spoken in language A and language B, plus both transcripts | stage 2 | **Yes** |
| **Text-only parallel sentences** A↔B (millions are fine; the more in-domain the better) | stage 1 and the replay in stage 2 | Strongly recommended |
| A held-out dev set of speech pairs | evaluation | Recommended |

Details that matter:
* **Utterance-level alignment.** Each row = one sentence, spoken in both languages (roughly ≤ 16 s each).
* **Clean transcripts** for both sides. Empty or mismatched text hurts the text bridge directly.
* **Where to get speech pairs:** real parallel speech corpora (for example n-way parallel read speech
  such as FLEURS, or speech-to-speech translation corpora such as CVSS — check which languages each
  covers and their licenses), or **synthetic** pairs made by running a good TTS on translated text.
  Synthetic targets limit naturalness to the TTS quality.
* **We trained EN↔HI on about 230,000 speech pairs** (doubled by the automatic reverse direction). We have
  not measured how much less would work.
* **Text-only pairs:** any parallel corpus for the pair (our Hindi run used IndicTrans2-generated and
  IITB-style English–Hindi pairs).

### Check before you train
1. **Codec round-trip.** Encode and decode a few clips in the new language with MioCodec and listen. If
   the codec reconstructs the language poorly, the translation model cannot fix that.
2. **Tokenizer efficiency.** Qwen2.5 splits Hindi into many tokens, so we use `--max_text_len 160`. For
   other scripts check the token counts; very long text per sentence means a larger `--max_text_len`
   (more memory) or a different base model.
3. **Base-model quality.** Qwen2.5-0.5B is weak in Hindi, and a small model will be weak in other
   low-resource languages too. A larger base model (`--base_model Qwen/Qwen2.5-1.5B` or bigger) is the
   first thing to try if text quality, not audio, is the limit.

### Code changes required (not implemented yet)
* Replace the two language tags with a table of tags (one per language); the number of special tokens
  then depends on the number of languages, so `vocab_spec.json` and the model's vocabulary size change.
  Models trained with the current en/hi layout are not compatible.
* Use rows with explicit language fields, for example
  `{"src_lang","tgt_lang","src_tokens","tgt_tokens","src_text","tgt_text"}`, instead of `en_*`/`hi_*`.
* When adding a language to an already-trained model, **mix in the old language pairs and the text-only
  replay** so the existing languages are not forgotten.

---

## Limitations
* Needs the **source transcript** at inference (ASR is a separate step).
* Text quality is bounded by Qwen2.5-0.5B's Hindi; occasional mistranslations are expected, and
  translation can degrade slightly after speech training (watch `stage2-mt-retention`).
* Trained on utterances up to about 16 s; long audio must be segmented.
* Audio quality and voice depend on MioCodec.
* No formal speech-quality evaluation yet (TODO: ASR-BLEU / human ratings).

## License and data notes (TODO)
Choose a license for this code. Check and state the licenses of: Qwen2.5-0.5B, MioCodec, the speech
dataset, and the text corpora you used. Do not upload audio or datasets you are not allowed to
redistribute.

## Acknowledgements
Qwen2.5 (Alibaba), MioCodec (Aratako), and the datasets you used (TODO: cite).