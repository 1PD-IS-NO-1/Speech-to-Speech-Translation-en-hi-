"""
infer_offline.py
----------------
Full-utterance speech translation (EN <-> HI) with a checkpoint trained by
qwen_s2st.py (text-bridge format). Keep this file next to qwen_s2st.py.

Pipeline:
  wav(src) -> MioCodec.encode -> src content tokens
  phase 1: [BOS][LANG_SRC] src_text [T2C] src_codec [SEP][LANG_TGT]
              -> target TEXT (beam search, stops at [T2C])
  phase 2: ... + tgt_text [T2C] -> target CODEC tokens (greedy, stops at [EOS])
  tgt tokens -> MioCodec.decode(tokens, global_embedding) -> wav(tgt)

Voice: the output reuses the SOURCE speaker's global_embedding by default;
pass --reference_wav to use another speaker's voice.

--src_text is REQUIRED (the transcript of the source audio; run an ASR such
as Whisper first). Optional modes:
  --forced_tgt_text "..."  skip text generation and speak this text instead
                           (e.g. from an external MT system like IndicTrans2)
  --no_src_codec           translate from src_text alone (the model was
                           trained with source-codec dropout for this)

Usage:
  python infer_offline.py \
      --checkpoint_dir runs2/qwen05b_s2st_lora \
      --direction en2hi \
      --input_wav english.wav \
      --src_text "and while we've made some advancements" \
      --output_wav hindi_out.wav
"""
import argparse

import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from train_qwen_codec_sts import CODEC_VOCAB_SIZE, VocabSpec
except ImportError:
    from qwen_s2st import CODEC_VOCAB_SIZE, VocabSpec


def load_lm(path, dtype, device):
    """Load the checkpoint and force lm_head to share the embedding table.
    (The model was trained with tied embeddings; some transformers versions
    drop lm_head.weight on save and then load a RANDOM head.)"""
    try:
        lm = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        lm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    lm.lm_head.weight = lm.get_input_embeddings().weight
    print("[tie] lm_head now shares the embedding table "
          "(the 'lm_head.weight MISSING' warning above is expected and now harmless)")
    return lm.eval().to(device)


def to_codec_tensor(ids, like: torch.Tensor, device):
    t = torch.tensor(ids, dtype=like.dtype, device=device)
    if like.dim() == 2:
        t = t.unsqueeze(0)
    return t


@torch.inference_mode()
def translate_text_bridge(lm, tokenizer, spec: VocabSpec, src_lang, tgt_lang, src_text,
                          src_codec_raw=None, num_beams=4, max_text_tokens=200,
                          max_codec_tokens=400, forced_tgt_text=None, device="cuda"):
    """Returns (tgt_text, tgt_codec_raw). Text: beam search; codec: greedy."""
    lm.eval()
    off = spec.codec_offset
    lang = lambda l: spec.lang_en if l == "en" else spec.lang_hi

    prompt_ids = [spec.bos, lang(src_lang)] + tokenizer.encode(src_text, add_special_tokens=False)
    prompt_ids.append(spec.t2c)
    if src_codec_raw:
        prompt_ids += [t + off for t in src_codec_raw]
    prompt_ids += [spec.sep, lang(tgt_lang)]

    # ---- phase 1: target text ----
    if forced_tgt_text is not None:
        text_ids = tokenizer.encode(forced_tgt_text, add_special_tokens=False)
        tgt_text = forced_tgt_text
    else:
        prompt = torch.tensor([prompt_ids], device=device)
        out = lm.generate(
            prompt, attention_mask=torch.ones_like(prompt),
            max_new_tokens=max_text_tokens, num_beams=num_beams, do_sample=False,
            early_stopping=True, repetition_penalty=1.0,
            eos_token_id=[spec.t2c, spec.eos], pad_token_id=spec.pad,
        )
        gen = out[0, prompt.shape[1]:].tolist()
        for i, t in enumerate(gen):  # cut at the first marker
            if t in (spec.t2c, spec.eos):
                gen = gen[:i]
                break
        text_ids = [t for t in gen if t < min(len(tokenizer), off)]
        tgt_text = tokenizer.decode(text_ids, skip_special_tokens=True)

    # ---- phase 2: target codec, conditioned on that text ----
    prompt2 = torch.tensor([prompt_ids + text_ids + [spec.t2c]], device=device)
    out2 = lm.generate(
        prompt2, attention_mask=torch.ones_like(prompt2),
        max_new_tokens=max_codec_tokens, num_beams=1, do_sample=False,
        eos_token_id=spec.eos, pad_token_id=spec.pad,
    )
    gen2 = out2[0, prompt2.shape[1]:].tolist()
    if spec.eos in gen2:
        gen2 = gen2[: gen2.index(spec.eos)]
    codec = [t - off for t in gen2 if off <= t < off + CODEC_VOCAB_SIZE]
    return tgt_text, codec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--direction", choices=["en2hi", "hi2en"], default="en2hi")
    ap.add_argument("--input_wav", required=True, help="Source-language audio")
    ap.add_argument("--output_wav", required=True)
    ap.add_argument("--src_text", required=True, help="Transcript of the source audio")
    ap.add_argument("--reference_wav", default=None,
                    help="Use this speaker's voice for the output instead of the source speaker's")
    ap.add_argument("--codec_model", default="Aratako/MioCodec-25Hz-24kHz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=400, help="max CODEC tokens")
    ap.add_argument("--max_text_tokens", type=int, default=200, help="max target TEXT tokens")
    ap.add_argument("--num_beams", type=int, default=4, help="beam width for the text phase")
    ap.add_argument("--forced_tgt_text", default=None)
    ap.add_argument("--no_src_codec", action="store_true",
                    help="translate from src_text only (source audio is used just for the voice)")
    ap.add_argument("--fp32", action="store_true", help="load the LM in fp32 instead of bf16")
    args = ap.parse_args()

    src_lang, tgt_lang = ("en", "hi") if args.direction == "en2hi" else ("hi", "en")

    from miocodec import MioCodecModel, load_audio

    print("Loading MioCodec...")
    codec = MioCodecModel.from_pretrained(args.codec_model).eval().to(args.device)

    print("Loading translation model + tokenizer...")
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    lm = load_lm(args.checkpoint_dir, dtype, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    spec = VocabSpec.load(args.checkpoint_dir)
    print(f"direction={src_lang}->{tgt_lang} codec_offset={spec.codec_offset}")

    print(f"Encoding {args.input_wav} ...")
    with torch.inference_mode():
        waveform = load_audio(args.input_wav, sample_rate=codec.config.sample_rate).to(args.device)
        feats = codec.encode(waveform)
        src_tokens = feats.content_token_indices.reshape(-1).detach().to("cpu", dtype=torch.long).tolist()

        if args.reference_wav:
            print(f"Using voice from {args.reference_wav}")
            ref_wave = load_audio(args.reference_wav, sample_rate=codec.config.sample_rate).to(args.device)
            global_embedding = codec.encode(ref_wave).global_embedding
        else:
            global_embedding = feats.global_embedding  # preserve source speaker's voice

    print(f"Translating ({len(src_tokens)} {src_lang} tokens) ...")
    tgt_text, tgt_tokens = translate_text_bridge(
        lm, tokenizer, spec, src_lang, tgt_lang, args.src_text,
        src_codec_raw=None if args.no_src_codec else src_tokens,
        num_beams=args.num_beams, max_text_tokens=args.max_text_tokens,
        max_codec_tokens=args.max_new_tokens, forced_tgt_text=args.forced_tgt_text,
        device=args.device,
    )
    print(f"{tgt_lang}_text: {tgt_text!r}")
    print(f"-> {len(tgt_tokens)} {tgt_lang} codec tokens")
    if len(tgt_tokens) == 0:
        raise RuntimeError(
            "Model produced zero codec tokens (immediate EOS, or the codec phase failed). "
            "Checkpoint may be undertrained or the input is out of distribution."
        )

    print("Decoding to audio ...")
    with torch.inference_mode():
        tgt_tensor = to_codec_tensor(tgt_tokens, feats.content_token_indices, args.device)
        resynth = codec.decode(content_token_indices=tgt_tensor, global_embedding=global_embedding)

    sf.write(args.output_wav, resynth.detach().cpu().numpy().squeeze(), codec.config.sample_rate)
    print(f"Wrote {args.output_wav}")


if __name__ == "__main__":
    main()