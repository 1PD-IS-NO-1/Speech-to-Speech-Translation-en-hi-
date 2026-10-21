"""
check_text_mt.py -- text-only translation sanity check for one or more checkpoints.

  python check_text_mt.py ./runs3/qwen05b_s2st/stage_mt ./runs3/qwen05b_s2st

Each argument is a model folder (must contain config.json, model.safetensors,
vocab_spec.json). Prints en->hi and hi->en translations so you can compare
the text-only stage with the final model.
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from train_qwen_codec_sts import VocabSpec, translate
except ImportError:
    from qwen_s2st import VocabSpec, translate

def ckpt_has_lm_head(path):
    """True if lm_head.weight is stored in the checkpoint (fixed/new checkpoints)."""
    try:
        import glob
        import os
        from safetensors import safe_open
        for fn in glob.glob(os.path.join(path, "*.safetensors")):
            with safe_open(fn, "pt") as f:
                if "lm_head.weight" in f.keys():
                    return True
        return False
    except Exception:
        return False  # unknown -> tie (correct for checkpoints made by qwen_s2st.py)


EN = ["Please let me know if you are free tomorrow evening.",
      "The meeting has been moved to next week because of the holiday.",
      "A simple question used to check someone's availability before making plans."]
HI = ["कृपया मुझे बताइए कि क्या आप कल शाम को खाली हैं।",
      "छुट्टी की वजह से बैठक अगले सप्ताह के लिए टाल दी गई है।"]

for d in sys.argv[1:]:
    print(f"\n===== {d} =====")
    lm = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16)
    if ckpt_has_lm_head(d):
        print('[head] lm_head loaded from the checkpoint')
    else:
        lm.lm_head.weight = lm.get_input_embeddings().weight  # old checkpoint: head was tied
        print('[tie] lm_head now shares the embedding table')
    lm = lm.cuda().eval()
    tok, spec = AutoTokenizer.from_pretrained(d), VocabSpec.load(d)
    for s in EN:
        print(f"en: {s}\nhi: {translate(lm, tok, spec, 'en', 'hi', s, text_only=True, max_new_tokens=200)[0]}\n")
    for s in HI:
        print(f"hi: {s}\nen: {translate(lm, tok, spec, 'hi', 'en', s, text_only=True, max_new_tokens=200)[0]}\n")
    del lm
    torch.cuda.empty_cache()