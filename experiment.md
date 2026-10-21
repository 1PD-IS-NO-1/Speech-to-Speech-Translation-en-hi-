# Experiments: how we got here

Put your earlier training scripts in this folder (for example the single file that contains all the
experiments, unchanged, as `train_qwen_s2s_experiment.py`). They are kept for reference only and are **not maintained**.
The maintained code is `../train_qwen_codec_sts.py`.

Suggested structure: one row per experiment. Fill the *Result* column with your own metrics
(chrF on the dev set, a sentence of listening notes, whatever you measured). The qualitative notes
below come from our runs; replace them with numbers when you have them.

| # | Experiment | What we changed | What we observed |
|---|---|---|---|
| 1 | **Codec-only**: EN speech tokens → HI speech tokens | no text at all, random-init speech embeddings | underperformed; hard to learn from ~100k pairs with randomly initialised speech embeddings |
| 2 | **Text bridge**: (EN text + EN tokens) → (HI text + HI tokens), plus HI→EN, full fine-tune, high LR (2e-4), 10 epochs | model writes target text first, then speech tokens | speech tokens matched the generated text well (the audio side worked), but the **translated text was wrong**: Qwen's translation ability had been overwritten |
| 3 | Embedding warm-up + LoRA, weighted loss, text-only MT mixed in, embeddings frozen | protect text knowledge | text better protected, but **audio got worse** (frozen embeddings / small LoRA could not learn the new modality well); also ran out of memory on an 8 GB GPU |
| 4 | **Two-stage curriculum** (current): MT first, then a fresh LoRA + trainable speech-token rows, split embeddings | see main README | good EN→HI text and audio; translation mostly preserved (some mild degradation in HI→EN on a few test sentences) |

Lessons:
1. Speech tokens are a new modality: they need real capacity (a LoRA with a decent rank *and* trainable
   embeddings for the new tokens), not just embeddings.
2. Translation knowledge is fragile under a high learning rate; teach/refresh translation first, keep
   replaying text-only pairs, and use a lower LR for pretrained weights.
3. Loss weighting matters because speech tokens outnumber text tokens ~10:1.
4. Check how weights are saved: a tied output layer can be dropped from the checkpoint and reloaded as
   a random layer (see "Troubleshooting" in the main README).