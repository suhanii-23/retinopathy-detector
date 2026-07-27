# Changes

## Why this pass happened

The README claimed 95.96% validation accuracy and 0.9172 QWK. Those numbers don't
add up: applying the README's own per-class recall table to the known APTOS 2019
class distribution (No DR 1805, Mild 370, Moderate 999, Severe 193, Proliferative
295) under the documented 15% stratified split implies accuracy of roughly 82.6%,
not 96%. The real numbers require re-running evaluation on Kaggle, so every metric
in this pass is a literal `TODO_ACCURACY` / `TODO_QWK` / `TODO` placeholder — none
were invented or guessed.

## New files

- **`preprocess.py`** — single source of truth for image preprocessing (crop,
  resize, Ben Graham high-pass, Xception `preprocess_input` normalization to
  [-1, 1]). Both `train.py` and `app.py` import from it instead of keeping their
  own copies, so training/serving skew is structurally impossible rather than
  merely unlikely.
- **`model.py`** — Xception + GAP + dropout/dense head, plus
  `set_backbone_trainable()`, which keeps BatchNorm layers frozen when the
  backbone is unfrozen during fine-tuning. The original notebook didn't do this;
  it's included here as a real fix, not boilerplate, because BN running stats
  updating from batches of 16 fundus images (nothing like ImageNet's
  distribution) destabilizes fine-tuning.
- **`train.py`** — two-phase training CLI (warmup frozen-backbone, then
  fine-tune unfrozen). Class weights are computed on the training split only
  (computing on the full dataframe would leak validation class frequencies).
  History is saved with TF 2.x keys (`accuracy`/`val_accuracy`), not the TF 1.x
  `acc`/`val_acc` the original notebook used — that mismatch is why no training
  curves were ever produced from it (`KeyError` against a TF 2.x history object).
- **`evaluate.py`** — loads the saved validation split and model, reports
  accuracy + QWK together, a `classification_report`, a two-panel confusion
  matrix (raw counts + row-normalized recall), training curves, a referable-DR
  (≥ Moderate) sensitivity/specificity breakdown, and an error-distance
  histogram (how many misclassifications are 1/2/3/4 classes away — what QWK
  rewards and accuracy hides). Everything is written to `metrics.json`.

## Bug fixes in `app.py` (highest priority)

1. **Wrong prediction logic.** The deployed code did
   `int((predictions[0] > 0.5).sum())`, which is ordinal/multi-label
   thresholding. The model has a softmax head — five probabilities summing to
   1.0, at most one ever exceeding 0.5 — so the result was always 0 or 1. In
   practice the app could only ever output "Normal" or "Mild"; a confident
   Proliferative prediction displayed as "Mild". Fixed to `np.argmax(probs)`,
   with a comment explaining why thresholding is incompatible with softmax so
   it doesn't regress.
2. **Normalization mismatch (training/serving skew).** The deployed code did
   `processed.astype("float32") / 255.0` (scaling to [0, 1]), but training used
   Xception's `preprocess_input` (scaling to [-1, 1]). Fixed by importing
   `preprocess_array` from `preprocess.py` and deleting the app's local
   preprocessing copy entirely.

Also while rewriting: the app now shows the full 5-class probability
distribution as a text bar chart, adds a referable/non-referable line (≥
Moderate), handles `image is None` (Analyze clicked with nothing uploaded), and
the disclaimer now explicitly calls out that Severe/Proliferative performance is
limited by data scarcity. The model is now fetched from HuggingFace Hub
(`suhanii23/retinopathy-model`, matching the actual deployment) via
`hf_hub_download` instead of a GitHub Releases URL that no longer matches the
model format (`.keras`, not `.h5`).

## `.gitignore` / tracking

Added `*.keras`, `.ipynb_checkpoints/`, `val_split.npz`, `history.json`, and
dataset file patterns (`*.csv`, `train_images/`, `test_images/`, `data/`,
`*.zip`) alongside the existing `venv/`, `.venv/`, `*.h5`, `__pycache__/`,
`.DS_Store` entries.

`venv/` was checked and was **never tracked by git** (`git ls-files venv`
returns nothing), so `git rm -r --cached venv` was not needed. The directory is
untouched on disk.

## `requirements.txt` / `requirements-dev.txt`

Split by concern: `requirements.txt` now covers inference only (gradio,
tensorflow, opencv-python-headless, pillow, numpy, huggingface_hub — the last
of which the old file was missing, since the old app pulled from a GitHub
Release, not the Hub). `requirements-dev.txt` adds scikit-learn, matplotlib,
seaborn, tqdm, pandas for `train.py`/`evaluate.py`, via `-r requirements.txt`.

## `README.md`

- Results table: 95.96% / 0.9172 → `TODO_ACCURACY` / `TODO_QWK`, with a note
  pointing at this file for why the old numbers don't hold up.
- Preprocessing step 4: "Normalize to [0, 1]" → correctly [-1, 1] via Xception's
  `preprocess_input`, with a note that this is the range Xception was
  pretrained on.
- "Project Structure" now lists the four files that actually exist
  (`preprocess.py`, `model.py`, `train.py`, `evaluate.py`) plus `app.py`;
  `gradcam.py` removed (never existed, was future work mislabeled as a file).
  Grad-CAM now appears only under Future Work.
- Per-class performance table kept but marked `TODO` — needs regenerating.
- Limitations section kept as-is, with two caveats added: single split reused
  for both early stopping and final reporting (mild optimistic bias), and no
  independent held-out test set.

## What I still need to do manually

1. **Re-run evaluation on Kaggle** using `evaluate.py` against the trained
   model and the saved `val_split.npz`/`history.json`, then replace
   `TODO_ACCURACY`, `TODO_QWK`, and the per-class table in `README.md` with the
   real numbers.
2. **Upload `preprocess.py` and `app.py` to the HuggingFace Space**
   (`suhanii23/retinopathy-detector`) — this repo's `app.py` now imports
   `model.py` and `preprocess.py`, so both need to go up alongside it, along
   with `model.py` for the `CLASS_NAMES` import.
3. **Update your resume bullet** — anything citing 95.96%/0.9172 needs to wait
   for the re-run numbers above; don't quote a number until `evaluate.py` has
   actually produced it.
