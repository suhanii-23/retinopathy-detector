[![HuggingFace Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/suhanii23/retinopathy-detector)
# 👁️ Diabetic Retinopathy Detector

An AI-powered tool for detecting and classifying diabetic retinopathy from retinal fundus images into five severity stages.

🔗 **Live Demo (HuggingFace Spaces)**
https://huggingface.co/spaces/suhanii23/retinopathy-detector

---

## Model

The trained model is not included in this repository. The deployed app downloads it at
runtime from the [`suhanii23/retinopathy-model`](https://huggingface.co/suhanii23/retinopathy-model)
HuggingFace Hub repo (`diabetic_retinopathy_model.keras`).

---

## What is Diabetic Retinopathy?

Diabetic retinopathy is a diabetes complication that affects the eyes, caused by damage to the blood vessels in the retina. It is one of the leading causes of blindness worldwide. Early detection is critical — this tool aims to assist in screening by automatically classifying retinal images into five severity stages.

| Stage         | Description                                          |
| ------------- | ---------------------------------------------------- |
| No DR         | No signs of diabetic retinopathy                     |
| Mild          | Early stage with microaneurysms                      |
| Moderate      | More severe, with blocked blood vessels              |
| Severe        | Many blood vessels blocked                           |
| Proliferative | Advanced stage with abnormal new blood vessel growth |

---

## Results

| Metric                         | Value                          |
| ------------------------------ | ------------------------------ |
| Validation Accuracy            | TODO_ACCURACY                  |
| Quadratic Weighted Kappa (QWK) | TODO_QWK                       |
| Dataset                        | APTOS 2019 Blindness Detection |
| Model                          | Xception (Transfer Learning)   |

> These numbers are being regenerated with `evaluate.py` against the saved validation
> split. The previous values published here were incorrect — see `CHANGES.md` for
> details on how that was discovered.

### Per-Class Performance

**TODO — needs regenerating with `evaluate.py`.**

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| No DR         | TODO      | TODO   | TODO     |
| Mild          | TODO      | TODO   | TODO     |
| Moderate      | TODO      | TODO   | TODO     |
| Severe        | TODO      | TODO   | TODO     |
| Proliferative | TODO      | TODO   | TODO     |

---

## Model Details

* **Architecture:** Xception (ImageNet) + GlobalAveragePooling2D + Dropout(0.5) + Dense(2048, relu) + Dropout(0.5) + Dense(5, softmax)
* **Input:** 299×299 retinal fundus images
* **Output:** 5-class ordinal prediction (No DR → Proliferative)
* **Training Dataset:** APTOS 2019 Blindness Detection (3,662 images)
* **Split:** `train_test_split(test_size=0.15, random_state=2006, stratify=y)` → 550 validation images
* **Training Platform:** Kaggle (P100 GPU)
* **Framework:** TensorFlow / Keras

---

### Training Strategy

**Phase 1 — Warmup (2 epochs)**

* Backbone frozen, only the classification head trained
* Learning rate: 1e-3

**Phase 2 — Fine-tuning (up to 20 epochs)**

* Backbone unfrozen (BatchNorm layers kept frozen — see `model.py`)
* Learning rate: 1e-4
* Batch size: 16, balanced class weights
* Callbacks (all monitoring `val_loss`): `ModelCheckpoint`, `ReduceLROnPlateau(patience=3, factor=0.5)`, `EarlyStopping(patience=8)`

### Augmentation

`rotation_range=360`, horizontal + vertical flip, `zoom_range=[0.98, 1.02]`, `width_shift_range=0.01`, `height_shift_range=0.01`. Validation data is not augmented. A fundus image has no canonical orientation, so full rotation and both flips are label-preserving. Brightness/contrast jitter is deliberately excluded — it would partially undo the Ben Graham illumination normalization below.

---

## Preprocessing

Ben Graham's preprocessing method is applied to all images, implemented once in `preprocess.py` and shared by both training and serving code:

1. Crop black borders from retinal images
2. Resize to 299×299
3. Apply Gaussian blur subtraction (`4*I - 4*blur(I) + 128`) to suppress low-frequency content (illumination, colour cast) and enhance vessels, microaneurysms, exudates and haemorrhages
4. Normalize via `tensorflow.keras.applications.xception.preprocess_input`, which scales pixels to **[-1, 1]** — the range Xception was pretrained on

```python
def preprocess_image(image_path, sigma_x=SIGMA_X):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigma_x), -4, 128)
    return preprocess_input(image.astype(np.float32))
```

---

## Project Structure

```
retinopathy-detector/
├── app.py              # Gradio inference app
├── preprocess.py        # Shared preprocessing (imported by train.py and app.py)
├── model.py             # Model architecture + backbone freeze/unfreeze
├── train.py             # Two-phase training CLI
├── evaluate.py           # Evaluation CLI (metrics, confusion matrix, curves)
├── requirements.txt      # Inference-only dependencies
├── requirements-dev.txt   # Training/evaluation dependencies
└── README.md
```

---

## How to Use

### Running Locally

```
git clone https://github.com/suhanii-23/retinopathy-detector
cd retinopathy-detector
pip install -r requirements.txt
python app.py
```

### Training

```
pip install -r requirements-dev.txt
python train.py --data-dir /path/to/aptos2019 --output /path/to/output
python evaluate.py --model /path/to/output/diabetic_retinopathy_model.keras \
    --val-split /path/to/output/val_split.npz --history /path/to/output/history.json
```

---

## Limitations

* **Severe and Proliferative recall are low** — 193 and 295 training images respectively, insufficient for robust learning relative to the 1805 No DR images
* **Not for clinical use** — this is an educational project, not a medical device
* **Image quality dependency** — performance degrades on low-quality or non-standard fundus images
* **Single split, reused for both stopping and reporting** — the same 550 validation images are used both to early-stop training and to report final metrics, which introduces a mild optimistic bias. A held-out test set never seen during model selection would give a more honest estimate.

---

## Future Work

* Add Grad-CAM heatmap overlay to show which retinal regions influenced the prediction
* Address class imbalance further with oversampling on minority classes
* Experiment with ensemble models such as Xception + EfficientNet
* Collect or augment more Severe and Proliferative samples
* Hold out a separate test set, independent of the validation split used for early stopping

---

## Disclaimer

⚠️ This tool is for **educational purposes only**.
It is **not intended for clinical diagnosis**. Always consult a qualified ophthalmologist for medical evaluation.

---

## Acknowledgements

* Dataset: APTOS 2019 Blindness Detection (Kaggle)
* Preprocessing: Ben Graham's retinal preprocessing method
* Base architecture: Xception (Chollet, 2017)
