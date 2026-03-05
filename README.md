# 👁️ Diabetic Retinopathy Detector

An AI-powered tool for detecting and classifying diabetic retinopathy from retinal fundus images into five severity stages.

🔗 **Live Demo (HuggingFace Spaces)**
https://huggingface.co/spaces/suhanii23/retinopathy-detector

---

## Model

The trained model file (`.h5`) is **not included in this repository** due to GitHub's 100MB file size limit.

You can access the deployed model through the live HuggingFace demo above.
If running locally, download the model and place it in the **project root directory** before running the application.

---

## What is Diabetic Retinopathy?

Diabetic retinopathy is a diabetes complication that affects the eyes, caused by damage to the blood vessels in the retina. It is one of the leading causes of blindness worldwide. Early detection is critical — this tool aims to assist in screening by automatically classifying retinal images into five severity stages.

| Stage         | Description                                          |
| ------------- | ---------------------------------------------------- |
| Normal        | No signs of diabetic retinopathy                     |
| Mild          | Early stage with microaneurysms                      |
| Moderate      | More severe, with blocked blood vessels              |
| Severe        | Many blood vessels blocked                           |
| Proliferative | Advanced stage with abnormal new blood vessel growth |

---

## Results

| Metric                         | Value                          |
| ------------------------------ | ------------------------------ |
| Validation Accuracy            | 95.96%                         |
| Quadratic Weighted Kappa (QWK) | 0.9172                         |
| Dataset                        | APTOS 2019 Blindness Detection |
| Model                          | Xception (Transfer Learning)   |

### Per-Class Performance

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Normal        | 0.98      | 0.99   | 0.99     |
| Mild          | 0.67      | 0.61   | 0.64     |
| Moderate      | 0.74      | 0.80   | 0.77     |
| Severe        | 0.32      | 0.41   | 0.36     |
| Proliferative | 0.83      | 0.45   | 0.59     |

> The model performs strongly on Normal and Moderate cases. Severe and Proliferative classes have lower recall due to class imbalance in the training data — these are known limitations of the current version.

---




### Training Curves

The model was trained in two phases. Validation accuracy plateaued at ~96% with no significant overfitting.


---

## Model Details

* **Architecture:** Xception with custom classification head
* **Input:** 299×299 retinal fundus images
* **Output:** 5-class ordinal prediction (Normal → Proliferative)
* **Training Dataset:** APTOS 2019 Blindness Detection
* **Training Platform:** Kaggle (P100 GPU)
* **Framework:** TensorFlow / Keras

---

### Training Strategy

**Phase 1 — Warmup (2 epochs)**

* Froze all base layers, trained only the classification head
* Learning rate: 1e-3
* Allowed the new head to stabilize before fine-tuning

**Phase 2 — Fine-tuning (20 epochs, early stopped at 13)**

* Unfroze all layers
* Learning rate: 1e-4
* Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

### Key Improvements Over Baseline

| Feature          | Baseline             | Improved                          |
| ---------------- | -------------------- | --------------------------------- |
| Class weights    | ❌ commented out      | ✅ balanced                        |
| Augmentation     | horizontal flip only | 360° rotation + both flips + zoom |
| Normalization    | inconsistent         | consistent across train/val/test  |
| Monitored metric | val_acc              | val_loss (more stable)            |
| Model saving     | weights only         | full model                        |

---

## Preprocessing

Ben Graham's preprocessing method is applied to all images:

1. Crop black borders from retinal images
2. Resize to 299×299
3. Apply Gaussian blur subtraction to enhance blood vessel contrast
4. Normalize pixel values to [0, 1]

```python
def preprocess_image(image_path, sigmaX=10):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (299, 299))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)
    return image
```

---

## Project Structure

```
retinopathy-detector/
├── app.py
├── gradcam.py
├── preprocess.py
├── evaluate.py
├── train.py
├── requirements.txt
├── confusion_matrix.png
├── validation_predictions.png
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

---

### Requirements

```
gradio==4.44.0
tensorflow==2.16.2
opencv-python-headless
pillow
numpy
requests
scikit-learn
matplotlib
seaborn
huggingface_hub==0.23.0
```

---

## Limitations

* **Severe class recall is low (41%)** — only 29 validation samples, insufficient for robust learning
* **Proliferative recall is low (45%)** — similar imbalance issue
* **Not for clinical use** — this is an educational project, not a medical device
* **Image quality dependency** — performance degrades on low-quality or non-standard fundus images

---

## Future Work

* Add Grad-CAM heatmap overlay to show which retinal regions influenced the prediction
* Address class imbalance further with oversampling on minority classes
* Experiment with ensemble models such as Xception + EfficientNet
* Collect or augment more Severe and Proliferative samples

---

## Disclaimer

⚠️ This tool is for **educational purposes only**.
It is **not intended for clinical diagnosis**. Always consult a qualified ophthalmologist for medical evaluation.

---

## Acknowledgements

* Dataset: APTOS 2019 Blindness Detection (Kaggle)
* Preprocessing: Ben Graham's retinal preprocessing method
* Base architecture: Xception (Chollet, 2017)
