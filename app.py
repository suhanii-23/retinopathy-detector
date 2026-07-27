"""
Gradio inference app for the diabetic retinopathy detector.

Preprocessing is imported from preprocess.py rather than reimplemented here.
That module is the single source of truth shared with train.py and evaluate.py,
which eliminates training/serving skew by construction — the most common and
most silent class of ML deployment bug.
"""

import gradio as gr
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

from preprocess import preprocess_array

CLASSES = ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]

CLASS_DESCRIPTIONS = {
    "Normal": "No signs of diabetic retinopathy detected.",
    "Mild": "Early stage, with microaneurysms present.",
    "Moderate": "More advanced, with some blocked blood vessels.",
    "Severe": "Many blood vessels are blocked.",
    "Proliferative": "Advanced stage, with abnormal new blood vessel growth.",
}

# Referable DR is the clinically actionable threshold: Moderate or worse means
# the patient should see an ophthalmologist. This is the decision a screening
# tool actually supports, so it is surfaced explicitly.
REFERABLE_THRESHOLD = 2

print("Downloading model from HuggingFace Hub...")
MODEL_PATH = hf_hub_download(
    repo_id="suhanii23/retinopathy-model",
    filename="diabetic_retinopathy_model.keras",
)
model = load_model(MODEL_PATH)
print("Model loaded.")


def predict_dr(image):
    if image is None:
        return "Please upload a retinal fundus image.", ""

    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    # Identical transform to training: crop -> resize 299 -> Ben Graham -> [-1, 1].
    x = preprocess_array(image)
    probs = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]

    # The head is a 5-way softmax, so the predicted class is the argmax.
    # (Thresholding each output at 0.5 and summing would be ordinal/multi-label
    # logic, which is incompatible with softmax: the probabilities sum to 1, so
    # at most one can exceed 0.5, and the result collapses to 0 or 1.)
    predicted = int(np.argmax(probs))
    label = CLASSES[predicted]
    confidence = float(probs[predicted])

    referable = predicted >= REFERABLE_THRESHOLD
    referral = (
        "**Referable** — grade is Moderate or worse; ophthalmologist review indicated."
        if referable
        else "**Non-referable** — below the standard screening referral threshold."
    )

    summary = (
        f"### {label}\n\n"
        f"{CLASS_DESCRIPTIONS[label]}\n\n"
        f"Model confidence: **{confidence:.1%}**\n\n"
        f"{referral}"
    )

    breakdown = "\n".join(
        f"{CLASSES[i]:<15s} {probs[i]:>7.2%}  {'#' * int(round(probs[i] * 40))}"
        for i in range(len(CLASSES))
    )

    return summary, breakdown


with gr.Blocks(title="Diabetic Retinopathy Detector") as demo:
    gr.Markdown(
        "# 👁️ Diabetic Retinopathy Detector\n"
        "Upload a retinal fundus image to grade diabetic retinopathy severity "
        "across five clinical stages.\n\n"
        "Xception backbone fine-tuned on the APTOS 2019 dataset, with Ben Graham "
        "contrast normalisation."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Retinal fundus image")
            predict_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            diagnosis = gr.Markdown()
            confidence_box = gr.Textbox(
                label="Probability distribution across all five stages",
                lines=6,
                show_copy_button=True,
            )

    predict_btn.click(
        fn=predict_dr,
        inputs=input_image,
        outputs=[diagnosis, confidence_box],
    )

    gr.Markdown(
        "---\n"
        "⚠️ **Educational project — not a medical device and not for clinical "
        "diagnosis.** Performance on the Severe and Proliferative classes is "
        "limited by training data scarcity. Always consult a qualified "
        "ophthalmologist."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
