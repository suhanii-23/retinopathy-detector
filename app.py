import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

from preprocess import denormalize_for_display, preprocess_array

MODEL_REPO_ID = "suhanii23/retinopathy-model"
MODEL_FILENAME = "diabetic_retinopathy_model.keras"

# Inlined rather than imported from model.py: importing model.py would pull in
# tensorflow.keras.applications (Xception) at Space startup just to read five
# strings, slowing down cold starts for no reason.
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

CLASS_DESCRIPTIONS = {
    'No DR': 'No signs of diabetic retinopathy detected.',
    'Mild': 'Mild non-proliferative diabetic retinopathy. Early stage with microaneurysms.',
    'Moderate': 'Moderate non-proliferative diabetic retinopathy. More severe than mild, with blocked blood vessels.',
    'Severe': 'Severe non-proliferative diabetic retinopathy. Many blood vessels are blocked.',
    'Proliferative DR': 'Proliferative diabetic retinopathy. Advanced stage with new abnormal blood vessel growth.',
}

REFERABLE_THRESHOLD = 2  # Moderate or worse

model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
model = load_model(model_path)
print("Model loaded successfully!")


def bar_chart(probs):
    lines = []
    for name, p in zip(CLASS_NAMES, probs):
        filled = int(round(p * 20))
        bar = '█' * filled + '░' * (20 - filled)
        lines.append(f'{name:<18} {bar} {p * 100:5.1f}%')
    return '\n'.join(lines)


def predict_dr(image):
    if image is None:
        return None, "Please upload an image before analyzing.", ""

    image_rgb = np.array(image.convert('RGB'))
    processed = preprocess_array(image_rgb)

    probs = model.predict(np.expand_dims(processed, axis=0), verbose=0)[0]

    # argmax, not thresholding: the model has a softmax head, so its five
    # outputs sum to 1.0 and at most one can ever exceed 0.5. Thresholding
    # each output independently is ordinal/multi-label logic that belongs
    # to a different kind of model — against a softmax it collapses the
    # result to always 0 or 1 (No DR or Mild), making Moderate, Severe and
    # Proliferative unreachable regardless of what the model predicts.
    predicted_class = int(np.argmax(probs))
    result_class = CLASS_NAMES[predicted_class]
    description = CLASS_DESCRIPTIONS[result_class]

    referable = predicted_class >= REFERABLE_THRESHOLD
    referral_line = (
        "**Referable — recommend referral to an ophthalmologist.**"
        if referable else
        "**Non-referable** based on this prediction."
    )

    diagnosis_md = f"**Diagnosis: {result_class}**\n\n{description}\n\n{referral_line}"
    display_image = denormalize_for_display(processed)

    return display_image, diagnosis_md, bar_chart(probs)


with gr.Blocks(title="Diabetic Retinopathy Detector") as demo:
    gr.Markdown("""
    # Diabetic Retinopathy Detection
    Upload a retinal fundus image to detect diabetic retinopathy severity.

    **Classes:** No DR → Mild → Moderate → Severe → Proliferative DR
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Retinal Image")
            predict_btn = gr.Button("Analyze Image", variant="primary")

        with gr.Column():
            processed_image = gr.Image(label="Preprocessed Image")
            diagnosis = gr.Markdown()
            probability_chart = gr.Textbox(label="Class Probabilities", lines=6)

    predict_btn.click(
        fn=predict_dr,
        inputs=input_image,
        outputs=[processed_image, diagnosis, probability_chart],
    )

    gr.Markdown("""
    ### Medical Disclaimer
    This tool is for **educational purposes only** and is **not a medical device**.
    It has not been clinically validated. Severe and Proliferative DR performance
    is limited by data scarcity in the training set — do not rely on this tool for
    diagnosis. Always consult a qualified ophthalmologist for medical evaluation.
    """)

if __name__ == "__main__":
    demo.launch()
