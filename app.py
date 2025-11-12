import gradio as gr
import numpy as np
import cv2
import os
import requests
from tensorflow.keras.models import load_model
from PIL import Image

# Constants
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
CLASSES = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']
CLASS_DESCRIPTIONS = {
    'Normal': 'No signs of diabetic retinopathy detected.',
    'Mild': 'Mild non-proliferative diabetic retinopathy. Early stage with microaneurysms.',
    'Moderate': 'Moderate non-proliferative diabetic retinopathy. More severe than mild, with blocked blood vessels.',
    'Severe': 'Severe non-proliferative diabetic retinopathy. Many blood vessels are blocked.',
    'Proliferative': 'Proliferative diabetic retinopathy. Advanced stage with new abnormal blood vessel growth.'
}

# --- 🔽 Ensure model is available locally ---
MODEL_PATH = "diabetic_retinopathy_full_model.h5"
MODEL_URL = "https://github.com/suhanii-23/retinopathy-detector/releases/download/v1.0-model/diabetic_retinopathy_full_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from GitHub release...")
    r = requests.get(MODEL_URL, allow_redirects=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Model downloaded successfully!")

# Load model
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# --- Image Preprocessing ---
def crop_image_from_gray(img, tol=7):
    """Ben Graham's preprocessing: crop black borders"""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def preprocess_image(image, sigmaX=10):
    """Apply Ben Graham's preprocessing method"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

# --- Prediction ---
def predict_dr(image):
    """Main prediction function"""
    try:
        processed = preprocess_image(image)
        img_array = processed.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array, verbose=0)
        binary_pred = (predictions > 0.5).astype(int)
        final_class = binary_pred.sum(axis=1)[0] - 1
        
        confidences = {CLASSES[i]: float(predictions[0][i]) for i in range(len(CLASSES))}
        result_class = CLASSES[final_class]
        description = CLASS_DESCRIPTIONS[result_class]
        
        return (
            processed,
            f"**Diagnosis: {result_class}**\n\n{description}",
            confidences
        )
    
    except Exception as e:
        return None, f"Error: {str(e)}", {}

# --- Gradio UI ---
with gr.Blocks(title="Diabetic Retinopathy Detector") as demo:
    gr.Markdown("""
    # 🏥 Diabetic Retinopathy Detection  
    Upload a retinal fundus image to detect diabetic retinopathy severity.
    
    **Classes:** Normal → Mild → Moderate → Severe → Proliferative
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Retinal Image")
            predict_btn = gr.Button("🔍 Analyze Image", variant="primary")
            
        with gr.Column():
            processed_image = gr.Image(label="Preprocessed Image")
            diagnosis = gr.Markdown()
            confidence = gr.Label(label="Confidence Scores", num_top_classes=5)
    
    predict_btn.click(
        fn=predict_dr,
        inputs=input_image,
        outputs=[processed_image, diagnosis, confidence]
    )
    
    gr.Markdown("""
    ### ⚠️ Medical Disclaimer  
    This tool is for educational purposes only. Always consult a qualified healthcare provider.
    """)

if __name__ == "__main__":
    demo.launch()
