import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load saved model
model = tf.keras.models.load_model("skinscan_model.keras")
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

CLASS_NAMES = ["acne", "benign_keratosis", "melanoma", "nevus"]
IMG_SIZE = (224, 224)

def predict_pil(image):
    if image is None:
        return "Please upload an image first.", None

    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    probs = model.predict(arr, verbose=0)[0]
    result = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    top_class = CLASS_NAMES[int(np.argmax(probs))]
    confidence = float(np.max(probs))

    summary = f"### Prediction: {top_class.replace('_', ' ').title()}\n**Confidence:** {confidence:.1%}"
    return summary, result

with gr.Blocks(title="SkinScan AI") as demo:
    gr.Markdown("# 🩺 SkinScan AI")
    gr.Markdown("### AI-Based Skin Image Classification")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload a skin image")
            analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            prediction_output = gr.Markdown(label="Prediction")
            probs_output = gr.Label(num_top_classes=4, label="Class probabilities")
    gr.Markdown(
        "---\n"
        "⚠️ **Educational prototype — not a medical diagnosis. Do not use this system to make medical decisions.**"
    )
    analyze_btn.click(fn=predict_pil, inputs=image_input, outputs=[prediction_output, probs_output])

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
