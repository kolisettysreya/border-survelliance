import gradio as gr
import cv2
import numpy as np
import os
from huggingface_hub import hf_hub_download


# ---- Load Model ----
def load_model():
    from ultralytics import YOLO
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        hf_hub_download(
            repo_id="Kolisetty/border-surveillance-yolov8",
            filename="best.pt",
            local_dir="models"
        )
    return YOLO(model_path)


print("Loading model...")
model = load_model()
print("Model loaded!")


# ---- Detection Function ----
def detect(image, confidence):
    if image is None:
        return None, "No image uploaded"

    # image comes in as RGB from Gradio
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model(img_bgr, conf=confidence)
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    boxes = results[0].boxes
    names = model.names

    if len(boxes) > 0:
        report = f"⚠️ ALERT — {len(boxes)} object(s) detected!\n\n"
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf_val = float(box.conf[0])
            report += f"  #{i + 1} → {names[cls]} ({conf_val:.2%} confidence)\n"
    else:
        report = "✅ No threats detected in this frame"

    return annotated_rgb, report


# ---- Gradio UI ----
with gr.Blocks(title="Border Surveillance System") as demo:
    gr.Markdown("""
    # 🛡️ Border Surveillance System
    AI-powered perimeter monitoring using **YOLOv8m** trained on **VisDrone** dataset.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 Input")
            input_image = gr.Image(label="Upload Surveillance Image")
            confidence = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.3,
                step=0.05,
                label="Confidence Threshold"
            )
            detect_btn = gr.Button("🔍 Run Detection", variant="primary")

        with gr.Column():
            gr.Markdown("### 📊 Output")
            output_image = gr.Image(label="Detection Result")
            output_text = gr.Textbox(label="Detection Report", lines=8)

    detect_btn.click(
        fn=detect,
        inputs=[input_image, confidence],
        outputs=[output_image, output_text]
    )

    gr.Markdown("""
    ---
    ### ℹ️ About
    | | |
    |--|--|
    | **Model** | YOLOv8m |
    | **Dataset** | VisDrone |
    | **Epochs** | 30 |
    | **Classes** | Person, Vehicle, Motorcycle, Drone, Animal |
    """)

demo.launch()