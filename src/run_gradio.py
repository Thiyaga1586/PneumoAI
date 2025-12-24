# import os
# import io
# import json
# import requests
# import gradio as gr
# from PIL import Image

# API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# def predict_via_api(img: Image.Image, true_label: str):
#     if img is None:
#         return "No image provided.", ""

#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     buf.seek(0)

#     files = {"file": ("upload.png", buf, "image/png")}
#     data = {}
#     if true_label and true_label != "UNKNOWN":
#         data["true_label"] = true_label

#     r = requests.post(API_URL, files=files, data=data, timeout=60)
#     r.raise_for_status()
#     out = r.json()

#     if "error" in out:
#         return f"ERROR: {out['error']}", json.dumps(out, indent=2)

#     txt = (
#         f"Label: {out['label']} | prob={out['probability']} | "
#         f"v={out['model_version']} | {out['latency_ms']} ms"
#     )
#     return txt, json.dumps(out, indent=2)

# with gr.Blocks(title="Pneumonia UI (FastAPI-backed)") as demo:
#     gr.Markdown("# Pneumonia Inference UI (Gradio → FastAPI)")
#     gr.Markdown(f"Backend: `{API_URL}`")

#     with gr.Row():
#         inp = gr.Image(type="pil", label="Upload Chest X-ray")
#         with gr.Column():
#             true_label = gr.Dropdown(
#                 choices=["UNKNOWN", "NORMAL", "PNEUMONIA"],
#                 value="UNKNOWN",
#                 label="True label (optional, for eval_latest)",
#             )
#             btn = gr.Button("Predict")

#     out_txt = gr.Textbox(label="Result")
#     out_raw = gr.Code(label="Raw JSON", language="json")

#     btn.click(fn=predict_via_api, inputs=[inp, true_label], outputs=[out_txt, out_raw])

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860)
