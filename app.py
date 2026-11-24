"""
Hugging Face Spaces Entry Point for AI-CDSS
Runs FastAPI Backend alongside Gradio on port 7860 (Hugging Face default)
"""

import os
import sys

# ── Ensure backend and root are on sys.path ──────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, "backend")

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

os.chdir(backend_dir)

# ── Import FastAPI Application ────────────────────────────────────────────────
from app.main import app as fastapi_app
import gradio as gr
import uvicorn

# ── Create a clean Gradio interface for the Hugging Face Space viewer ─────────
def system_status():
    return {
        "Platform": "AI-CDSS: Clinical Decision Support System",
        "Backend Status": "Online & Healthy",
        "Regulatory Scope": "Class II non-autonomous SaMD Prototype",
        "Active Models": "ViT-Base/16, ResNet-1D, XGBoost, Attention U-Net, FAISS RAG",
        "API Documentation": "/docs",
        "Health Endpoint": "/api/health"
    }

with gr.Blocks(title="AI-CDSS Clinical Platform") as demo:
    gr.Markdown(
        """
        # 🩺 AI-CDSS: Explainable Clinical Decision Support System
        ### Physician-in-the-Loop Diagnostic & Prognostic Intelligence Backend
        
        This Hugging Face Space powers the active REST API for the AI-CDSS platform.
        - **Swagger API Docs:** Accessible at [`/docs`](./docs)
        - **Health Check:** Accessible at [`/api/health`](./api/health)
        - **React Doctor Portal:** Connect your frontend via `VITE_API_BASE_URL`
        """
    )
    with gr.Row():
        check_btn = gr.Button("Inspect System Telemetry", variant="primary")
    status_output = gr.JSON(label="System Status & Loaded Modules")
    check_btn.click(fn=system_status, outputs=status_output)

# Mount Gradio onto the root of FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
