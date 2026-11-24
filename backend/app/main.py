import os
import sys

# Ensure repository root is on sys.path so 'ml.*' imports resolve anywhere
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_current_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.database.session import engine, Base, apply_lightweight_schema_updates
from app.api import api_router

# Initialize database tables
Base.metadata.create_all(bind=engine)
apply_lightweight_schema_updates()

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description=(
        "Explainable AI-Based Clinical Decision Support System for Doctors. "
        "NOTICE: This system provides clinical decision support with doctor-in-the-loop oversight. "
        "It does NOT claim to autonomously diagnose diseases or prescribe medications."
    ),
    debug=settings.DEBUG
)

# Enable CORS for React frontend (including any Vercel domain and local ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+|http://127\.0\.0\.1:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/")
def root():
    return {
        "status": "online",
        "system": settings.APP_NAME,
        "mode": "Doctor-in-the-loop Clinical Decision Support",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "database": "connected",
        "active_modules": [
            "auth", "patient_management", "ocr_clinical_nlp", "lab_analysis",
            "xgboost_foundation", "rag_retrieval", "rag_grounded_answering",
            "mri_classification", "tumor_progression_prediction",
            "multimodal_3d_segmentation", "early_warning_system"
        ]
    }
