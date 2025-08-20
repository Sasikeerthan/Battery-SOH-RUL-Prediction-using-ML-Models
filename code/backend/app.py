from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.model_runner import run_models
import shutil
import os

app = FastAPI()
# Mount outputs folder for serving images
app.mount("/outputs", StaticFiles(directory="backend/outputs"), name="outputs")  # <-- ADD THIS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run_models/")
async def run_models_api(
    models: str = Form(...),
    dataset_name: str = Form(...),
    file: UploadFile = None,
):
    os.makedirs("uploads", exist_ok=True)

    if file:
        dataset_path = f"uploads/{file.filename}"
        with open(dataset_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    else:
        dataset_path = f"datasets/{dataset_name}"  # You should save B0005-8 here

    selected_models = models.split(",")
    return run_models(dataset_path, selected_models)
