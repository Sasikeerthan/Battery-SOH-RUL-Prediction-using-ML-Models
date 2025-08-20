
import os
import pandas as pd
from backend.models.battery_soh_rul_prediction import run_mlp
from backend.models.istm_check import run_lstm
from backend.models.Transformer import run_transformer
from backend.data_utils import load_mat_data


# backend/model_runner.py



def run_models(dataset_path: str, selected_models: list[str]) -> dict:
    from numpy import generic

    df = load_mat_data(dataset_path)
    results = {}

    for model_name in selected_models:
        model_func = {
            "MLP": run_mlp,
            "LSTM": run_lstm,
            "Transformer": run_transformer
        }.get(model_name)

        if model_func:
            model_result = model_func(df.copy())
            metrics = model_result["metrics"]

            # Convert NumPy types to Python types
            for key in metrics:
                if isinstance(metrics[key], generic):
                    metrics[key] = metrics[key].item()

            results[model_name] = {
                "metrics": metrics,
                # Optional: convert df to JSON-safe format if needed
                "df": model_result["df"].to_dict(orient="records"),
                "max_capacity": model_result.get("max_capacity", None),
                "images": model_result.get("images", {})
            }

    return results
