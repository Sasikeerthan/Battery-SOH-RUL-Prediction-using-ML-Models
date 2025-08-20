import streamlit as st
import requests
import pandas as pd

st.title("🔋 Battery SOH & RUL Prediction")

# File Upload
uploaded_file = st.file_uploader("Upload a .mat battery dataset (optional)", type=["mat"])

# OR select from existing datasets
preloaded_datasets = ["B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"]
selected_dataset = st.selectbox("Or select from saved datasets:", preloaded_datasets)

# Select Models
selected_models = st.multiselect(
    "Choose models to run",
    ["MLP", "LSTM", "Transformer"],
    default=["MLP", "LSTM", "Transformer"]
)

if st.button("🚀 Run Models"):
    with st.spinner("Processing..."):
        if uploaded_file:
            files = {"file": uploaded_file}
            data = {"models": ",".join(selected_models), "dataset_name": uploaded_file.name}
        else:
            files = {}
            data = {
                "models": ",".join(selected_models),
                "dataset_name": selected_dataset
            }

        response = requests.post("http://localhost:8000/run_models/", data=data, files=files)

        if response.status_code == 200:
            results = response.json()
            for model_name, output in results.items():
                st.subheader(f"📊 {model_name} Results")
                
                #max_capacity
                st.write(f"🔋 **Max Capacity:** `{output['max_capacity']:.3f} Ah`")




                
                

                # Table
                st.write("### 📄 Prediction Table")
                st.dataframe(pd.DataFrame(output["df"]))
                # Metrics
                # Select only the MAE and RMSE metrics
                selected_metrics = {k: output["metrics"][k] for k in ["MAE", "RMSE"] if k in output["metrics"]}

                st.write("### 📈 Metrics")
                st.json(selected_metrics)


                # SOH Plot
                st.write("### 🔋 SOH Plot")
                soh_url = f"http://localhost:8000/outputs/{model_name}_soh.png"
                st.image(soh_url, caption=f"{model_name} - SOH Prediction", use_column_width=True)

                # RUL Plot
                st.write("### ⏳ RUL Plot")
                rul_url = f"http://localhost:8000/outputs/{model_name}_rul.png"
                

                st.image(rul_url, caption=f"{model_name} - RUL Prediction", use_column_width=True)
        else:
            st.error("❌ Failed to run models.")
