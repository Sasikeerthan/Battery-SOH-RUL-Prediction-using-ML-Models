import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class SOHPredictor(nn.Module):
    def __init__(self, seq_len=5):
        super(SOHPredictor, self).__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def run_mlp(df, model_name="MLP"):
    # Compute SOH
    initial_capacity = df['Capacity (Ah)'][:5].max()
    df['SOH (%)'] = (df['Capacity (Ah)'] / initial_capacity) * 100

    # Create sequences
    def create_sequences(data, seq_len=5):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)

    SEQ_LEN = 5
    soh_values = df['SOH (%)'].values
    scaler = MinMaxScaler()
    soh_scaled = scaler.fit_transform(soh_values.reshape(-1, 1)).flatten()
    X, y = create_sequences(soh_scaled, SEQ_LEN)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Train the model
    model = SOHPredictor(seq_len=SEQ_LEN)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(100):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    # Predict
    model.eval()
    with torch.no_grad():
        predicted = model(X_tensor).squeeze().numpy()
    actual = y_tensor.squeeze().numpy()

    predicted_soh = predicted * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_
    actual_soh = actual * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_

    # RUL calculation
    threshold = 80.0
    rul = []
    for i in range(len(predicted_soh)):
        rem = 0
        for j in range(i, len(predicted_soh)):
            if predicted_soh[j] >= threshold:
                rem += 1
            else:
                break
        rul.append(rem)

    results_df = pd.DataFrame({
        'Cycle': df['Cycle'].iloc[-len(predicted_soh):].values,
        'Capacity (Ah)': df['Capacity (Ah)'].iloc[-len(predicted_soh):].values,
        'SOH (%)': actual_soh,
        'Predicted SOH (%)': predicted_soh,
        'Predicted RUL (cycles)': rul
    })

    # SOH Plot
    plt.figure(figsize=(10, 5))
    plt.plot(results_df['Cycle'], results_df['SOH (%)'], label='Actual')
    plt.plot(results_df['Cycle'], results_df['Predicted SOH (%)'], label='Predicted')
    plt.xlabel('Cycle')
    plt.ylabel('SOH (%)')
    plt.title(f'SOH Prediction - {model_name}')
    plt.legend()
    plt.grid()
    soh_path = f"backend/outputs/{model_name.lower()}_soh.png"
    plt.savefig(soh_path)
    plt.close()

    # RUL Plot
    plt.figure(figsize=(10, 5))
    plt.plot(results_df['Cycle'], results_df['Predicted RUL (cycles)'], marker='o', linestyle='-')
    plt.xlabel("Cycle")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title(f"RUL Prediction - {model_name}")
    plt.grid(True)
    rul_path = f"backend/outputs/{model_name.lower()}_rul.png"
    plt.savefig(rul_path)
    plt.close()

    metrics = {
        "MAE": mean_absolute_error(actual_soh, predicted_soh),
        "RMSE": np.sqrt(mean_squared_error(actual_soh, predicted_soh)),
        "R2": r2_score(actual_soh, predicted_soh)
    }

    return {
        "df": results_df,
        "max_capacity":initial_capacity,
        "metrics": metrics,
        "images": {
            "soh_plot": soh_path,
            "rul_plot": rul_path
        }
    }
