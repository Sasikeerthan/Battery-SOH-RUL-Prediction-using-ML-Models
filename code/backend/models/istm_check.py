import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LSTMSOHPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        return self.fc2(out)

def run_lstm(df):
    model_name = "LSTM"
    
    # SOH Calculation
    initial_capacity = max(df['Capacity (Ah)'][:5])
    df['SOH (%)'] = (df['Capacity (Ah)'] / initial_capacity) * 100

    def create_sequences(data, seq_length=5):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    soh_values = df['SOH (%)'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    soh_scaled = scaler.fit_transform(soh_values).flatten()
    X, y = create_sequences(soh_scaled, seq_length=5)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(2)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = LSTMSOHPredictor()
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

    model.eval()
    with torch.no_grad():
        predicted = model(X_tensor).squeeze().numpy()
    actual = y_tensor.squeeze().numpy()

    predicted_soh = predicted * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_
    actual_soh = actual * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_

    threshold = 80
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

    # 🔽 Save Plots
    soh_path = f"backend/outputs/{model_name.lower()}_soh.png"
    rul_path = f"backend/outputs/{model_name.lower()}_rul.png"

    plt.figure(figsize=(10, 5))
    plt.plot(results_df['Cycle'], results_df['SOH (%)'], label='Actual')
    plt.plot(results_df['Cycle'], results_df['Predicted SOH (%)'], label='Predicted')
    plt.xlabel('Cycle')
    plt.ylabel('SOH (%)')
    plt.title(f'SOH Prediction - {model_name}')
    plt.legend()
    plt.grid()
    plt.savefig(soh_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(results_df['Cycle'], results_df['Predicted RUL (cycles)'], marker='o', linestyle='-')
    plt.xlabel("Cycle")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title(f"RUL Prediction - {model_name}")
    plt.grid(True)
    plt.savefig(rul_path)
    plt.close()

    metrics = {
        "MAE": mean_absolute_error(actual_soh, predicted_soh),
        "RMSE": np.sqrt(mean_squared_error(actual_soh, predicted_soh)),
        "R2": r2_score(actual_soh, predicted_soh)
    }

    return {
        "df": results_df,
        "max_capacity" : initial_capacity,
        "metrics": metrics,
        "images": {
            "soh_plot": soh_path,
            "rul_plot": rul_path
        }
    }
