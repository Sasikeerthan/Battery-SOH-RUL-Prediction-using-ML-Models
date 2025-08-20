import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

class TransformerSOHPredictor(nn.Module):
    def __init__(self, input_dim=1, seq_len=5, model_dim=64, num_heads=4, num_layers=2):
        super(TransformerSOHPredictor, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * model_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        return self.regressor(x)

def run_transformer(df):
    model_name = "Transformer"

    initial_capacity = df['Capacity (Ah)'][:5].max()
    df['SOH (%)'] = (df['Capacity (Ah)'] / initial_capacity) * 100

    def create_sequences(data, seq_len=5):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)

    SEQ_LEN = 5
    soh_values = df['SOH (%)'].values
    scaler = MinMaxScaler()
    soh_norm = scaler.fit_transform(soh_values.reshape(-1, 1)).flatten()
    X, y = create_sequences(soh_norm, SEQ_LEN)

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(2)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TransformerSOHPredictor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor).squeeze()
        loss = criterion(output, y_tensor.squeeze())
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted = model(X_tensor).squeeze().cpu().numpy()
        actual = y_tensor.squeeze().cpu().numpy()

    soh_min, soh_max = y_tensor.min().item(), y_tensor.max().item()
    predicted_soh = predicted * (soh_max - soh_min) + soh_min
    actual_soh = actual * (soh_max - soh_min) + soh_min

    threshold = 0.8 * (soh_max - soh_min) + soh_min
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
        'SOH (%)': actual_soh * 100,
        'Predicted SOH (%)': predicted_soh * 100,
        'Predicted RUL (cycles)': rul
    })

    # Save plots
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
        "max_capacity": initial_capacity,
        "metrics": metrics,
        "images": {
            "soh_plot": soh_path,
            "rul_plot": rul_path
        }
    }
