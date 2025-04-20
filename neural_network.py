import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import kagglehub

path = kagglehub.dataset_download("fedesoriano/body-fat-prediction-dataset")

print("Path to dataset files:", path)

csv_file = os.path.join(path, "bodyfat.csv")
df = pd.read_csv(csv_file)

# Prepare lists to store metrics
train_losses = []
val_losses = []
# Engineered features
df["BMI"] = df["Weight"] / (df["Height"] ** 2) * 703
df["WaistHipRatio"] = df["Abdomen"] / df["Hip"]

# Feature selection
selected_features = ["Wrist", "BMI", "WaistHipRatio", "Neck", "Forearm", "Thigh", "Hip", "Biceps", "Ankle"]
X = df[selected_features].values.astype(np.float32)
y = df["BodyFat"].values.astype(np.float32).reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create DataLoaders
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

# Define model
class BodyFatNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

model = BodyFatNN(input_dim=X.shape[1])

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    model.train()
    epoch_train_loss = 0.0
    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_test))
        val_loss = criterion(val_preds, torch.tensor(y_test))
        val_losses.append(val_loss.item())

    # Print every 10 epochs
    if epoch % 10 == 0 or epoch == 99:
        val_mae = torch.mean(torch.abs(val_preds - torch.tensor(y_test)))
        print(f"Epoch {epoch+1}/100 | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss.item():.4f} | Val MAE: {val_mae.item():.4f}")
