import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

from models.model import SimpleNN
from utils.diffusion import prepare_diffusion


# -----------------------------
# Data Loading
# -----------------------------
def load_data(data_path, params):
    df = pd.read_csv(data_path, header=None, low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.drop(df.index[0])     # remove header row
    df.drop(0, axis=1, inplace=True)

    df = df.dropna().reset_index(drop=True)

    # 🔥 Drop cost coefficients
    n_g = params["n_g"]
    df = df.iloc[:, :-n_g]

    return df


# -----------------------------
# Normalization
# -----------------------------
def normalize_data(df, save_path="scaler.joblib"):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(df.values)

    joblib.dump(scaler, save_path)

    return torch.tensor(data, dtype=torch.float32)


# -----------------------------
# Loss
# -----------------------------
def diffusion_loss(model, x, t, diffusion_dict, device):
    from utils.diffusion import forward_diffusion_sample

    x_noisy, noise = forward_diffusion_sample(
        x, t, diffusion_dict, device
    )
    noise_pred = model(x_noisy, t)
    return nn.MSELoss()(noise, noise_pred)


# -----------------------------
# Training Loop
# -----------------------------
def train(model, dataset, diffusion_dict, args, device):

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=250, gamma=0.9)

    losses = []

    for epoch in range(args.epochs):
        for batch in dataloader:
            batch = batch.to(device)

            t = torch.randint(
                1,
                diffusion_dict["T"],
                (batch.shape[0],),
                device=device
            ).long()

            loss = diffusion_loss(model, batch, t, diffusion_dict, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        scheduler.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    return losses


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/IEEE118_Pd_Qd_Pg_Qg_train.csv")
    parser.add_argument("--config", type=str, default="configs/IEEE_118_Parameters.json")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_name", type=str, default="diffopf")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config
    with open(args.config) as f:
        params = json.load(f)["dims"]

    # Load data
    df = load_data(args.data_path, params)

    # Normalize
    dataset = normalize_data(df)

    # Prepare diffusion
    diffusion_dict = prepare_diffusion()

    # Model
    dim = 2*params["n_d"] + 2*params["n_g"]
    model = SimpleNN(dim, dim).to(device)

    # Train
    losses = train(model, dataset, diffusion_dict, args, device)

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    model_path = f"checkpoints/{args.save_name}.pth"
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")

    # Save loss
    np.save(f"checkpoints/loss_{args.save_name}.npy", np.array(losses))


if __name__ == "__main__":
    main()