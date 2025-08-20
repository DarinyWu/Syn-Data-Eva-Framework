# Tabular Diffusion Pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from utils import *
# -------------------------------
# Diffusion Hyperparameters
# -------------------------------
T = 1000  # number of diffusion steps
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1. - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# -------------------------------
# MLP Denoiser
# -------------------------------
class Denoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        t = t.float().view(-1, 1) / T  # [batch_size, 1], normalized
        xt = torch.cat([x, t], dim=1)  # shape: [batch_size, input_dim + 1]
        return self.net(xt)

# -------------------------------
# Forward Diffusion Process
# -------------------------------

def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    alpha_bars_device = alpha_bars.to(x_0.device)
    sqrt_alpha_bar = alpha_bars_device[t].sqrt().view(-1, 1)
    sqrt_one_minus = (1 - alpha_bars_device[t]).sqrt().view(-1, 1)
    return sqrt_alpha_bar * x_0 + sqrt_one_minus * noise

# -------------------------------
# Loss Function
# -------------------------------
def diffusion_loss(model, x_0, device):
    batch_size = x_0.shape[0]
    t = torch.randint(0, T, (batch_size,), device=device)
    noise = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, noise)
    predicted_noise = model(x_t, t)
    return F.mse_loss(predicted_noise, noise)

# -------------------------------
# Training Function
# -------------------------------
def train_diffusion(model, data_tensor, device, epochs=1000, batch_size=128):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    dataloader = torch.utils.data.DataLoader(data_tensor, batch_size=batch_size, shuffle=True)
    losses = []

    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.to(device)
            loss = diffusion_loss(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6f}")

    plt.plot(losses)
    plt.title("Diffusion Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("diffusion_loss.png")
    plt.close()

def enforce_onehot(tensor, df_template):
    # Infer one-hot column groups from template
    col_names = df_template.columns
    groups = {
        'cat1': [i for i, c in enumerate(col_names) if c.startswith('cat1_')],
        'cat2': [i for i, c in enumerate(col_names) if c.startswith('cat2_')],
        'catn': [i for i, c in enumerate(col_names) if c.startswith('catn_')],
    }

    for group_cols in groups.values():
        soft = tensor[:, group_cols]
        max_idx = soft.argmax(dim=1, keepdim=True)
        onehot = torch.zeros_like(soft).scatter_(1, max_idx, 1.0)
        tensor[:, group_cols] = onehot

    return tensor

# -------------------------------
# Sampling Function
# -------------------------------
def sample_diffusion(model, num_samples, input_dim, device, df_template):
    x_t = torch.randn(num_samples, input_dim).to(device)
    model.eval()

    with torch.no_grad():
        for t in reversed(range(T)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = model(x_t, t_tensor)

            alpha = alphas[t].to(device)
            alpha_bar = alpha_bars[t].to(device)
            beta = betas[t].to(device)

            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            x_t = (1 / alpha.sqrt()) * (
                    x_t - ((1 - alpha) / (1 - alpha_bar).sqrt()) * noise_pred
            ) + beta.sqrt() * noise

        # Enforce one-hot structure for categorical parts (optional but helpful)
        x_t = enforce_onehot(x_t, df_template)

    return x_t.cpu().numpy()


# -------------------------------
# Preprocessing
# -------------------------------

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = pd.get_dummies(df, columns=['cat1', 'cat2', 'catn'])
    cat_columns = [col for col in df.columns if col.startswith('cat1_') or col.startswith('cat2_') or col.startswith('catn_')]
    cont_columns = ['conti_1', 'conti_2','...']
    scaler = MinMaxScaler()
    df[cont_columns] = scaler.fit_transform(df[cont_columns])
    x_cat = df[cat_columns].astype(np.float32).values
    x_cont = df[cont_columns].astype(np.float32).values
    data_tensor = torch.tensor(np.hstack([x_cat, x_cont]), dtype=torch.float32)
    return df, data_tensor, scaler

# -------------------------------
# Main Run Function
# -------------------------------
def run_diffusion_pipeline(csv_path, num_samples, epochs, device='cpu'):
    df, data_tensor, scaler = preprocess_data(csv_path)
    input_dim = data_tensor.shape[1]

    model = Denoiser(input_dim=input_dim).to(device)
    train_diffusion(model, data_tensor, device, epochs=epochs)

    samples = sample_diffusion(model, num_samples, input_dim, device, df)
    df_sample = pd.DataFrame(samples, columns=df.columns)

    # Inverse transform continuous variables
    cont_cols = ['conti_1', 'conti_2','...']
    df_sample[cont_cols] = scaler.inverse_transform(df_sample[cont_cols])
    df_sample[cont_cols] = df_sample[cont_cols].astype(int)

    # Decode categorical one-hots
    for prefix in ['cat1', 'cat2', 'catn']:
        cols = [col for col in df.columns if col.startswith(prefix + '_')]
        df_sample[prefix] = df_sample[cols].values.argmax(axis=1)
        df_sample.drop(columns=cols, inplace=True)

    return df_sample

# Customize your Usage:
