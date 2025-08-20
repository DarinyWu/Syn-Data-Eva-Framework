import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from utils import *

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Define the Affine Coupling Layer and Flow
# -------------------------------
class AffineCoupling(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask):
        super(AffineCoupling, self).__init__()
        self.mask = mask
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        x_masked = x * self.mask
        s = self.scale_net(x_masked) * (1 - self.mask)
        t = self.translate_net(x_masked) * (1 - self.mask)
        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det_jacobian = torch.sum(s, dim=1)
        return y, log_det_jacobian

    def inverse(self, y):
        y_masked = y * self.mask
        s = self.scale_net(y_masked) * (1 - self.mask)
        t = self.translate_net(y_masked) * (1 - self.mask)
        x = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        log_det_jacobian = -torch.sum(s, dim=1)
        return x, log_det_jacobian

class NormalizingFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(NormalizingFlow, self).__init__()
        self.layers = nn.ModuleList()
        mask = self.create_mask(input_dim, even=True)
        for _ in range(num_layers):
            self.layers.append(AffineCoupling(input_dim, hidden_dim, mask))
            mask = 1 - mask
        self.base_mean = torch.zeros(input_dim).to(device)
        self.base_log_std = torch.zeros(input_dim).to(device)

    def create_mask(self, input_dim, even=True):
        mask = torch.zeros(input_dim)
        mask[::2] = 1 if even else 0
        mask[1::2] = 0 if even else 1
        return mask.to(device)

    def forward(self, x):
        log_det_jacobian = torch.zeros(x.size(0)).to(device)
        for layer in self.layers:
            x, ldj = layer(x)
            log_det_jacobian += ldj
        log_prob_z = self.base_log_prob(x)
        log_prob = log_prob_z + log_det_jacobian
        return log_prob

    def inverse(self, z):
        for layer in reversed(self.layers):
            z, _ = layer.inverse(z)
        return z

    def base_log_prob(self, z):
        return -0.5 * (((z - self.base_mean) / torch.exp(self.base_log_std)) ** 2 + 2 * self.base_log_std + np.log(2 * np.pi)).sum(1)

# -------------------------------
# Data Preprocessing
# -------------------------------
def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = pd.get_dummies(df, columns=['cat1', 'cat2', 'catn'])
    cat_columns = [col for col in df.columns if col.startswith('cat1_') or col.startswith('cat2_') or col.startswith('catn_')]
    cont_columns = ['conti_1', 'conti_2']
    scaler = MinMaxScaler()
    df[cont_columns] = scaler.fit_transform(df[cont_columns])
    x_cat = df[cat_columns].astype(np.float32).values
    x_cont = df[cont_columns].astype(np.float32).values
    data_tensor = torch.tensor(np.hstack([x_cat, x_cont]), dtype=torch.float32)
    return df, data_tensor, scaler

# -------------------------------
# Loss and Training
# -------------------------------
def compute_nf_loss(model, x):
    log_prob = model(x)
    loss = -torch.mean(log_prob)
    return loss

def train_nf(model, dataloader, optimizer, epochs, device):
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            loss = compute_nf_loss(model, x)
            if not torch.isfinite(loss):
                print("⚠️ Loss is not finite. Stopping training.")
                return
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        epoch_loss /= len(dataloader.dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}")

# -------------------------------
# Run Pipeline
# -------------------------------
def run_nf_pipeline(csv_path, hidden_dim=128, num_layers=6, epochs=100, batch_size=128, num_samples=1000):
    df, data_tensor, scaler = preprocess_data(csv_path)
    dataloader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)
    model = NormalizingFlow(input_dim=data_tensor.shape[1], hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_nf(model, dataloader, optimizer, epochs, device)

    # Generate synthetic data
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, data_tensor.shape[1]).to(device)
        x_gen = model.inverse(z).cpu().numpy()

    df_synth = pd.DataFrame(x_gen, columns=df.columns)
    df_synth[['conti_1', 'conti_2']] = scaler.inverse_transform(df_synth[['conti_1', 'conti_2']])

    # Decode one-hot back to original categorical values
    cat1_cols = [col for col in df_synth.columns if col.startswith("cat1_")]
    cat2_cols = [col for col in df_synth.columns if col.startswith("cat2_")]
    catn_cols = [col for col in df_synth.columns if col.startswith("catn_")]

    df_synth["cat1"] = df_synth[cat1_cols].idxmax(axis=1).str.extract("cat1_(\\d+)").astype(int)
    df_synth["cat2"] = df_synth[cat2_cols].idxmax(axis=1).str.extract("cat2_(\\d+)").astype(int)
    df_synth["catn"] = df_synth[catn_cols].idxmax(axis=1).str.extract("catn_(\\d+)").astype(int)

    df_final = df_synth[['cat1', 'cat2', 'catn','conti_1', 'conti_2','...']]
    return df_final

#Customize your Usage