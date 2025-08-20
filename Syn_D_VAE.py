import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax
from utils import *
import matplotlib.pyplot as plt
# -------------------------------
# VAE with Split Decoder
# -------------------------------
class VAE(nn.Module):
    def __init__(self, z_dim, hidden_dim, cat_dim, cont_dim):
        super(VAE, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.cat_decoder = nn.Linear(hidden_dim, cat_dim)  # no activation (logits)
        self.cont_decoder = nn.Sequential(
            nn.Linear(hidden_dim, cont_dim),
            nn.Sigmoid()  # for [0,1] continuous features
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(cat_dim + cont_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, z_dim)
        self.log_var_layer = nn.Linear(hidden_dim, z_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        log_var = torch.clamp(self.log_var_layer(h), min=-10, max=300)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.shared(z)
        return self.cat_decoder(h), self.cont_decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# -------------------------------
# VAE Loss
# -------------------------------
def compute_loss(x_cat, x_cont, cat_out, cont_out, mu, log_var, beta=0.0001):
    # For continuous features
    recon_loss_cont = nn.functional.mse_loss(cont_out, x_cont, reduction='sum') / x_cont.size(0)

    # For categorical (using BCE on softmaxed logits)
    probs = torch.softmax(cat_out, dim=1)
    recon_loss_cat = -torch.sum(x_cat * torch.log(probs + 1e-9)) / x_cat.size(0)

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x_cat.size(0)

    return recon_loss_cat + recon_loss_cont + beta * kl_loss, recon_loss_cat, recon_loss_cont, kl_loss

# -------------------------------
# Postprocess Function
# -------------------------------
def postprocess_generated(cat_logits, cont_output, cat_dims, scaler):
    decoded = {}
    start = 0
    col_names = ['cat1', 'cat2', 'catn']

    for i, dim in enumerate(cat_dims):
        block = softmax(cat_logits[:, start:start+dim], axis=1)
        decoded[col_names[i]] = np.argmax(block, axis=1)
        start += dim

    cont_block = scaler.inverse_transform(cont_output)
    decoded['conti_1'] = cont_block[:, 0].astype(int)
    decoded['conti_2'] = cont_block[:, 1].astype(int)

    return pd.DataFrame(decoded)

def train_vae(model, data_tensor, cat_dim, cont_dim, device, epochs=30):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_losses, recon_cat_losses, recon_cont_losses, kl_losses = [], [], [], []

    for epoch in range(epochs):
        model.train()
        x = data_tensor.to(device)
        x_cat = x[:, :cat_dim]
        x_cont = x[:, cat_dim:]

        (cat_out, cont_out), mu, log_var = model(x)
        loss, loss_cat, loss_cont, kl = compute_loss(x_cat, x_cont, cat_out, cont_out, mu, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store losses
        total_losses.append(loss.item())
        recon_cat_losses.append(loss_cat.item())
        recon_cont_losses.append(loss_cont.item())
        kl_losses.append(kl.item())

        print(f"Epoch [{epoch+1}/{epochs}] | Total: {loss.item():.2f} | Cat: {loss_cat.item():.2f} | Cont: {loss_cont.item():.2f} | KL: {kl.item():.2f}")

    loss_log = pd.DataFrame({
        'epoch': list(range(1, epochs + 1)),
        'total_loss': total_losses,
        'recon_cat_loss': recon_cat_losses,
        'recon_cont_loss': recon_cont_losses,
        'kl_loss': kl_losses
    })
    loss_log.to_csv("vae_training_log.csv", index=False)
    # Plot and save loss
    plt.figure(figsize=(10, 6))
    plt.plot(total_losses, label='Total Loss')
    # plt.plot(recon_cat_losses, label='Cat Recon Loss')
    # plt.plot(recon_cont_losses, label='Cont Recon Loss')
    # plt.plot(kl_losses, label='KL Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("vae_training_loss.png")
    plt.close()


def preprocess_split_data(filepath):
    df = pd.read_csv(filepath)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['cat1', 'cat2', 'catn'])
    cat_columns = [col for col in df.columns if col.startswith('cat1_') or col.startswith('cat2_') or col.startswith('catn_')]
    cont_columns = ['conti_1', 'conti_2']

    # Normalize continuous
    scaler = MinMaxScaler()
    df[cont_columns] = scaler.fit_transform(df[cont_columns])

    x_cat = df[cat_columns].astype(np.float32).values
    x_cont = df[cont_columns].astype(np.float32).values
    data_tensor = torch.tensor(np.hstack([x_cat, x_cont]), dtype=torch.float32)

    return df, data_tensor, scaler, len(x_cat[0]), len(x_cont[0])

def generate_synthetic_data(model, num_samples, z_dim, cat_dims, scaler, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim).to(device)
        cat_logits, cont_out = model.decode(z)
        cat_logits = cat_logits.cpu().numpy()
        cont_out = cont_out.cpu().numpy()
    return postprocess_generated(cat_logits, cont_out, cat_dims, scaler)


def run_vae_pipeline(csv_path, z_dim=32, hidden_dim=128, num_samples=10000, epochs=300, device='cpu'):
    df, data_tensor, scaler, cat_dim, cont_dim = preprocess_split_data(csv_path)
    model = VAE(z_dim=z_dim, hidden_dim=hidden_dim, cat_dim=cat_dim, cont_dim=cont_dim)

    train_vae(model, data_tensor, cat_dim, cont_dim, device=device, epochs=epochs)

    # Save the model
    torch.save(model.state_dict(), "model_vae.pt")

    # optional load the trained model
    # model.load_state_dict(torch.load("vae_model.pt"))
    # model.to(device)

    cat_dims = [len([c for c in df.columns if c.startswith(prefix + '_')]) for prefix in ['cat1', 'cat2', 'catn']]
    df_synth = generate_synthetic_data(model, num_samples=num_samples, z_dim=z_dim, cat_dims=cat_dims, scaler=scaler, device=device)
    return df_synth

#Customize your Usage


