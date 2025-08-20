import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax
import matplotlib.pyplot as plt
from utils import *
# -------------------------------
# Generator
# -------------------------------
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.model(z)

# -------------------------------
# Discriminator
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_split_data(filepath):
    df = pd.read_csv(filepath)
    df = pd.get_dummies(df, columns=['cat1', 'cat2', 'catn'])
    cat_columns = [col for col in df.columns if col.startswith('cat1_') or col.startswith('cat2_') or col.startswith('catn_')]
    cont_columns = ['conti_1', 'conti_2']
    scaler = MinMaxScaler()
    df[cont_columns] = scaler.fit_transform(df[cont_columns])
    x_cat = df[cat_columns].astype(np.float32).values
    x_cont = df[cont_columns].astype(np.float32).values
    data_tensor = torch.tensor(np.hstack([x_cat, x_cont]), dtype=torch.float32)
    return df, data_tensor, scaler, len(x_cat[0]), len(x_cont[0])

# -------------------------------
# Postprocessing
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

# -------------------------------
# GAN Training
# -------------------------------
def train_gan(generator, discriminator, data_tensor, noise_dim, device, epochs=300, batch_size=128):
    generator.to(device)
    discriminator.to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    criterion = nn.BCELoss()

    losses = {'G': [], 'D': []}
    real_label, fake_label = 1.0, 0.0
    dataset = torch.utils.data.DataLoader(data_tensor, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        d_loss_total = 0
        g_loss_total = 0
        for real_data in dataset:
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # Train Discriminator
            z = torch.randn(batch_size, noise_dim).to(device)
            fake_data = generator(z)

            d_real = discriminator(real_data)
            d_fake = discriminator(fake_data.detach())

            d_loss_real = criterion(d_real, torch.ones_like(d_real))
            d_loss_fake = criterion(d_fake, torch.zeros_like(d_fake))
            d_loss = d_loss_real + d_loss_fake

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            z = torch.randn(batch_size, noise_dim).to(device)
            fake_data = generator(z)
            d_fake = discriminator(fake_data)
            g_loss = criterion(d_fake, torch.ones_like(d_fake))

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()

        losses['D'].append(d_loss_total / len(dataset))
        losses['G'].append(g_loss_total / len(dataset))
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss_total:.4f} | G Loss: {g_loss_total:.4f}")

    # Save plot
    plt.figure()
    plt.plot(losses['D'], label='Discriminator')
    plt.plot(losses['G'], label='Generator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    plt.tight_layout()
    plt.savefig("....")
    plt.close()

    # Save log if you want
    pd.DataFrame(losses).to_csv("...", index=False)

    return losses

# -------------------------------
# Generate Synthetic Data
# -------------------------------
def generate_synthetic_data_gan(generator, num_samples, noise_dim, cat_dims, scaler, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, noise_dim).to(device)
        synth = generator(z).cpu().numpy()
    return postprocess_generated(synth[:, :sum(cat_dims)], synth[:, sum(cat_dims):], cat_dims, scaler)

# -------------------------------
# Full GAN Pipeline
# -------------------------------
def run_gan_pipeline(csv_path, noise_dim=64, hidden_dim=128, num_samples=47283, epochs=300, device='cpu'):
    df, data_tensor, scaler, cat_dim, cont_dim = preprocess_split_data(csv_path)
    input_dim = cat_dim + cont_dim

    generator = Generator(input_dim=noise_dim, output_dim=input_dim, hidden_dim=hidden_dim)
    discriminator = Discriminator(input_dim=input_dim, hidden_dim=hidden_dim)

    train_gan(generator, discriminator, data_tensor, noise_dim, device=device, epochs=epochs)

    cat_dims = [len([c for c in df.columns if c.startswith(prefix + '_')]) for prefix in ['week', 'o', 'd']]
    df_synth = generate_synthetic_data_gan(generator, num_samples=num_samples, noise_dim=noise_dim, cat_dims=cat_dims, scaler=scaler, device=device)

    return df_synth

#Customize your Usage