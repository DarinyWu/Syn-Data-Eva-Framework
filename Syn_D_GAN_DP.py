# WGAN-GP_DP full pipeline
import torch
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
import torch.nn as nn
import torch.autograd as autograd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.special import softmax
from utils import *
# -------------------------------
# Critic (Discriminator in WGAN_DP)
# -------------------------------
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

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
# Gradient Penalty Function
# -------------------------------
def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand_as(real_samples)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates)
    grad_outputs = torch.ones_like(d_interpolates)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

# -------------------------------
# Data Preprocessing
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
# Synthetic Generation
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
# Training Function with differential privacy
# -------------------------------

def train_wgan_dp(generator, critic, data_tensor, noise_dim, device, epochs=100, batch_size=128, clip_value=0.01, n_critic=5):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # Prepare two dataloaders: one for critic (private), one for generator
    sample_rate = batch_size / len(data_tensor)
    private_dataloader = torch.utils.data.DataLoader(
        dataset=data_tensor,
        batch_size=batch_size,
        sampler=UniformWithReplacementSampler(
            num_samples=len(data_tensor),
            sample_rate=sample_rate,
        ),
        drop_last=True
    )

    # Separate DataLoader for generator if needed
    generator_dataloader = torch.utils.data.DataLoader(
        dataset=data_tensor,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Wrap only critic with Opacus
    privacy_engine = PrivacyEngine()
    critic, d_optimizer, private_dataloader = privacy_engine.make_private(
        module=critic,
        optimizer=d_optimizer,
        data_loader=private_dataloader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
    )

    g_losses, d_losses = [], []

    for epoch in range(epochs):
        # === Train Critic ===
        for _ in range(n_critic):
            for real_data in private_dataloader:
                real_data = real_data.to(device)
                real_data.requires_grad_()
                bsz = real_data.size(0)

                z = torch.randn(bsz, noise_dim).to(device)
                fake_data = generator(z).detach()

                d_real = critic(real_data).mean()
                d_fake = critic(fake_data).mean()

                d_loss = -d_real + d_fake

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

                d_losses.append(d_loss.item())
                break  # only 1 batch per outer loop iteration (satisfy Opacus rules)

        # === Train Generator ===
        for real_data in generator_dataloader:
            real_data = real_data.to(device)
            bsz = real_data.size(0)

            z = torch.randn(bsz, noise_dim).to(device)
            fake_data = generator(z)
            g_loss = -critic(fake_data).mean()

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            g_losses.append(g_loss.item())
            break  # one batch per generator update

        # Track ε
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_losses[-1]:.4f} | G Loss: {g_losses[-1]:.4f} | ε: {epsilon:.2f}")

    # Plot losses
    plt.figure()
    plt.plot(d_losses, label='Critic (DP)')
    plt.plot(g_losses, label='Generator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('...')
    plt.legend()
    plt.tight_layout()
    plt.savefig("...")
    plt.close()

    return g_losses, d_losses

# -------------------------------
# Generate Synthetic Data
# -------------------------------
def generate_synthetic_data(generator, num_samples, noise_dim, cat_dims, scaler, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, noise_dim).to(device)
        synth = generator(z).cpu().numpy()
    return postprocess_generated(synth[:, :sum(cat_dims)], synth[:, sum(cat_dims):], cat_dims, scaler)

# -------------------------------
# Full WGAN-GP Pipeline
# -------------------------------
def run_wgan_dp_pipeline(csv_path, noise_dim=64, hidden_dim=128, num_samples=47283, epochs=300, device='cpu'):
    df, data_tensor, scaler, cat_dim, cont_dim = preprocess_split_data(csv_path)
    input_dim = cat_dim + cont_dim

    generator = Generator(noise_dim, input_dim, hidden_dim).to(device)
    critic = Critic(input_dim, hidden_dim).to(device)

    train_wgan_dp(generator, critic, data_tensor, noise_dim, device, epochs)

    cat_dims = [len([c for c in df.columns if c.startswith(prefix + '_')]) for prefix in ['cat1', 'cat2', 'catn']]
    df_synth = generate_synthetic_data(generator, num_samples, noise_dim, cat_dims, scaler, device)

    return df_synth


#Customize your Usage