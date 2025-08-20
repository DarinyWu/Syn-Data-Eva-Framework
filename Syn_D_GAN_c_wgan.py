# Conditional WGAN-GP pipeline
import torch
import torch.nn as nn
import torch.autograd as autograd
import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from utils import *
# -------------------------------
# Critic (Discriminator in WGAN)
# -------------------------------
class Critic(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, c):
        x_cond = torch.cat([x, c], dim=1)
        return self.model(x_cond)

# -------------------------------
# Generator
# -------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, output_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z, c):
        z_cond = torch.cat([z, c], dim=1)
        return self.model(z_cond)

# -------------------------------
# Gradient Penalty Function
# -------------------------------
def compute_gradient_penalty(critic, real_samples, fake_samples, cond, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand_as(real_samples)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates, cond)
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
    return df, data_tensor, scaler, len(x_cat[0]), len(x_cont[0]), len([c for c in df.columns if c.startswith('._')])

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
# Training Function
# -------------------------------
def train_cwgan_gp(generator, critic, data_tensor, week_tensor, noise_dim, cond_dim, device, epochs=100, batch_size=128, lambda_gp=10, n_critic=5):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

    g_losses, d_losses = [], []
    dataset = torch.utils.data.TensorDataset(data_tensor, week_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for i, (real_data, cond) in enumerate(data_loader):
            real_data = real_data.to(device)
            cond = cond.to(device)
            batch_size = real_data.size(0)

            # Train Critic
            for _ in range(n_critic):
                z = torch.randn(batch_size, noise_dim).to(device)
                fake_data = generator(z, cond).detach()

                d_real = critic(real_data, cond).mean()
                d_fake = critic(fake_data, cond).mean()
                gp = compute_gradient_penalty(critic, real_data, fake_data, cond, device)
                d_loss = -d_real + d_fake + lambda_gp * gp

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            # Train Generator
            z = torch.randn(batch_size, noise_dim).to(device)
            fake_data = generator(z, cond)
            g_loss = -critic(fake_data, cond).mean()

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    plt.figure()
    plt.plot(d_losses, label='Critic')
    plt.plot(g_losses, label='Generator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('...')
    plt.tight_layout()
    plt.savefig("...")
    plt.close()

    return g_losses, d_losses

# -------------------------------
# Generate Synthetic Data
# -------------------------------
def generate_conditioned_data(generator, week_val, num_samples, noise_dim, cat_dims, scaler, cond_dim, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, noise_dim).to(device)
        c = torch.zeros(num_samples, cond_dim).to(device)
        c[:, week_val] = 1.0  # one-hot for specified week
        synth = generator(z, c).cpu().numpy()
    return postprocess_generated(synth[:, :sum(cat_dims)], synth[:, sum(cat_dims):], cat_dims, scaler)

# -------------------------------
# Full Conditional WGAN-GP Pipeline
# -------------------------------
def run_cwgan_gp_pipeline(csv_path, noise_dim=64, hidden_dim=128, num_samples=6600, epochs=300, device='cpu'):
    df, data_tensor, scaler, cat_dim, cont_dim, cond_dim = preprocess_split_data(csv_path)
    input_dim = cat_dim + cont_dim

    # Extract only the week one-hot columns for conditioning
    week_tensor = torch.tensor(df[[c for c in df.columns if c.startswith('._')]].values, dtype=torch.float32)

    generator = Generator(noise_dim, cond_dim, input_dim, hidden_dim).to(device)
    #after training
    generator.load_state_dict(torch.load("..pt", map_location=device))
    generator.to(device)
    # generator.eval()
    #critic = Critic(input_dim, cond_dim, hidden_dim).to(device)

    #train_cwgan_gp(generator, critic, data_tensor, week_tensor, noise_dim, cond_dim, device, epochs)

    #torch.save(generator.state_dict(), "cwgan_generator.pt")

    cat_dims = [len([c for c in df.columns if c.startswith(prefix + '_')]) for prefix in ['.', '.', '.']]
    df_synth = []
    for week in range(7):
        df_week = generate_conditioned_data(
            generator,
            week_val=week,
            num_samples=num_samples,
            noise_dim=noise_dim,
            cat_dims=cat_dims,
            scaler=scaler,
            cond_dim=cond_dim,
            device=device
        )
        df_synth.append(df_week)
    df_syn = pd.concat(df_synth, ignore_index=True)

    return df_syn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Customize your Usage


