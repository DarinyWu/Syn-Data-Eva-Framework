# WGAN-GP full pipeline
import torch
import torch.nn as nn
import torch.autograd as autograd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.special import softmax
from utils import *
# -------------------------------
# Critic (Discriminator in WGAN)
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
    cont_columns = ['conti_1', 'conti_2','...']
    scaler = MinMaxScaler()
    df[cont_columns] = scaler.fit_transform(df[cont_columns])
    x_cat = df[cat_columns].astype(np.float32).values
    x_cont = df[cont_columns].astype(np.float32).values
    data_tensor = torch.tensor(np.hstack([x_cat, x_cont]), dtype=torch.float32)
    return df, data_tensor, scaler, len(x_cat[0]), len(x_cont[0])

def split_dataset(data_tensor, num_teachers):
    subset_size = len(data_tensor) // num_teachers
    return [data_tensor[i * subset_size:(i + 1) * subset_size] for i in range(num_teachers)]

def train_teacher_discriminators(generator, teacher_sets, input_dim, hidden_dim, device):
    teachers = []
    for subset in teacher_sets:
        teacher = Critic(input_dim, hidden_dim).to(device)
        optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-4)

        loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)
        for epoch in range(5):  # or more
            for real in loader:
                real = real.to(device)
                bsz = real.size(0)
                z = torch.randn(bsz, generator.model[0].in_features).to(device)
                fake = generator(z).detach()

                loss = -teacher(real).mean() + teacher(fake).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        teachers.append(teacher)
    return teachers


def noisy_votes(teachers, samples, noise_scale=1.0):
    votes = []
    for t in teachers:
        with torch.no_grad():
            pred = t(samples).cpu().numpy().flatten()
            votes.append(pred > 0)  # 1 if "real", 0 if "fake"

    votes = np.array(votes)
    counts = votes.sum(axis=0)  # How many said "real"

    # Add Laplace noise for DP
    noisy_counts = counts + np.random.laplace(loc=0.0, scale=noise_scale, size=counts.shape)
    return (noisy_counts > (len(teachers) / 2)).astype(int)  # Final label: real if majority

def train_student_discriminator(student, samples, noisy_labels, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    X = samples.detach()
    y = torch.tensor(noisy_labels, dtype=torch.float32).unsqueeze(1)

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=128, shuffle=True)

    for epoch in range(5):  # or more
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = student(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_generator_against_student(generator, student, noise_dim, device):
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

    for _ in range(1):  # one step per student update
        z = torch.randn(128, noise_dim).to(device)
        fake = generator(z)
        preds = student(fake)

        loss = -preds.mean()  # WGAN-style

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

def run_pate_gan_pipeline(csv_path, noise_dim=64, hidden_dim=128, num_teachers=5, epochs=300, num_samples=1000, device='cpu'):
    # === Step 1: Preprocess data ===
    df, data_tensor, scaler, cat_dim, cont_dim = preprocess_split_data(csv_path)
    input_dim = cat_dim + cont_dim

    # === Step 2: Initialize models ===
    generator = Generator(noise_dim, input_dim, hidden_dim).to(device)
    student = Critic(input_dim, hidden_dim).to(device)

    # === Step 3: Partition data for teachers ===
    teacher_sets = split_dataset(data_tensor, num_teachers)

    for epoch in range(epochs):
        # === Step 4: Train teacher discriminators ===
        teachers = train_teacher_discriminators(generator, teacher_sets, input_dim, hidden_dim, device)

        # === Step 5: Generate synthetic samples ===
        z = torch.randn(512, noise_dim).to(device)
        fake_samples = generator(z)

        # === Step 6: Aggregate noisy votes ===
        labels = noisy_votes(teachers, fake_samples, noise_scale=1.0)

        # === Step 7: Train student discriminator on noisy labels ===
        train_student_discriminator(student, fake_samples, labels, device)

        # === Step 8: Train generator to fool student ===
        train_generator_against_student(generator, student, noise_dim, device)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} completed.")

    # # after training
    #torch.save(generator.state_dict(), "generator_pate_gan.pt")
    generator.load_state_dict(torch.load("generator_pate_gan.pt", map_location=device))
    generator.to(device)
    generator.eval()
    # === Step 9: Generate final synthetic data ===
    cat_dims = [len([c for c in df.columns if c.startswith(prefix + '_')]) for prefix in ['cat1', 'cat2', 'catn']]
    with torch.no_grad():
        z = torch.randn(num_samples, noise_dim).to(device)
        synth = generator(z).cpu().numpy()

    # === Step 10: Decode and return DataFrame ===
    df_synth = postprocess_generated(synth[:, :sum(cat_dims)], synth[:, sum(cat_dims):], cat_dims, scaler)
    return df_synth
#Customize your Usage