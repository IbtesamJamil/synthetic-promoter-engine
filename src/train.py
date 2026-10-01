import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import PromoterGenerator, PromoterCritic

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "promoters_onehot.pt")
OUTPUT_FASTA = os.path.join(PROJECT_ROOT, "data", "generated_promoters.fasta")

# Hyperparameters
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 20          # Increase for production runs
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10.0     # Gradient penalty coefficient
LR = 0.0002

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty for WGAN-GP."""
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    fake = torch.ones(real_samples.size(0), 1, device=device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA_GP
    return gradient_penalty

def one_hot_to_fasta(one_hot_tensor):
    """Converts a (4, 600) probability tensor into a DNA string."""
    bases = ['A', 'C', 'G', 'T']
    indices = torch.argmax(one_hot_tensor, dim=0)  # Pick highest probability base per position
    return ''.join([bases[idx] for idx in indices])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    # Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"One-hot tensor file not found at: {DATA_PATH}")

    real_data = torch.load(DATA_PATH)
    dataset = TensorDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize models
    generator = PromoterGenerator(latent_dim=LATENT_DIM).to(device)
    critic = PromoterCritic().to(device)

    opt_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.9))
    opt_c = optim.Adam(critic.parameters(), lr=LR, betas=(0.5, 0.9))

    print("\nStarting WGAN-GP Training Loop...\n")

    for epoch in range(1, EPOCHS + 1):
        for i, (real_batch,) in enumerate(dataloader):
            real_batch = real_batch.to(device)
            current_batch_size = real_batch.size(0)

            # ---------------------
            #  Train Critic
            # ---------------------
            for _ in range(CRITIC_ITERATIONS):
                z = torch.randn(current_batch_size, LATENT_DIM, device=device)
                fake_batch = generator(z)

                real_score = critic(real_batch)
                fake_score = critic(fake_batch.detach())

                gp = compute_gradient_penalty(critic, real_batch, fake_batch.detach(), device)
                c_loss = -(torch.mean(real_score) - torch.mean(fake_score)) + gp

                opt_c.zero_grad()
                c_loss.backward()
                opt_c.step()

            # ---------------------
            #  Train Generator
            # ---------------------
            z = torch.randn(current_batch_size, LATENT_DIM, device=device)
            gen_batch = generator(z)
            g_loss = -torch.mean(critic(gen_batch))

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        w_distance = (torch.mean(real_score) - torch.mean(fake_score)).item()
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Critic Loss: {c_loss.item():.4f} | Gen Loss: {g_loss.item():.4f} | W-Distance: {w_distance:.4f}")

    # ---------------------
    #  Export Generated DNA
    # ---------------------
    print("\nGenerating candidate synthetic promoter sequences...")
    generator.eval()
    with torch.no_grad():
        sample_z = torch.randn(10, LATENT_DIM, device=device)
        synthetic_tensors = generator(sample_z)

    with open(OUTPUT_FASTA, "w", encoding="utf-8") as f:
        for idx, tensor in enumerate(synthetic_tensors, 1):
            dna_seq = one_hot_to_fasta(tensor)
            f.write(f">synthetic_promoter_{idx}_WGAN_GP\n{dna_seq}\n")

    print(f"\nSUCCESS: Exported 10 synthetic sequences to:\n  {OUTPUT_FASTA}")

if __name__ == "__main__":
    main()