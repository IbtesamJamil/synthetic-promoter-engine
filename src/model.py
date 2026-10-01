import torch
import torch.nn as nn

class PromoterGenerator(nn.Module):
    """
    Generator Network: Maps a 100-dimensional random latent vector z
    to a 4x600 one-hot DNA sequence representation using 1D Transposed Convolutions.
    """
    def __init__(self, latent_dim: int = 100, seq_length: int = 600, num_channels: int = 4):
        super(PromoterGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.num_channels = num_channels

        # Project and reshape latent vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 75),
            nn.BatchNorm1d(128 * 75),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsample sequence length: 75 -> 150 -> 300 -> 600
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 150)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # (B, 32, 300)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(32, num_channels, kernel_size=4, stride=2, padding=1), # (B, 4, 600)
            nn.Softmax(dim=1)  # Produce base probabilities across A, C, G, T channels
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fc(z)
        out = out.view(-1, 128, 75)
        sequence = self.conv_blocks(out)
        return sequence


class PromoterCritic(nn.Module):
    """
    Critic (Discriminator) Network: Evaluates 4x600 one-hot DNA sequences
    and scores their realism using 1D Convolutions and LayerNorm (required for WGAN-GP).
    """
    def __init__(self, seq_length: int = 600, num_channels: int = 4):
        super(PromoterCritic, self).__init__()

        self.model = nn.Sequential(
            # Input: (B, 4, 600) -> (B, 32, 300)
            nn.Conv1d(num_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 32, 300) -> (B, 64, 150)
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, 64),  # LayerNorm equivalent for 1D CNNs
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 64, 150) -> (B, 128, 75)
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 75, 1)  # Output scalar score
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.model(sequence)


if __name__ == "__main__":
    # Smoke test model shapes
    batch_size = 16
    latent_dim = 100
    
    generator = PromoterGenerator(latent_dim=latent_dim)
    critic = PromoterCritic()

    z = torch.randn(batch_size, latent_dim)
    fake_dna = generator(z)
    scores = critic(fake_dna)

    print("Model Architecture Test:")
    print(f"  Latent Input Shape  : {z.shape}")
    print(f"  Generated DNA Shape : {fake_dna.shape}  (Expect: [16, 4, 600])")
    print(f"  Critic Scores Shape : {scores.shape}    (Expect: [16, 1])")