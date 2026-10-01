import os
import torch
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "clean_promoters.txt")
OUTPUT_ONE_HOT_PATH = os.path.join(PROJECT_ROOT, "data", "promoters_onehot.pt")
OUTPUT_KMER_PATH = os.path.join(PROJECT_ROOT, "data", "promoters_kmer.pt")

NUCLEOTIDE_MAP = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0.25, 0.25, 0.25, 0.25]
}

def sequence_to_one_hot(seq: str) -> torch.Tensor:
    one_hot_list = [NUCLEOTIDE_MAP.get(base, NUCLEOTIDE_MAP['N']) for base in seq.upper()]
    return torch.tensor(one_hot_list, dtype=torch.float32).transpose(0, 1)

def sequence_to_kmers(seq: str, k: int = 6) -> list[str]:
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]

def build_kmer_vocab(sequences: list[str], k: int = 6) -> dict[str, int]:
    unique_kmers = set()
    for seq in sequences:
        unique_kmers.update(sequence_to_kmers(seq, k=k))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, kmer in enumerate(sorted(unique_kmers), start=2):
        vocab[kmer] = idx
    return vocab

def kmers_to_ids(kmers: list[str], vocab: dict[str, int]) -> torch.Tensor:
    ids = [vocab.get(kmer, vocab["<UNK>"]) for kmer in kmers]
    return torch.tensor(ids, dtype=torch.long)

class PromoterDataset(Dataset):
    def __init__(self, one_hot_tensors: torch.Tensor, kmer_tensors: torch.Tensor = None):
        self.one_hot_tensors = one_hot_tensors
        self.kmer_tensors = kmer_tensors

    def __len__(self):
        return len(self.one_hot_tensors)

    def __getitem__(self, idx):
        if self.kmer_tensors is not None:
            return {
                "one_hot": self.one_hot_tensors[idx],
                "kmer_ids": self.kmer_tensors[idx]
            }
        return self.one_hot_tensors[idx]

def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Clean promoters dataset not found at: {INPUT_PATH}\nRun parse_data.py first.")

    print(f"Reading cleaned sequences from: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        sequences = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(sequences)} sequences.")

    print("\n1. Generating One-Hot Tensors (Shape: N x 4 x 600)...")
    one_hot_list = [sequence_to_one_hot(seq) for seq in sequences]
    one_hot_tensor = torch.stack(one_hot_list)
    
    torch.save(one_hot_tensor, OUTPUT_ONE_HOT_PATH)
    print(f"   Saved One-Hot Tensor: {one_hot_tensor.shape} to {OUTPUT_ONE_HOT_PATH}")

    K_SIZE = 6
    print(f"\n2. Generating {K_SIZE}-mer Token Representations...")
    vocab = build_kmer_vocab(sequences, k=K_SIZE)
    print(f"   Vocabulary Size ({K_SIZE}-mers): {len(vocab):,} unique tokens.")

    kmer_id_list = []
    for seq in sequences:
        kmers = sequence_to_kmers(seq, k=K_SIZE)
        ids = kmers_to_ids(kmers, vocab)
        kmer_id_list.append(ids)

    kmer_tensor = torch.stack(kmer_id_list)
    
    kmer_data = {
        "tensors": kmer_tensor,
        "vocab": vocab,
        "k": K_SIZE
    }
    torch.save(kmer_data, OUTPUT_KMER_PATH)
    print(f"   Saved K-Mer Tensors: {kmer_tensor.shape} to {OUTPUT_KMER_PATH}")

    print("\n3. Testing PyTorch DataLoader batching...")
    dataset = PromoterDataset(one_hot_tensor, kmer_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    sample_batch = next(iter(loader))
    print(f"   Batch One-Hot Shape : {sample_batch['one_hot'].shape}")
    print(f"   Batch K-Mer Shape   : {sample_batch['kmer_ids'].shape}")

    print("\nSUCCESS: Tokenization and tensor preparation complete!")

if __name__ == "__main__":
    main()