import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "human_promoters.fasta")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "clean_promoters.txt")

def parse_and_filter_promoters(fasta_path, max_n_percent=5.0):
    print(f"Reading from: {fasta_path}")
    
    sequences = []
    current_seq = []

    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("<", "!", "#", ";")):
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq).upper())
                    current_seq = []
            else:
                clean_line = "".join([c for c in line.upper() if c in "ACGTN"])
                if clean_line:
                    current_seq.append(clean_line)

        if current_seq:
            sequences.append("".join(current_seq).upper())

    print(f"Total raw sequences found: {len(sequences)}")

    # Filter out high-N sequences
    clean_sequences = []
    for seq in sequences:
        if len(seq) == 0:
            continue
        n_count = seq.count('N')
        n_pct = (n_count / len(seq)) * 100
        
        # Keep sequence only if N content is below threshold
        if n_pct <= max_n_percent:
            clean_sequences.append(seq)

    print(f"Clean sequences remaining (<= {max_n_percent}% 'N's): {len(clean_sequences)}")

    # Save clean sequences
    with open(OUTPUT_PATH, "w") as f:
        for seq in clean_sequences:
            f.write(seq + "\n")

    print(f"Saved clean dataset to: {OUTPUT_PATH}")
    
    if len(clean_sequences) > 0:
        print("\n--- SAMPLE CLEAN SEQUENCE (First 100 bp) ---")
        print(clean_sequences[0][:100] + "...")

if __name__ == "__main__":
    parse_and_filter_promoters(DATA_PATH)