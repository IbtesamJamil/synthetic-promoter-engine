import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REAL_PATH = os.path.join(PROJECT_ROOT, "data", "clean_promoters.txt")
SYNTH_PATH = os.path.join(PROJECT_ROOT, "data", "generated_promoters.fasta")

def main():
    if not os.path.exists(REAL_PATH) or not os.path.exists(SYNTH_PATH):
        raise FileNotFoundError("Data files missing. Run parse_data.py and train.py first.")

    with open(REAL_PATH, "r", encoding="utf-8") as f:
        real_seqs = [line.strip().upper() for line in f if line.strip()]

    with open(SYNTH_PATH, "r", encoding="utf-8") as f:
        synth_seqs = [line.strip().upper() for line in f if line.strip() and not line.startswith(">")]

    real_gc = sum((s.count('G') + s.count('C')) / len(s) for s in real_seqs) / len(real_seqs) * 100
    synth_gc = sum((s.count('G') + s.count('C')) / len(s) for s in synth_seqs) / len(synth_seqs) * 100

    real_tata = sum(s.count("TATAAA") for s in real_seqs) / len(real_seqs)
    synth_tata = sum(s.count("TATAAA") for s in synth_seqs) / len(synth_seqs)

    real_gcbox = sum(s.count("GGGCGG") for s in real_seqs) / len(real_seqs)
    synth_gcbox = sum(s.count("GGGCGG") for s in synth_seqs) / len(synth_seqs)

    print("=" * 60)
    print("      SYNTHETIC PROMOTER BIOLOGICAL EVALUATION SUITE      ")
    print("=" * 60)
    print(f"Dataset Size        : {len(real_seqs)} Real | {len(synth_seqs)} Generated")
    print(f"Real Avg GC%        : {real_gc:.2f}%")
    print(f"Synthetic Avg GC%   : {synth_gc:.2f}% (Delta: {abs(real_gc - synth_gc):.2f}%)")
    print("-" * 60)
    print(f"TATA-Box (TATAAA)   : Real = {real_tata:.3f}/seq | Synth = {synth_tata:.3f}/seq")
    print(f"GC-Box   (GGGCGG)   : Real = {real_gcbox:.3f}/seq | Synth = {synth_gcbox:.3f}/seq")
    print("=" * 60)

if __name__ == "__main__":
    main()