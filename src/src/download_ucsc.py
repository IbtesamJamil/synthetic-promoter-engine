import json
import os
import ssl
import urllib.parse
import urllib.request

# Get paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "human_promoters.fasta")

ctx = ssl._create_unverified_context()
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

print("1. Querying UCSC Genome Browser API (hg38 RefSeq)...")

# Clean URL parameters
params = urllib.parse.urlencode(
    {"genome": "hg38", "track": "refSeqComposite", "maxItemsOutput": "3000"}
)
track_url = f"https://api.genome.ucsc.edu/getData/track?{params}"

req = urllib.request.Request(track_url, headers=headers)

try:
    with urllib.request.urlopen(req, context=ctx) as res:
        data = json.loads(res.read().decode("utf-8"))

    genes = data.get("refSeqComposite", [])
    print(
        f"   Found {len(genes)} gene regions. Fetching promoter sequences...\n"
    )

    fasta_records = []
    seen = set()

    for g in genes:
        gene_name = g.get("name", "")
        if not gene_name or gene_name in seen:
            continue
        seen.add(gene_name)

        chrom = g["chrom"]
        strand = g["strand"]

        # 600 bp promoter [-499, +100]
        if strand == "+":
            tss = g["txStart"]
            start = max(0, tss - 499)
            end = tss + 101
        else:
            tss = g["txEnd"]
            start = max(0, tss - 100)
            end = tss + 500

        seq_params = urllib.parse.urlencode(
            {"genome": "hg38", "chrom": chrom, "start": str(start), "end": str(end)}
        )
        seq_url = f"https://api.genome.ucsc.edu/getData/sequence?{seq_params}"
        seq_req = urllib.request.Request(seq_url, headers=headers)

        try:
            with urllib.request.urlopen(seq_req, context=ctx) as s_res:
                s_data = json.loads(s_res.read().decode("utf-8"))
                seq = s_data.get("dna", "").upper()

                if len(seq) == 600:
                    fasta_records.append(
                        f">{gene_name}_{chrom}:{start}-{end}({strand})\n{seq}\n"
                    )
        except Exception:
            continue

        if len(fasta_records) % 250 == 0 and len(fasta_records) > 0:
            print(f"   Fetched {len(fasta_records)} promoters...")

        if len(fasta_records) >= 2000:
            break

    # Save to data directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(fasta_records)

    print(
        f"\nSUCCESS: Downloaded {len(fasta_records)} promoter sequences to:\n  {OUTPUT_PATH}"
    )

except Exception as e:
    print(f"\nDownload Failed: {e}")