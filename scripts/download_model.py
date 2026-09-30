"""Download the base model from Hugging Face into ./models/<name>."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--repo", default="Qwen/Qwen3.5-4B", help="HF repo id (verify the exact name on huggingface.co)")
p.add_argument("--out", default=None, help="target dir (default: models/<repo name>)")
a = p.parse_args()

out = Path(a.out or Path("models") / a.repo.split("/")[-1])
snapshot_download(repo_id=a.repo, local_dir=out)
print(f"Saved to {out}")
