"""Buat 6 file prompt (S1-S3 x varian A/B) di folder prompts/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cvscreen import load_config, write_all_prompts  # noqa: E402

if __name__ == "__main__":
    for p in write_all_prompts(load_config()):
        print("ditulis:", p.relative_to(Path(__file__).parent))
