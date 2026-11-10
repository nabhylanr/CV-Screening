"""Hitung metrik dari output di results/raw (atau demo/raw dengan --demo)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cvscreen import DEMO, RESULTS, run_evaluation  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="pakai data demo sintetis")
    base = DEMO if ap.parse_args().demo else RESULTS
    res = run_evaluation(base)
    if res["long"].empty:
        print(f"Belum ada output di {base / 'raw'}. Tempel hasil AI lewat Streamlit atau simpan file .txt di sana.")
        sys.exit(0)
    for i in res["issues"]:
        print("PERINGATAN:", i["file"], "-", i["masalah"])
    print("\nMetrik:\n", res["metrics"].round(3).to_string(index=False))
    print("\nRubric (1-3):\n", res["rubric"].to_string(index=False))
    print(f"\nCSV disimpan di {base}")
