#!/bin/bash
# Pengganti gemini-3.5-flash (kuota harian habis): jalankan gemma-4-31b-it lalu susun ulang ringkasan.
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
mkdir -p logs
# Arsipkan jawaban gemma lama yang tidak berisi JSON (versi sebelum perbaikan "thinking"), supaya diulang
python - <<'PY'
from pathlib import Path
import shutil, time
dst = Path("results_arsip") / f"gemma_tanpa_json_{time.strftime('%H%M%S')}"
for f in [*Path("results_pilot/raw/gemma").glob("*.txt"), *Path("results_dataset/raw/gemma").glob("*.txt"),
          *Path("results_variants/raw/gemma").glob("*.txt")]:
    if '"category"' not in f.read_text(encoding="utf-8"):
        (dst / f.parent.parent.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), dst / f.parent.parent.parent.name / f.name)
        print("diarsipkan (tanpa JSON):", f)
PY
echo "Mulai $(date '+%H:%M'). Perkiraan 20-30 menit. Jangan tutup jendela ini."
caffeinate -i bash -c '
{
  echo "=== gemma: pilot (cek format) ==="
  python eval_dataset.py run --tool gemma --pilot
  echo "=== gemma: uji utama ==="
  python eval_dataset.py run --tool gemma --runs 1
  echo "=== gemma: konsistensi ==="
  python eval_dataset.py run --tool gemma --cases DS02,DS05,DS09 --runs 3
  echo "=== gemma: varian ==="
  python eval_dataset.py run --tool gemma --varian
  echo "=== gemma: SELESAI $(date +%H:%M) ==="
} 2>&1 | tee logs/run_gemma.log'
python eval_dataset.py evaluate > logs/evaluate_final.log 2>&1
echo; echo "SELESAI $(date '+%H:%M'). Hasil: results_dataset/hasil_perbandingan.xlsx"
read -n 1 -s -r -p "Tekan tombol apa saja untuk menutup..."
