#!/bin/bash
# Jalankan SEMUA uji coba (3 model x uji utama + konsistensi + varian) lalu buat ringkasan.
# Cara pakai: klik dua kali file ini di Finder, atau: ./run_all.command
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
mkdir -p logs
echo "Mulai $(date '+%H:%M'). Log per model ada di folder logs/. Jangan tutup jendela ini."

jalankan_model () {
  tool=$1
  {
    echo "=== $tool: uji utama (10 kasus x EN/ID) ==="
    python eval_dataset.py run --tool "$tool" --runs 1
    echo "=== $tool: konsistensi (DS02, DS05, DS09 x 3 run) ==="
    python eval_dataset.py run --tool "$tool" --cases DS02,DS05,DS09 --runs 3
    echo "=== $tool: varian fairness & ketahanan ==="
    python eval_dataset.py run --tool "$tool" --varian
    echo "=== $tool: SELESAI $(date '+%H:%M') ==="
  } > "logs/run_$tool.log" 2>&1
  echo "$tool selesai $(date '+%H:%M') (cek logs/run_$tool.log)"
}

# Ketiga model berjalan paralel (batas rate masing-masing terpisah), laptop dijaga agar tidak sleep
caffeinate -i bash -c "$(declare -f jalankan_model); jalankan_model gemma & sleep 5; jalankan_model gptoss & sleep 5; jalankan_model qwen & wait"

echo "Menyusun ringkasan akhir..."
python eval_dataset.py evaluate > logs/evaluate_final.log 2>&1
tail -n 60 logs/evaluate_final.log
echo
echo "SELESAI $(date '+%H:%M'). Hasil: results_dataset/hasil_perbandingan.xlsx dan results_dataset/report.md"
read -n 1 -s -r -p "Tekan tombol apa saja untuk menutup..."
