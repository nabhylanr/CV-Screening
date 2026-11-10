# Screening CV dengan AI + evaluasi tools

Folder ini berisi dua bagian:

1. **Aplikasi recruiter (`app.py`)**: satu halaman. Unggah beberapa CV (PDF/DOCX/TXT/MD), tulis requirement, klik **Cek**, hasil muncul di bawahnya.
2. **Uji coba tools dengan dataset publik (`eval_dataset.py`)**: 10 pasangan CV–JD berlabel, tiap CV dalam bahasa Inggris dan Indonesia.

```
app.py                  aplikasi recruiter (Streamlit)
src/screening.py        logika aplikasi: baca file, samarkan identitas, panggil AI, cek bukti, tentukan kategori
src/cvscreen.py         fungsi bantu bersama (parsing JSON, normalisasi kategori, cek kutipan di CV)
src/prompt_dataset.txt  satu template prompt untuk uji coba (JD & CV diisi otomatis)
eval_dataset.py         buat prompt uji coba + hitung metrik
data/dataset/           10 kasus (cv_en, cv_id, jd) + labels.csv + SOURCE.md
```

## Menjalankan aplikasi recruiter

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Di *Pengaturan model AI*, pilih penyedia gratis: Gemini (Google AI Studio), Groq, Mistral, atau Ollama (model lokal, tanpa internet). Nama model bisa berubah; cek dokumentasi penyedia bila muncul error "model not found". API key bisa disimpan di `.streamlit/secrets.toml` (`GEMINI_API_KEY = "..."`). Jangan commit file itu.

Cara aplikasi menilai:
- AI diminta mengutip bukti per requirement. **Kategori dihitung oleh kode**, bukan oleh skor AI: Good Fit = semua must-have terbukti, Potential = minimal setengah, No Fit = kurang dari setengah.
- Kutipan dicek otomatis ke teks CV. Kutipan yang tidak ditemukan, atau hanya berasal dari ringkasan/daftar skill, tidak dihitung dan ditandai "perlu dicek" (mitigasi keyword stuffing dan halusinasi).
- Nama, email, telepon, dan URL disamarkan sebelum dikirim ke AI (bisa dimatikan) untuk mengurangi bias.
- Kalimat yang menyerupai instruksi untuk AI di dalam CV ditandai sebagai peringatan (prompt injection).
- Hasil = shortlist + bukti. Keputusan tetap di recruiter.

Peringatan: layanan AI gratis dapat memakai data yang dikirim untuk meningkatkan produknya. Untuk uji coba, gunakan CV dari `data/dataset/`.

## Uji coba tools (10 kasus dataset publik)

Sumber: HuggingFace `cnamuangtoun/resume-job-description-fit`, split test, label Good / Potential / No Fit. Detail pemilihan, penyamaran data pribadi, dan catatan kualitas label ada di `data/dataset/SOURCE.md`.

```bash
python eval_dataset.py show DS03 id --copy   # rakit prompt kasus DS03 (bahasa Indonesia) dan salin ke clipboard
# tempel di chat BARU, simpan jawaban AI apa adanya ke:
#   results_dataset/raw/<tool>/<case>_<lang>_run<n>.txt   contoh: results_dataset/raw/gemini/DS03_id_run1.txt
python eval_dataset.py evaluate    # akurasi vs label, akurasi toleran 1 tingkat, beda EN-ID, konsistensi antar run, grounding
```

### Menjalankan otomatis lewat API (semua gratis)

Semua model yang dipakai gratis lewat free tier, tanpa kartu kredit:

| `--tool` | Model default | Pembuat model | Penyedia gratis | API key |
|---|---|---|---|---|
| `gemma` | gemma-4-31b-it | Google (open-weight) | Google AI Studio | `GEMINI_API_KEY` |
| `gemini` | gemini-3.5-flash (opsional; tier gratis ~20 request/hari, tidak cukup untuk 52 panggilan) | Google | Google AI Studio | `GEMINI_API_KEY` |
| `gptoss` | openai/gpt-oss-120b | OpenAI (open-weight) | Groq | `GROQ_API_KEY` |
| `qwen` | qwen/qwen3.8-27b | Alibaba (open-weight) | Groq (key yang sama) | `GROQ_API_KEY` |
| `mistral` (opsional) | mistral-small-latest | Mistral AI | Mistral | `MISTRAL_API_KEY` |
| `ollama` (opsional) | llama3.1 | Meta | lokal di laptop | - |

Isi key di file `.env` di folder ini (sudah masuk `.gitignore`):

```
GEMINI_API_KEY=...
GROQ_API_KEY=...
```

**Langkah 0 – cek nama model.** Nama model gratis sering berganti. `python eval_dataset.py models` menampilkan model yang aktif untuk API key-mu; kalau default di tabel tidak ada, pakai `--model <nama>`.

**Langkah 1 – pilot (sangat disarankan sebelum uji coba).** Jalankan setiap model di 3 kasus pilot (`data/pilot/`, dari split train, bukan data uji). Yang dicek hanya format: JSON valid, kategori & skor valid, 3–6 requirement, status valid, kutipan bukti ada di CV, plus pemakaian token. Label tidak dipakai, jadi prompt tidak "disetel" ke data uji.

```bash
python eval_dataset.py run --tool gemini --pilot
python eval_dataset.py run --tool gptoss --pilot
python eval_dataset.py run --tool qwen --pilot
# hasil: results_pilot/pilot_report.csv
```

Jika ada tool yang BELUM LOLOS, perbaiki `src/prompt_dataset.txt`, hapus folder `results_pilot/`, ulangi pilot. Setelah semua lolos, **jangan ubah prompt lagi**; versi prompt (hash) tercatat otomatis di log.

**Langkah 2 – uji coba 10 kasus.**

```bash
python eval_dataset.py run --tool gemini --runs 1     # 20 panggilan (10 kasus x EN/ID)
python eval_dataset.py run --tool gptoss --runs 1
python eval_dataset.py run --tool qwen --runs 1
python eval_dataset.py run --tool gemini --cases DS02,DS05,DS09 --runs 3   # konsistensi: ulang 3x di sebagian kasus
```

**Langkah 3 – fairness & ketahanan (setelah langkah 2).** 20 varian per model (lihat `data/variants/METODE.md`): 5 kasus dengan nama laki-laki vs perempuan, dan 5 kasus dengan keyword tanpa bukti / instruksi tersembunyi. Hasilnya dibandingkan dengan CV asli dari langkah 2.

```bash
python eval_dataset.py run --tool gemini --varian
python eval_dataset.py run --tool gptoss --varian
python eval_dataset.py run --tool qwen --varian
python eval_dataset.py evaluate
```

- Hasil yang sudah ada dilewati, jadi aman dijalankan ulang kalau terputus (`--overwrite` untuk menimpa).
- Jeda antar panggilan otomatis disesuaikan batas rate gratis (Groq dibatasi token per menit, jadi ±30 detik per panggilan; 20 panggilan ≈ 10 menit).
- Setiap panggilan dicatat di `results_dataset/run_log.csv`: waktu, tool, model, versi prompt, temperature yang benar-benar dipakai, status, durasi, token input & output.
- Setelah setiap `run`, perbandingan semua model otomatis diperbarui dan disimpan di `results_dataset/`:

| File | Isi |
|---|---|
| `hasil_perbandingan.xlsx` | semua tabel di bawah dalam satu file (siap diunggah ke Google Sheets); sheet pertama = ringkasan 5 aspek |
| `ringkasan_5_aspek.csv` | 1 baris per model: akurasi, konsistensi bahasa & run, grounding, fairness (selisih skor, kategori beda, impact ratio), ketahanan (ASR & deteksi per serangan) |
| `fairness.csv`, `fairness_detail.csv` | skor netral vs nama laki-laki vs perempuan per kasus |
| `ketahanan.csv`, `ketahanan_detail.csv` | kategori & skor CV asli vs varian serangan, terdeteksi atau tidak |
| `report.md` | ringkasan siap baca; salinan bertanggal di `history/` untuk setiap evaluasi |
| `metrics.csv` | 1 baris per model: akurasi, akurasi EN/ID, beda EN–ID, konsistensi antar run, grounding |
| `per_case.csv` | kategori tiap model per kasus & bahasa vs label |
| `confusion.csv` | label vs jawaban AI per model |
| `scores_long.csv` | 1 baris per penilaian (skor, kategori, flags, ringkasan, file mentah) |
| `requirements_detail.csv` | setiap requirement yang dinilai AI + kutipan bukti + apakah kutipan ada di CV |
| `raw/<tool>/` | jawaban AI apa adanya |
- Nama model dan batas free tier bisa berubah; jika muncul error "model not found", ganti dengan `--model`.
- Free tier Gemini dan Mistral dapat memakai data yang dikirim untuk melatih model; aman karena data uji berasal dari dataset publik.

Karena label dataset noisy, laporkan juga akurasi toleran 1 tingkat dan tinjau kasus yang ditandai "Dipertanyakan" di `labels.csv`.

## Keterbatasan
- Sampel kecil (10 kasus): hasil indikatif, bukan generalisasi statistik.
- Label dataset noisy; dataset tidak mencantumkan lisensi; konteks pasar kerja AS.
- Model chat versi gratis bisa berubah sewaktu-waktu dan temperature tidak bisa dikontrol; catat tanggal pengujian.
- Model yang diuji adalah model gratis via API, bukan versi chat ChatGPT/Claude; hasil tidak otomatis berlaku untuk produk chat tersebut.

## Catatan penggunaan AI
Kode dan terjemahan CV ke bahasa Indonesia dibuat dengan bantuan Claude (Anthropic); desain evaluasi, peninjauan label, dan interpretasi hasil oleh penulis.
