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

Di *Pengaturan model AI*, pilih penyedia dengan akses gratis: Gemini (API key dari Google AI Studio), Groq (API key gratis), atau Ollama (model lokal, tanpa internet). Nama model bisa berubah; cek dokumentasi penyedia bila muncul error "model not found". API key bisa disimpan di `.streamlit/secrets.toml` (`GEMINI_API_KEY = "..."`). Jangan commit file itu.

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

Karena label dataset noisy, laporkan juga akurasi toleran 1 tingkat dan tinjau kasus yang ditandai "Dipertanyakan" di `labels.csv`.

## Keterbatasan
- Sampel kecil (10 kasus): hasil indikatif, bukan generalisasi statistik.
- Label dataset noisy; dataset tidak mencantumkan lisensi; konteks pasar kerja AS.
- Model chat versi gratis bisa berubah sewaktu-waktu dan temperature tidak bisa dikontrol; catat tanggal pengujian.
- Aplikasi memakai API, sedangkan uji coba memakai versi chat; perilaku keduanya belum tentu sama.

## Catatan penggunaan AI
Kode dan terjemahan CV ke bahasa Indonesia dibuat dengan bantuan Claude (Anthropic); desain evaluasi, peninjauan label, dan interpretasi hasil oleh penulis.
