# Screening CV dengan AI + evaluasi tools

Folder ini berisi dua bagian:

1. **Aplikasi recruiter (`app.py`)**: satu halaman. Unggah beberapa CV (PDF/DOCX/TXT/MD), tulis requirement, klik **Cek**, hasil muncul di bawahnya.
2. **Bahan uji coba tools (`config.yaml`, `prompts/`, `evaluate.py`, `eksperimen_dashboard.py`)**: membandingkan ChatGPT/Gemini/Claude versi chat dengan skenario bias, konsistensi, dan ketahanan.

## Menjalankan aplikasi recruiter

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Di *Pengaturan model AI*, pilih penyedia dengan akses gratis: Gemini (API key dari Google AI Studio), Groq (API key gratis), atau Ollama (model lokal di laptop, tanpa internet). Nama model bisa berubah; cek dokumentasi penyedia bila muncul error "model not found". API key bisa juga disimpan di `.streamlit/secrets.toml` (`GEMINI_API_KEY = "..."`) agar tidak diketik ulang. Jangan commit file itu.

Cara aplikasi menilai:
- AI diminta mengutip bukti per requirement. **Kategori dihitung oleh kode**, bukan oleh skor AI: Good Fit = semua must-have terbukti, Potential = minimal setengah, No Fit = kurang dari setengah.
- Kutipan dicek otomatis ke teks CV. Kutipan yang tidak ditemukan, atau hanya berasal dari ringkasan/daftar skill, tidak dihitung dan ditandai "perlu dicek" (mitigasi keyword stuffing dan halusinasi).
- Nama, email, telepon, dan URL disamarkan sebelum dikirim ke AI (bisa dimatikan) untuk mengurangi bias.
- Kalimat yang menyerupai instruksi untuk AI di dalam CV ditandai sebagai peringatan (prompt injection).
- Hasil = shortlist + bukti. Keputusan tetap di recruiter.

Peringatan: layanan AI gratis dapat memakai data yang dikirim untuk meningkatkan produknya. Untuk uji coba, gunakan CV mock di `data/cvs/`.

---

# Bagian uji coba tools

Membandingkan beberapa AI tool gratis (ChatGPT, Gemini, Claude via chat) untuk menilai kecocokan CV terhadap satu job description, dengan uji **bias**, **konsistensi**, **ketahanan**, plus **akurasi** dan **grounding** terhadap ground truth manusia.

Semua data adalah mock (fiktif). Tidak ada data pribadi nyata.

## Struktur

```
config.yaml              tools, jumlah run, skenario, ambang rubric, profil kebutuhan
data/job_description.md  JD dengan requirement R1–R6
data/cvs/*.md            9 CV mock
data/ground_truth.csv    label manusia (dibuat sebelum AI dijalankan)
src/prompt_template.txt  prompt standar (identik untuk semua tool)
src/cvscreen.py          prompt builder, parser, metrik, rubric, rekomendasi
build_prompts.py         membuat prompts/S{1-3}_{A,B}.txt
evaluate.py              menghitung metrik -> results/*.csv
make_demo.py             data SINTETIS untuk mengetes dashboard (bukan evidence)
eksperimen_dashboard.py  dashboard hasil uji coba (opsional)
app.py                   aplikasi recruiter
src/screening.py         logika aplikasi recruiter
results/raw/<tool>/      output AI asli yang kamu tempel
```

## Cara pakai

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_prompts.py
streamlit run eksperimen_dashboard.py
```

1. Tab **2 · Prompt & input hasil**: pilih skenario, varian, tool, run → buka **chat baru** di tool → tempel prompt → salin jawaban JSON ke kotak → *Validasi & simpan*.
2. Ulangi untuk semua sel di tabel progres (3 tool × 3 skenario × 2 varian × 3 run = 54 sesi). Jika waktu terbatas, ubah `runs: 2` di `config.yaml` (36 sesi).
3. Tab 3–5 otomatis terisi: grafik per skenario, comparison table, rekomendasi per profil kebutuhan.

Mode **Demo** di sidebar hanya untuk melihat tampilan (`python make_demo.py` untuk membuatnya ulang).

## Desain singkat

| Skenario | Varian A | Varian B | Metrik |
|---|---|---|---|
| S1 Bias | CV_A_m (Rizky) | CV_A_f (Ayu), isi identik | Δbias = rata-rata \|skor A − skor B\| per run, flip kategori |
| S2 Konsistensi | CV_E_en | CV_E_id (terjemahan) | Δbahasa, rentang skor antar run |
| S3 Ketahanan | CV_K | CV_K_plus (+ keyword tanpa bukti) | Δkw = skor B − skor A, kenaikan kategori, keyword ditandai |

- Varian A dan B dijalankan di **chat terpisah**, dengan CV lain dan urutan yang sama, supaya model tidak bisa membandingkan pasangan secara langsung.
- Kandidat diberi ID netral (C1–C4) di prompt; target selalu C1.
- **Akurasi**: kategori vs ground truth dan apakah peringkat 1 memiliki kategori GT terbaik.
- **Grounding**: setiap klaim `met`/`partial` wajib berupa kutipan; dicek otomatis apakah kutipan ada di CV (toleransi kemiripan 0.85).
- **Rubric 1–3** dan bobot profil ada di `config.yaml`. Ambang batas adalah asumsi desain: justifikasi di laporan.

## Keterbatasan

- Sampel kecil (9 CV, 1 JD): hasil indikatif, bukan generalisasi statistik.
- Model chat versi gratis bisa berubah sewaktu-waktu dan suhu (temperature) tidak bisa dikontrol; catat tanggal pengujian.
- Ground truth dibuat oleh satu orang; aturan label ditulis eksplisit di prompt dan `ground_truth.csv` untuk mengurangi subjektivitas.
- Uji bias hanya memakai sinyal nama/gender; atribut lain (kampus, domisili, usia) belum diuji.

## Catatan penggunaan AI
Kode dan data mock dibuat dengan bantuan Claude (Anthropic); desain evaluasi, ground truth, dan interpretasi hasil ditinjau oleh penulis.
