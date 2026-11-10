# Sumber data: 10 kasus CV–JD

**Dataset:** `cnamuangtoun/resume-job-description-fit` (HuggingFace), split `test` (1.759 baris: 857 No Fit, 444 Potential Fit, 458 Good Fit). Diakses 10 November 2026.

**Lisensi:** halaman dataset tidak mencantumkan lisensi maupun dataset card. Sebutkan ini di laporan dan gunakan hanya untuk keperluan evaluasi non-komersial.

## Cara pemilihan
1. Filter: panjang CV 1.200–3.500 karakter, panjang JD 600–3.000 karakter, CV tanpa pola email/telepon → 124 baris lolos.
2. Acak dengan seed 42, ambil bertingkat per label: 4 Good Fit, 3 Potential Fit, 3 No Fit, tanpa CV atau JD yang berulang.
3. Nomor baris asli ada di `labels.csv` (kolom `dataset_row`, indeks 0 tanpa header).

## Struktur per kasus
- `DSxx/cv_en.txt`: CV asli (bahasa Inggris), identik dengan dataset kecuali bagian yang disamarkan.
- `DSxx/cv_id.txt`: terjemahan Indonesia (dibuat dengan bantuan Claude; istilah teknis, nama perusahaan, dan format berantakan sengaja dipertahankan).
- `DSxx/jd.txt`: job description pasangannya (tetap bahasa Inggris).
- `labels.csv`: label dataset sebagai ground truth + observasi awal.

## Penyamaran data pribadi
- DS01 CV: nama dua supervisor → `[NAMA DIHAPUS]`; JD: tautan video → `[URL]`.
- DS02 JD: nama, email, telepon, dan LinkedIn recruiter → `[KONTAK RECRUITER DIHAPUS]`.
- DS05 CV: situs web pribadi → `[URL]`.
CV di dataset tidak memuat nama kandidat.

## Catatan kualitas data (penting untuk interpretasi)
- **Label noisy.** Dari pembacaan awal, sekitar setengah label dipertanyakan (lihat `labels.csv`). Model DistilBERT yang dilatih pada dataset ini hanya mencapai akurasi ±0,50. Jika AI tidak setuju dengan label, belum tentu AI yang salah.
- **Sebagian CV tampak hasil augmentasi.** Kalimat ringkasan yang sama persis ("Highly motivated Sales Associate…") muncul di CV insinyur elektro (DS05), developer (DS09), dan akuntan pajak (DS10). Ini juga menguji apakah AI terkecoh oleh ringkasan yang tidak sesuai isi pengalaman.
- **Format berantakan** (kata menempel, "toto", "Expected in") dibiarkan apa adanya karena mencerminkan hasil parsing PDF di dunia nyata.
- Semua JD dan CV berkonteks Amerika Serikat.
