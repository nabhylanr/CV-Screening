"""Aplikasi screening CV untuk recruiter.  Jalankan: streamlit run app.py"""
import os
import sys
from pathlib import Path

import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

sys.path.insert(0, str(Path(__file__).parent / "src"))
import screening as S  # noqa: E402

st.set_page_config(page_title="Screening CV", layout="centered")

EXAMPLE_MUST = """Writes SQL to query and aggregate data
Uses Python (pandas) for data cleaning and analysis
Has built dashboards with Looker Studio, Tableau, or Power BI
Applies basic statistics (descriptive statistics, hypothesis testing)"""
EXAMPLE_NICE = """Experience with A/B testing or experiment analysis
Experience in edtech or presenting insights to non-technical stakeholders"""
CAT_ORDER = {"Good Fit": 0, "Potential Fit": 1, "No Fit": 2}


def _load_env():
    """Ambil API key dari file .env di folder ini (jika ada)."""
    env = Path(__file__).parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.environ.get(name, "")
    except Exception:  # noqa: BLE001  (tidak ada secrets.toml)
        return os.environ.get(name, "")


st.title("Screening CV")
st.caption("Unggah CV, tulis requirement posisi, lalu klik **Cek**. AI menyusun bukti; keputusan tetap di tangan recruiter.")

# ---------------- input ----------------
position = st.text_input("Posisi", value="Data Analyst Intern")
c1, c2 = st.columns(2)
must_txt = c1.text_area("Must-have (satu per baris)", value=EXAMPLE_MUST, height=170)
nice_txt = c2.text_area("Nice-to-have (satu per baris)", value=EXAMPLE_NICE, height=170)
files = st.file_uploader("CV kandidat (PDF, DOCX, TXT, MD · bisa beberapa file)",
                         type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)

with st.expander("Pengaturan model AI"):
    provider = st.selectbox("Penyedia", list(S.PROVIDERS))
    preset = S.PROVIDERS[provider]
    model = st.text_input("Nama model", value=preset["model"],
                          help="Nama model bisa berubah; cek daftar model di dokumentasi penyedia.")
    base_url = preset["base_url"]
    if preset["kind"] == "openai":
        base_url = st.text_input("Base URL", value=preset["base_url"])
    key_name = S.KEY_ENV.get(provider, "OPENAI_COMPAT_API_KEY")
    api_key = st.text_input("API key", value=secret(key_name), type="password",
                            help=f"Bisa juga diisi lewat environment variable / secrets: {key_name}")
    anonymize = st.checkbox("Samarkan nama & kontak sebelum dikirim ke AI", value=True)
    st.caption("Layanan gratis bisa memakai data yang dikirim untuk melatih model. Gunakan CV contoh/anonim saat uji coba.")

run = st.button("Cek", type="primary", width="stretch")

# ---------------- proses ----------------
if run:
    must = [x.strip() for x in must_txt.splitlines() if x.strip()]
    nice = [x.strip() for x in nice_txt.splitlines() if x.strip()]
    problems = []
    if not files:
        problems.append("Unggah minimal satu CV.")
    if not must:
        problems.append("Isi minimal satu must-have.")
    if provider in S.KEY_ENV and not api_key:
        problems.append("Isi API key di Pengaturan model AI.")
    if not model:
        problems.append("Isi nama model.")
    for p in problems:
        st.error(p)

    if not problems:
        reqs = S.build_requirements(must, nice)
        results, bar = [], st.progress(0.0, text="Memulai...")
        for i, f in enumerate(files):
            bar.progress(i / len(files), text=f"Menilai {f.name} ({i + 1}/{len(files)})")
            entry = {"file": f.name}
            try:
                text = S.read_cv(f.name, f.getvalue())
                if len(text) < 50:
                    raise ValueError("Teks CV tidak terbaca (mungkin PDF hasil scan).")
                sent = S.mask_identity(text) if anonymize else text
                raw = S.call_llm(S.build_prompt(position, reqs, sent), provider, api_key, model, base_url)
                entry.update(S.analyze(raw, reqs, sent))
            except Exception as e:  # noqa: BLE001
                entry["error"] = str(e)
            results.append(entry)
        bar.progress(1.0, text="Selesai")
        st.session_state["results"] = results

# ---------------- output ----------------
results = st.session_state.get("results")
if results:
    ok = [r for r in results if "error" not in r]
    for r in results:
        if "error" in r:
            st.error(f"{r['file']}: {r['error']}")

    if ok:
        ok.sort(key=lambda r: (CAT_ORDER[r["category"]], -r["must_met"], -r["nice_met"], r["perlu_dicek"], -(r["score"] or 0)))
        st.divider()
        st.subheader("Hasil")
        m1, m2, m3 = st.columns(3)
        for col, cat in zip((m1, m2, m3), CAT_ORDER):
            col.metric(cat, sum(r["category"] == cat for r in ok))

        table = pd.DataFrame([{
            "Peringkat": i + 1, "Kandidat": r["file"], "Kategori": r["category"],
            "Must-have": f"{r['must_met']}/{r['must_total']}", "Nice-to-have": f"{r['nice_met']}/{r['nice_total']}",
            "Skor AI": r["score"], "Perlu dicek": r["perlu_dicek"] + len(r["flags"]),
        } for i, r in enumerate(ok)])
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Kategori dihitung dari must-have yang buktinya terverifikasi di CV, bukan dari skor AI. "
                   "Good Fit = semua must-have terpenuhi · Potential = minimal setengah · No Fit = kurang dari setengah.")

        for i, r in enumerate(ok):
            with st.expander(f"{i + 1}. {r['file']} · {r['category']} · must-have {r['must_met']}/{r['must_total']}"):
                if r["summary"]:
                    st.write(r["summary"])
                for fl in r["flags"]:
                    st.warning(fl)
                detail = pd.DataFrame(r["requirements"])[["id", "requirement", "cek bukti", "status AI", "bukti"]]
                st.dataframe(detail, hide_index=True, width="stretch", column_config={
                    "requirement": st.column_config.TextColumn(width="medium"),
                    "cek bukti": st.column_config.TextColumn(width="medium")})

        export = pd.DataFrame([{**{k: v for k, v in r.items() if k != "requirements"},
                                "flags": "; ".join(r["flags"])} for r in ok])
        st.download_button("Unduh hasil (CSV)", export.to_csv(index=False).encode(), "hasil_screening.csv")
        st.info("Hasil ini adalah shortlist awal. Baca CV kandidat sebelum memutuskan, terutama baris "
                "'Bukti tidak ditemukan di CV', 'Hanya klaim (ringkasan/skill)', dan peringatan.")
