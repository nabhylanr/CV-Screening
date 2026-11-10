"""Dashboard evaluasi AI tools untuk screening CV.  Jalankan: streamlit run app.py"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cvscreen import (  # noqa: E402
    CRITERIA, DATA, DEMO, RESULTS, build_prompt, candidate_map, load_config, load_cv, load_gt,
    parse_output, raw_path, recommend, run_evaluation, save_raw,
)

st.set_page_config(page_title="Evaluasi AI untuk screening CV", layout="wide")
cfg = load_config()
TOOLS, RUNS = cfg["tools"], cfg["runs"]
SCEN = cfg["scenarios"]

st.sidebar.title("Evaluasi AI · screening CV")
mode = st.sidebar.radio("Sumber data", ["Hasil nyata", "Demo (sintetis)"])
BASE = RESULTS if mode == "Hasil nyata" else DEMO
if BASE == DEMO:
    st.sidebar.warning("Mode demo: angka dikarang oleh make_demo.py, bukan hasil tool. Jangan dipakai sebagai evidence.")
st.sidebar.caption(f"Tools: {', '.join(TOOLS)} · {RUNS} run per varian")


@st.cache_data(show_spinner=False)
def evaluate(base: str, _stamp: float):
    return run_evaluation(Path(base))


def stamp(base: Path) -> float:
    files = list((base / "raw").rglob("*.txt"))
    return max((f.stat().st_mtime for f in files), default=0) + len(files)


res = evaluate(str(BASE), stamp(BASE))
long, metrics, rubric = res["long"], res["metrics"], res["rubric"]

tab_setup, tab_input, tab_hasil, tab_tabel, tab_rek = st.tabs(
    ["1 · Setup", "2 · Prompt & input hasil", "3 · Hasil per skenario", "4 · Comparison table", "5 · Rekomendasi"]
)

# ---------------- 1. Setup ----------------
with tab_setup:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Job description")
        st.markdown((DATA / "job_description.md").read_text(encoding="utf-8"))
    with c2:
        st.subheader("Skenario uji")
        for k, s in SCEN.items():
            st.markdown(f"**{k} · {s['name']}**  \n{s['description']}  \n"
                        f"Varian A: `{s['target']['A']}` · Varian B: `{s['target']['B']}` · CV lain: "
                        + ", ".join(f"`{c}`" for c in s["others"]))
        st.info("Varian A dan B dijalankan di chat terpisah: model tidak melihat pasangan CV sekaligus.")
    st.subheader("Ground truth (dibuat manusia sebelum AI dijalankan)")
    st.dataframe(load_gt(), hide_index=True, width="stretch")
    with st.expander("Lihat isi CV mock"):
        cv_pick = st.selectbox("CV", load_gt().cv_id.tolist())
        st.code(load_cv(cv_pick), language="markdown")

# ---------------- 2. Prompt & input ----------------
with tab_input:
    st.subheader("Progres sesi")
    grid = []
    for tool in TOOLS:
        row = {"tool": tool}
        for sc in SCEN:
            for v in ("A", "B"):
                done = sum(raw_path(BASE, tool, sc, v, r).exists() for r in range(1, RUNS + 1))
                row[f"{sc}-{v}"] = f"{done}/{RUNS}"
        grid.append(row)
    st.dataframe(pd.DataFrame(grid), hide_index=True, width="stretch")

    a, b, c, d = st.columns(4)
    sc = a.selectbox("Skenario", list(SCEN))
    v = b.selectbox("Varian", ["A", "B"])
    tool = c.selectbox("Tool", TOOLS)
    run = d.selectbox("Run", list(range(1, RUNS + 1)))
    cmap = candidate_map(cfg, sc, v)

    st.markdown("**Langkah:** buka chat BARU di tool terpilih → salin prompt → tempel jawaban AI di kotak bawah → simpan.")
    with st.expander("Prompt (klik ikon salin di pojok kanan atas)", expanded=False):
        st.code(build_prompt(cfg, sc, v), language="text")

    path = raw_path(BASE, tool, sc, v, run)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    pasted = st.text_area("Output AI (JSON)", value=existing, height=260, key=f"{tool}{sc}{v}{run}{mode}")
    if st.button("Validasi & simpan", type="primary", disabled=BASE == DEMO):
        try:
            rows, warns = parse_output(pasted, cmap)
            save_raw(BASE, tool, sc, v, run, pasted)
            st.success(f"Tersimpan: {path.relative_to(BASE.parent)} · {len(rows)} kandidat terbaca")
            for w in warns:
                st.warning(w)
            st.cache_data.clear()
        except Exception as e:  # noqa: BLE001
            st.error(f"Tidak tersimpan, JSON tidak valid: {e}")
    if BASE == DEMO:
        st.caption("Penyimpanan dimatikan di mode demo.")
    if res["issues"]:
        with st.expander(f"Peringatan parsing ({len(res['issues'])})"):
            st.dataframe(pd.DataFrame(res["issues"]), hide_index=True, width="stretch")


def need_data():
    st.info("Belum ada output. Isi lewat tab 2, atau pilih mode demo di sidebar untuk melihat contoh tampilan.")


# ---------------- 3. Hasil per skenario ----------------
with tab_hasil:
    if long.empty:
        need_data()
    else:
        st.subheader("Akurasi terhadap ground truth")
        acc = metrics.melt(id_vars="tool", value_vars=["category_accuracy", "top1_accuracy"],
                           var_name="metrik", value_name="nilai")
        acc["metrik"] = acc.metrik.map({"category_accuracy": "Kategori benar", "top1_accuracy": "Top-1 benar"})
        st.plotly_chart(px.bar(acc, x="tool", y="nilai", color="metrik", barmode="group", range_y=[0, 1],
                               text_auto=".0%"), width="stretch")

        def pair_chart(scenario, label_a, label_b, title):
            t = long[(long.scenario == scenario) & long.is_target].copy()
            t["versi"] = t.variant.map({"A": label_a, "B": label_b})
            fig = px.strip(t, x="tool", y="score", color="versi", hover_data=["run", "category"], title=title)
            fig.update_traces(marker=dict(size=11, opacity=0.8))
            fig.update_layout(yaxis_range=[0, 100])
            return fig

        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(pair_chart("S1", "Nama A (CV_A_m)", "Nama B (CV_A_f)", "S1 · Bias: CV identik beda nama"),
                        width="stretch")
        c2.plotly_chart(pair_chart("S2", "EN", "ID", "S2 · Konsistensi bahasa"), width="stretch")
        c3.plotly_chart(pair_chart("S3", "Asli", "+ keyword", "S3 · Ketahanan keyword"), width="stretch")
        st.caption("Tiap titik = satu run. Ideal: titik dua warna bertumpuk (S1, S2) dan '+ keyword' tidak lebih tinggi (S3).")

        st.subheader("Kestabilan antar run")
        spread = long.groupby(["tool", "scenario", "variant", "cv_id"]).score.agg(lambda s: s.max() - s.min()).reset_index(name="rentang")
        st.plotly_chart(px.box(spread, x="tool", y="rentang", points="all",
                               labels={"rentang": "max − min skor antar run"}), width="stretch")

        st.subheader("Grounding: kutipan bukti benar-benar ada di CV?")
        g = long.groupby("tool")[["n_claims", "n_verified"]].sum().reset_index()
        g["tidak terverifikasi"] = g.n_claims - g.n_verified
        st.plotly_chart(px.bar(g, x="tool", y=["n_verified", "tidak terverifikasi"],
                               labels={"value": "jumlah klaim", "variable": ""}), width="stretch")
        with st.expander("Klaim yang kutipannya tidak ditemukan di CV"):
            from cvscreen import REQS, evidence_in_cv
            bad = []
            for _, r in long.iterrows():
                for q in REQS:
                    if r[f"{q}_status"] in ("met", "partial") and not evidence_in_cv(r[f"{q}_evidence"], load_cv(r.cv_id)):
                        bad.append({"tool": r.tool, "scenario": r.scenario, "run": r.run, "cv": r.cv_id,
                                    "req": q, "evidence": r[f"{q}_evidence"] or "(kosong)"})
            st.dataframe(pd.DataFrame(bad), hide_index=True, width="stretch")

# ---------------- 4. Comparison table ----------------
with tab_tabel:
    if long.empty:
        need_data()
    else:
        st.subheader("Rubric 1–3 per kriteria")
        st.dataframe(rubric.set_index("tool").style.background_gradient(cmap="RdYlGn", vmin=1, vmax=3),
                     width="stretch")
        st.subheader("Metrik mentah")
        show = metrics.rename(columns={
            "category_accuracy": "Akurasi kategori", "top1_accuracy": "Top-1", "delta_bias": "Δ bias",
            "bias_category_flip": "Flip kategori (S1)", "run_range": "Rentang run", "delta_lang": "Δ bahasa",
            "lang_category_flip": "Flip kategori (S2)", "run_category_flip_rate": "Rate flip antar run",
            "consistency_worst": "Konsistensi (terburuk)", "delta_kw": "Δ keyword",
            "kw_category_inflation": "Kategori naik (S3)", "kw_flagged_rate": "Keyword ditandai",
            "grounding_rate": "Grounding", "n_sessions": "Sesi"})
        st.dataframe(show.set_index("tool").round(3), width="stretch")
        with st.expander("Ambang batas rubric (dari config.yaml)"):
            st.json(cfg["rubric"])
            st.caption("Ambang batas adalah asumsi desain; sebutkan dan justifikasi di laporan.")

        st.subheader("Trade-off otomatis")
        for _, r in rubric.iterrows():
            strong = [c for c in CRITERIA if r[c] == 3]
            weak = [c for c in CRITERIA if r[c] == 1]
            st.markdown(f"**{r.tool}** — kuat: {', '.join(strong) or '-'} · lemah: {', '.join(weak) or '-'}")
        st.download_button("Unduh comparison table (CSV)",
                           rubric.merge(metrics, on="tool").to_csv(index=False).encode(), "comparison_table.csv")

# ---------------- 5. Rekomendasi ----------------
with tab_rek:
    if long.empty:
        need_data()
    else:
        st.subheader("Recommendation criteria berdasarkan kebutuhan")
        profile = st.selectbox("Profil kebutuhan", list(cfg["profiles"]) + ["Kustom"])
        if profile == "Kustom":
            cols = st.columns(len(CRITERIA))
            weights = {c: cols[i].slider(c, 0.0, 1.0, 0.2, 0.05) for i, c in enumerate(CRITERIA)}
        else:
            st.caption(cfg["profiles"][profile]["description"])
            weights = cfg["profiles"][profile]["weights"]
            st.markdown("Bobot: " + " · ".join(f"{k} {v:.0%}" for k, v in weights.items()))
        rec = recommend(rubric, weights)
        st.dataframe(rec[["tool", "skor_tertimbang", *CRITERIA, "catatan"]].round(2), hide_index=True,
                     width="stretch")

        st.subheader("Rekomendasi tiap profil")
        summary = []
        for name, p in cfg["profiles"].items():
            r = recommend(rubric, p["weights"]).iloc[0]
            summary.append({"profil": name, "tool teratas": r.tool, "skor": round(r.skor_tertimbang, 2),
                            "kriteria bernilai 1": r.catatan})
        st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
        st.warning("Human oversight: AI hanya menyusun shortlist dan bukti. Keputusan lolos/tidak tetap oleh recruiter, "
                   "dan tool dengan kriteria bernilai 1 tidak dipakai tanpa review manual untuk aspek tersebut.")
        st.caption(f"Bukti dari {len(long)} penilaian CV. Ukuran sampel kecil: hasil bersifat indikatif.")
