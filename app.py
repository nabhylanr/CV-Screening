"""Aplikasi screening CV untuk recruiter.  Jalankan: streamlit run app.py"""
import html
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

sys.path.insert(0, str(Path(__file__).parent / "src"))
import screening as S  # noqa: E402

# Model fix: Qwen (open-weight Alibaba) via Groq, sama dengan tool "qwen" di eval_dataset.py
PROVIDER = "Groq"
MODEL = "qwen/qwen3.8-27b"
KEY_NAME = "GROQ_API_KEY"

st.set_page_config(page_title="Screening CV", page_icon="📄", layout="centered")

EXAMPLE_MUST = """Writes SQL to query and aggregate data
Uses Python (pandas) for data cleaning and analysis
Has built dashboards with Looker Studio, Tableau, or Power BI
Applies basic statistics (descriptive statistics, hypothesis testing)"""
EXAMPLE_NICE = """Experience with A/B testing or experiment analysis
Experience in edtech or presenting insights to non-technical stakeholders"""
CAT_ORDER = {"Good Fit": 0, "Potential Fit": 1, "No Fit": 2}
CAT_STYLE = {  # kelas CSS, emoji untuk judul expander
    "Good Fit": ("good", "🟢"),
    "Potential Fit": ("potential", "🟡"),
    "No Fit": ("nofit", "🔴"),
}
CHECK_STYLE = {
    "Terverifikasi": "ok",
    "Hanya klaim (ringkasan/skill)": "warn",
    "Bukti tidak ditemukan di CV": "bad",
    "-": "muted",
}


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


def esc(x) -> str:
    return html.escape(str(x or ""))


# ---------------- gaya ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --brand: #0a0a0a; --brand-2: #262626;
  --good: #059669; --good-bg: #ecfdf5;
  --pot: #b45309; --pot-bg: #fffbeb;
  --bad: #dc2626; --bad-bg: #fef2f2;
  --muted: #64748b; --line: #e2e8f0;
}
html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, button, input, textarea {
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
}
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 820px; }
#MainMenu, footer { visibility: hidden; }

/* hero */
.hero {
  background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%);
  border-radius: 22px; padding: 30px 32px; color: #fff; margin-bottom: 1.6rem;
  box-shadow: 0 18px 40px -18px rgba(0,0,0,.55); position: relative; overflow: hidden;
}
.hero::after {
  content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
  border-radius: 50%; background: rgba(255,255,255,.10);
}
.hero h1 { color: #fff !important; font-size: 2.1rem; font-weight: 800; margin: 0; padding: 0; letter-spacing: -.02em; }
.hero .chip {
  display: inline-flex; align-items: center; gap: 6px; margin-top: 12px;
  background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
  padding: 5px 12px; border-radius: 999px; font-size: .82rem; font-weight: 600;
}

/* judul langkah */
.step { display: flex; align-items: center; gap: 10px; margin: .2rem 0 .6rem; }
.step .num {
  width: 28px; height: 28px; border-radius: 9px; display: grid; place-items: center;
  background: linear-gradient(135deg, var(--brand), var(--brand-2)); color: #fff;
  font-weight: 700; font-size: .85rem;
}
.step .title { font-weight: 700; font-size: 1.05rem; }
.step .hint { color: var(--muted); font-size: .85rem; margin-left: auto; }

/* kontainer bergaris jadi kartu */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 18px !important; border-color: var(--line) !important;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -16px rgba(15,23,42,.18);
}

/* input */
.stTextArea textarea, .stTextInput input {
  border-radius: 12px !important; font-size: .95rem !important; line-height: 1.55 !important;
}
.stTextArea label p, .stTextInput label p { font-weight: 600 !important; }

/* tombol utama */
.stButton > button[kind="primary"], .stDownloadButton > button {
  border-radius: 14px; padding: .75rem 1rem; font-weight: 700; font-size: 1rem; border: none;
  background: linear-gradient(135deg, var(--brand), var(--brand-2)); color: #fff;
  box-shadow: 0 10px 24px -12px rgba(0,0,0,.8); transition: transform .12s ease, box-shadow .12s ease;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px); box-shadow: 0 14px 28px -12px rgba(0,0,0,.9); color: #fff;
}

/* ringkasan kategori */
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: .4rem 0 1.2rem; }
.stat { border-radius: 16px; padding: 16px 18px; border: 1px solid var(--line); }
.stat .label { font-size: .82rem; font-weight: 600; opacity: .85; }
.stat .value { font-size: 2rem; font-weight: 800; line-height: 1.1; margin-top: 4px; }
.stat.good { background: var(--good-bg); color: var(--good); border-color: #a7f3d0; }
.stat.potential { background: var(--pot-bg); color: var(--pot); border-color: #fde68a; }
.stat.nofit { background: var(--bad-bg); color: var(--bad); border-color: #fecaca; }

/* badge & baris requirement */
.badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .75rem; font-weight: 700; white-space: nowrap; }
.badge.good { background: var(--good-bg); color: var(--good); }
.badge.potential { background: var(--pot-bg); color: var(--pot); }
.badge.nofit { background: var(--bad-bg); color: var(--bad); }
.badge.ok { background: var(--good-bg); color: var(--good); }
.badge.warn { background: var(--pot-bg); color: var(--pot); }
.badge.bad { background: var(--bad-bg); color: var(--bad); }
.badge.muted { background: #f1f5f9; color: var(--muted); }
.badge.kind { background: #f1f5f9; color: #0a0a0a; }

.cand-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 10px; }
.summary { font-size: .95rem; line-height: 1.6; margin: 0 0 12px; }
.bar { height: 8px; background: #f1f5f9; border-radius: 999px; overflow: hidden; margin: 2px 0 14px; }
.bar > span { display: block; height: 100%; background: linear-gradient(90deg, var(--brand), var(--brand-2)); border-radius: 999px; }
.req { border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; margin-bottom: 8px; }
.req .top { display: flex; gap: 8px; align-items: flex-start; flex-wrap: wrap; }
.req .text { font-weight: 600; font-size: .92rem; flex: 1 1 260px; }
.req .quote {
  margin-top: 8px; padding: 8px 12px; border-left: 3px solid var(--brand); background: #f8fafc;
  border-radius: 0 10px 10px 0; font-size: .86rem; color: #334155; font-style: italic;
}
.req .nil { margin-top: 6px; font-size: .84rem; color: var(--muted); }
.note { color: var(--muted); font-size: .84rem; line-height: 1.55; }

@media (max-width: 640px) {
  .stats { grid-template-columns: 1fr; }
  .hero { padding: 24px 22px; }
  .hero h1 { font-size: 1.7rem; }
  .step .hint { display: none; }
}
</style>
""", unsafe_allow_html=True)


def step(num: int, title: str, hint: str = "") -> None:
    st.markdown(f'<div class="step"><div class="num">{num}</div><div class="title">{title}</div>'
                f'<div class="hint">{hint}</div></div>', unsafe_allow_html=True)


# ---------------- header ----------------
st.markdown('<div class="hero"><h1>Screening CV</h1>'
            '</div>', unsafe_allow_html=True)

# ---------------- input ----------------
with st.container(border=True):
    step(1, "Posisi")
    position = st.text_input("Posisi", value="Data Analyst Intern", label_visibility="collapsed")

with st.container(border=True):
    step(2, "Requirement", "satu requirement per baris")
    must_txt = st.text_area("Must-have", value=EXAMPLE_MUST, height=230)
    nice_txt = st.text_area("Nice-to-have", value=EXAMPLE_NICE, height=180)

with st.container(border=True):
    step(3, "CV kandidat", "PDF, DOCX, TXT, MD · bisa beberapa file")
    files = st.file_uploader("CV kandidat", type=["pdf", "docx", "txt", "md"],
                             accept_multiple_files=True, label_visibility="collapsed")

run = st.button("Cek CV", type="primary", width="stretch")

# ---------------- proses ----------------
if run:
    api_key = secret(KEY_NAME)
    must = [x.strip() for x in must_txt.splitlines() if x.strip()]
    nice = [x.strip() for x in nice_txt.splitlines() if x.strip()]
    problems = []
    if not files:
        problems.append("Unggah minimal satu CV.")
    if not must:
        problems.append("Isi minimal satu must-have.")
    if not api_key:
        problems.append(f"API key belum diatur. Isi {KEY_NAME} di file .env atau secrets Streamlit.")
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
                sent = S.mask_identity(text)  # nama & kontak selalu disamarkan sebelum dikirim ke AI
                raw = S.call_llm(S.build_prompt(position, reqs, sent), PROVIDER, api_key, MODEL,
                                 S.PROVIDERS[PROVIDER]["base_url"])
                entry.update(S.analyze(raw, reqs, sent))
            except Exception as e:  # noqa: BLE001
                entry["error"] = str(e)
            results.append(entry)
        bar.empty()
        st.session_state["results"] = results


def render_candidate(r: dict) -> None:
    cls, _ = CAT_STYLE[r["category"]]
    score = r["score"] if r["score"] is not None else "–"
    review = r["perlu_dicek"] + len(r["flags"])
    pct = int(100 * r["must_met"] / r["must_total"]) if r["must_total"] else 0
    head = (f'<div class="cand-head"><span class="badge {cls}">{esc(r["category"])}</span>'
            f'<span class="badge kind">Must-have {r["must_met"]}/{r["must_total"]}</span>'
            f'<span class="badge kind">Nice-to-have {r["nice_met"]}/{r["nice_total"]}</span>'
            f'<span class="badge muted">Skor AI {esc(score)}</span>')
    if review:
        head += f'<span class="badge warn">{review} perlu dicek</span>'
    head += "</div>"
    head += f'<div class="bar"><span style="width:{pct}%"></span></div>'
    if r["summary"]:
        head += f'<p class="summary">{esc(r["summary"])}</p>'
    st.markdown(head, unsafe_allow_html=True)

    for fl in r["flags"]:
        st.warning(fl, icon="⚠️")

    rows = []
    for q in r["requirements"]:
        chk = q["cek bukti"]
        body = (f'<div class="req"><div class="top">'
                f'<span class="badge kind">{esc(q["id"])}</span>'
                f'<span class="text">{esc(q["requirement"])}</span>'
                f'<span class="badge muted">AI: {esc(q["status AI"])}</span>'
                f'<span class="badge {CHECK_STYLE.get(chk, "muted")}">{esc(chk if chk != "-" else "Tidak terpenuhi")}</span>'
                f'</div>')
        body += (f'<div class="quote">“{esc(q["bukti"])}”</div>' if q["bukti"]
                 else '<div class="nil">Tidak ada kutipan bukti.</div>')
        rows.append(body + "</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


# ---------------- output ----------------
results = st.session_state.get("results")
if results:
    ok = [r for r in results if "error" not in r]
    for r in results:
        if "error" in r:
            st.error(f"{r['file']}: {r['error']}")

    if ok:
        ok.sort(key=lambda r: (CAT_ORDER[r["category"]], -r["must_met"], -r["nice_met"], r["perlu_dicek"], -(r["score"] or 0)))
        st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
        st.subheader("Hasil screening")

        stats = "".join(
            f'<div class="stat {CAT_STYLE[cat][0]}"><div class="label">{cat}</div>'
            f'<div class="value">{sum(r["category"] == cat for r in ok)}</div></div>'
            for cat in CAT_ORDER)
        st.markdown(f'<div class="stats">{stats}</div>', unsafe_allow_html=True)

        table = pd.DataFrame([{
            "#": i + 1, "Kandidat": r["file"], "Kategori": r["category"],
            "Must-have": f"{r['must_met']}/{r['must_total']}", "Nice-to-have": f"{r['nice_met']}/{r['nice_total']}",
            "Skor AI": r["score"], "Perlu dicek": r["perlu_dicek"] + len(r["flags"]),
        } for i, r in enumerate(ok)])
        st.dataframe(table, hide_index=True, width="stretch")
        st.markdown('<p class="note">Kategori dihitung dari must-have yang buktinya terverifikasi di CV, bukan dari skor AI. '
                    '<b>Good Fit</b> = semua must-have terpenuhi · <b>Potential</b> = minimal setengah · '
                    '<b>No Fit</b> = kurang dari setengah.</p>', unsafe_allow_html=True)

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.markdown("#### Detail kandidat")
        for i, r in enumerate(ok):
            emoji = CAT_STYLE[r["category"]][1]
            with st.expander(f"{emoji}  {i + 1}. {r['file']} — {r['category']} · must-have {r['must_met']}/{r['must_total']}",
                             expanded=(i == 0)):
                render_candidate(r)

        export = pd.DataFrame([{**{k: v for k, v in r.items() if k != "requirements"},
                                "flags": "; ".join(r["flags"])} for r in ok])
        st.download_button("⬇  Unduh hasil (CSV)", export.to_csv(index=False).encode(), "hasil_screening.csv",
                           width="stretch")
        st.info("Hasil ini adalah shortlist awal. Baca CV kandidat sebelum memutuskan, terutama requirement "
                "berlabel 'Bukti tidak ditemukan di CV', 'Hanya klaim (ringkasan/skill)', dan peringatan.")
