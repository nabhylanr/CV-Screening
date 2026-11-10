"""Uji coba dengan 10 kasus dataset publik (CV EN + terjemahan ID, masing-masing dengan JD-nya).

  python eval_dataset.py show DS03 id          -> tampilkan prompt kasus DS03 versi Indonesia
  python eval_dataset.py show DS03 id --copy   -> sekaligus salin ke clipboard (macOS)
  python eval_dataset.py evaluate              -> hitung metrik dari results_dataset/raw/<tool>/<case>_<lang>_run<n>.txt

Prompt dirakit saat dipanggil dari satu template (src/prompt_dataset.txt) + file di data/dataset/<case>/.
Satu prompt = satu chat BARU. Simpan jawaban AI apa adanya sebagai file .txt dengan nama di atas.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cvscreen import evidence_in_cv, extract_json, normalize_category  # noqa: E402

ROOT = Path(__file__).parent
DS = ROOT / "data" / "dataset"
TEMPLATE = ROOT / "src" / "prompt_dataset.txt"
RAW = ROOT / "results_dataset" / "raw"
LANGS = ["en", "id"]
ORDER = {"Good Fit": 0, "Potential Fit": 1, "No Fit": 2}


def cases() -> pd.DataFrame:
    return pd.read_csv(DS / "labels.csv")


def build_prompt(case_id: str, lang: str) -> str:
    """Rakit prompt dari template + JD dan CV kasus tersebut."""
    case_dir = DS / case_id.upper()
    if not case_dir.is_dir():
        raise SystemExit(f"Kasus {case_id} tidak ada. Pilihan: {', '.join(cases().case_id)}")
    if lang not in LANGS:
        raise SystemExit(f"Bahasa harus salah satu dari {LANGS}")
    jd = (case_dir / "jd.txt").read_text(encoding="utf-8").strip()
    cv = (case_dir / f"cv_{lang}.txt").read_text(encoding="utf-8").strip()
    return TEMPLATE.read_text(encoding="utf-8").replace("{job_description}", jd).replace("{cv}", cv)


def show(args: list[str]):
    if len(args) < 2:
        raise SystemExit("Pakai: python eval_dataset.py show <case_id> <en|id> [--copy]")
    prompt = build_prompt(args[0], args[1])
    print(prompt)
    if "--copy" in args:
        import subprocess
        try:
            subprocess.run(["pbcopy"], input=prompt.encode("utf-8"), check=True)
            print("\n[prompt disalin ke clipboard]", file=sys.stderr)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("\n[gagal menyalin: pbcopy tidak tersedia]", file=sys.stderr)


def collect() -> tuple[pd.DataFrame, list[str]]:
    labels = cases().set_index("case_id")
    rows, issues = [], []
    for f in sorted(RAW.glob("*/*.txt")):
        tool = f.parent.name
        try:
            cid, lang, run = f.stem.split("_")
            run = int(run.replace("run", ""))
            data = extract_json(f.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            issues.append(f"{f.relative_to(ROOT)}: {e}")
            continue
        cv = (DS / cid / f"cv_{lang}.txt").read_text(encoding="utf-8")
        reqs = data.get("requirements", [])
        claims = [r for r in reqs if str(r.get("status", "")).lower() in ("met", "partial")]
        verified = sum(evidence_in_cv(str(r.get("evidence", "") or ""), cv) for r in claims)
        cat = normalize_category(data.get("category", ""))
        if cat is None:
            issues.append(f"{f.relative_to(ROOT)}: kategori tidak valid {data.get('category')!r}")
        rows.append({"tool": tool, "case_id": cid, "lang": lang, "run": run, "category": cat,
                     "score": pd.to_numeric(data.get("score"), errors="coerce"),
                     "label": labels.loc[cid, "label_dataset"], "n_claims": len(claims), "n_verified": verified,
                     "flags": "; ".join(map(str, data.get("flags", []) or []))})
    return pd.DataFrame(rows), issues


def evaluate():
    long, issues = collect()
    for i in issues:
        print("PERINGATAN:", i)
    if long.empty:
        print(f"Belum ada output di {RAW}")
        return
    long["benar"] = long.category == long.label
    long["selisih_1_tingkat"] = (long.category.map(ORDER) - long.label.map(ORDER)).abs() <= 1
    out = []
    for tool, d in long.groupby("tool"):
        m = {"tool": tool, "penilaian": len(d)}
        m["akurasi_vs_label"] = d.benar.mean()
        m["akurasi_toleran_1_tingkat"] = d.selisih_1_tingkat.mean()
        for lang in LANGS:
            m[f"akurasi_{lang}"] = d[d.lang == lang].benar.mean()
        # Konsistensi bahasa: kasus & run yang sama, EN vs ID
        p = d.pivot_table(index=["case_id", "run"], columns="lang", values=["score", "category"], aggfunc="first")
        if {("score", "en"), ("score", "id")} <= set(p.columns):
            both = p.dropna(subset=[("score", "en"), ("score", "id")])
            m["delta_bahasa_skor"] = (both[("score", "en")] - both[("score", "id")]).abs().mean()
            m["kategori_beda_EN_ID"] = (both[("category", "en")] != both[("category", "id")]).mean()
        # Konsistensi antar run
        g = d.groupby(["case_id", "lang"])
        multi = g.run.nunique() > 1
        if multi.any():
            m["rentang_skor_antar_run"] = g.score.agg(lambda s: s.max() - s.min())[multi].mean()
            m["kategori_berubah_antar_run"] = g.category.nunique()[multi].gt(1).mean()
        m["grounding"] = d.n_verified.sum() / d.n_claims.sum() if d.n_claims.sum() else None
        out.append(m)
    metrics = pd.DataFrame(out)
    (ROOT / "results_dataset").mkdir(exist_ok=True)
    long.to_csv(ROOT / "results_dataset" / "scores_long.csv", index=False)
    metrics.to_csv(ROOT / "results_dataset" / "metrics.csv", index=False)
    print(metrics.round(3).to_string(index=False))
    print("\nPer kasus (kategori AI per bahasa vs label):")
    print(long.pivot_table(index=["case_id", "label"], columns=["tool", "lang"], values="category",
                           aggfunc=lambda s: "/".join(sorted(set(s.dropna())))).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "show":
        show(sys.argv[2:])
    elif cmd == "evaluate":
        evaluate()
    else:
        print(__doc__)
