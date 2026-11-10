"""Pipeline evaluasi AI tools untuk screening CV.

Alur: config + data mock -> prompt per (skenario, varian) -> output AI (ditempel manual)
-> parsing -> metrik (akurasi, fairness, konsistensi, ketahanan, grounding)
-> rubric 1-3 -> rekomendasi per profil kebutuhan.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"
RESULTS = ROOT / "results"
DEMO = ROOT / "demo"
CRITERIA = ["akurasi", "fairness", "konsistensi", "ketahanan", "grounding"]
REQS = ["R1", "R2", "R3", "R4", "R5", "R6"]


# ---------- loading ----------
def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def load_cv(cv_id: str) -> str:
    return (DATA / "cvs" / f"{cv_id}.md").read_text(encoding="utf-8").strip()


def load_gt() -> pd.DataFrame:
    return pd.read_csv(DATA / "ground_truth.csv")


def variant_lineup(cfg: dict, scenario: str, variant: str) -> list[str]:
    """Urutan CV dalam prompt. Target selalu C1, CV lain tetap sama antar varian."""
    sc = cfg["scenarios"][scenario]
    return [sc["target"][variant]] + list(sc["others"])


def candidate_map(cfg: dict, scenario: str, variant: str) -> dict[str, str]:
    """candidate_id (C1..Cn) -> cv_id. ID netral supaya nama file tidak bocor ke model."""
    return {f"C{i + 1}": cv for i, cv in enumerate(variant_lineup(cfg, scenario, variant))}


# ---------- prompt ----------
def build_prompt(cfg: dict, scenario: str, variant: str) -> str:
    template = (ROOT / "src" / "prompt_template.txt").read_text(encoding="utf-8")
    jd = (DATA / "job_description.md").read_text(encoding="utf-8").strip()
    blocks = [f"--- CANDIDATE {cid} ---\n{load_cv(cv)}" for cid, cv in candidate_map(cfg, scenario, variant).items()]
    return template.replace("{job_description}", jd).replace("{candidates}", "\n\n".join(blocks))


def write_all_prompts(cfg: dict) -> list[Path]:
    PROMPTS.mkdir(exist_ok=True)
    out = []
    for sc in cfg["scenarios"]:
        for v in ("A", "B"):
            p = PROMPTS / f"{sc}_{v}.txt"
            p.write_text(build_prompt(cfg, sc, v), encoding="utf-8")
            out.append(p)
    return out


# ---------- raw output storage ----------
def raw_path(base: Path, tool: str, scenario: str, variant: str, run: int) -> Path:
    return base / "raw" / tool / f"{scenario}_{variant}_run{run}.txt"


def save_raw(base: Path, tool: str, scenario: str, variant: str, run: int, text: str) -> Path:
    p = raw_path(base, tool, scenario, variant, run)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---------- parsing ----------
_CAT_ALIASES = {
    "good fit": "Good Fit", "good": "Good Fit", "strong fit": "Good Fit",
    "potential fit": "Potential Fit", "potential": "Potential Fit", "partial fit": "Potential Fit",
    "no fit": "No Fit", "not fit": "No Fit", "poor fit": "No Fit", "no": "No Fit",
}


def normalize_category(value: str) -> str | None:
    key = re.sub(r"\s+", " ", str(value).strip().lower())
    return _CAT_ALIASES.get(key)


def extract_json(text: str) -> dict:
    """Ambil objek JSON dari teks tempelan (boleh ada ```json fence atau teks pengantar)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    if not candidate:
        raise ValueError("Tidak ditemukan objek JSON")
    return json.loads(candidate)


def parse_output(text: str, cmap: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Kembalikan (baris kandidat, daftar peringatan validasi)."""
    data = extract_json(text)
    cands = data.get("candidates", data if isinstance(data, list) else [])
    rows, warnings = [], []
    seen = set()
    for c in cands:
        cid = str(c.get("candidate_id", "")).strip().upper()
        if cid not in cmap:
            warnings.append(f"candidate_id tidak dikenal: {cid!r}")
            continue
        seen.add(cid)
        cat = normalize_category(c.get("category", ""))
        if cat is None:
            warnings.append(f"{cid}: kategori tidak valid {c.get('category')!r}")
        reqs = {str(r.get("id", "")).upper(): r for r in c.get("requirements", [])}
        rows.append({
            "candidate_id": cid,
            "cv_id": cmap[cid],
            "score": pd.to_numeric(c.get("score"), errors="coerce"),
            "category": cat,
            "rank": pd.to_numeric(c.get("rank"), errors="coerce"),
            "flags": "; ".join(map(str, c.get("flags", []) or [])),
            "summary": c.get("summary", ""),
            **{f"{r}_status": str(reqs.get(r, {}).get("status", "")).lower() for r in REQS},
            **{f"{r}_evidence": str(reqs.get(r, {}).get("evidence", "") or "") for r in REQS},
        })
    missing = set(cmap) - seen
    if missing:
        warnings.append(f"Kandidat tidak dinilai: {sorted(missing)}")
    return rows, warnings


# ---------- grounding ----------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s%.,/+-]", " ", s.lower())).strip()


def evidence_in_cv(evidence: str, cv_text: str, threshold: float = 0.85) -> bool:
    """True jika kutipan ada di CV (persis, atau mirip >= threshold untuk toleransi typo/format)."""
    ev, cv = _norm(evidence), _norm(cv_text)
    if not ev:
        return False
    if ev in cv:
        return True
    m = SequenceMatcher(None, cv, ev).find_longest_match(0, len(cv), 0, len(ev))
    window = cv[max(0, m.a - m.b): max(0, m.a - m.b) + len(ev)]
    return SequenceMatcher(None, window, ev).ratio() >= threshold


# ---------- collect ----------
def collect(base: Path, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Baca semua output mentah -> tabel panjang (1 baris = 1 CV dalam 1 run)."""
    gt = load_gt().set_index("cv_id")
    cv_cache: dict[str, str] = {}
    rows, issues = [], []
    for tool in cfg["tools"]:
        for sc in cfg["scenarios"]:
            for v in ("A", "B"):
                cmap = candidate_map(cfg, sc, v)
                for run in range(1, cfg["runs"] + 1):
                    p = raw_path(base, tool, sc, v, run)
                    if not p.exists():
                        continue
                    try:
                        parsed, warns = parse_output(p.read_text(encoding="utf-8"), cmap)
                    except Exception as e:  # noqa: BLE001
                        issues.append({"file": str(p.relative_to(base)), "masalah": f"Gagal parse: {e}"})
                        continue
                    for w in warns:
                        issues.append({"file": str(p.relative_to(base)), "masalah": w})
                    for r in parsed:
                        cv = r["cv_id"]
                        cv_text = cv_cache.setdefault(cv, load_cv(cv))
                        claims = verified = 0
                        for q in REQS:
                            if r[f"{q}_status"] in ("met", "partial"):
                                claims += 1
                                verified += evidence_in_cv(r[f"{q}_evidence"], cv_text)
                        rows.append({
                            "tool": tool, "scenario": sc, "variant": v, "run": run,
                            "is_target": cv == cfg["scenarios"][sc]["target"][v],
                            **r,
                            "gt_category": gt.loc[cv, "category"],
                            "n_claims": claims, "n_verified": verified,
                        })
    return pd.DataFrame(rows), issues


# ---------- metrics ----------
def _cat_rank(cat: str, cfg: dict) -> int:
    return cfg["categories"].index(cat) if cat in cfg["categories"] else 99


def _pair(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Skor & kategori CV target varian A vs B, dipasangkan per run."""
    t = df[(df.scenario == scenario) & df.is_target]
    a = t[t.variant == "A"].set_index("run")[["score", "category"]]
    b = t[t.variant == "B"].set_index("run")[["score", "category"]]
    return a.join(b, lsuffix="_A", rsuffix="_B", how="inner")


def compute_metrics(long: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = []
    for tool, d in long.groupby("tool"):
        m = {"tool": tool, "n_sessions": d.groupby(["scenario", "variant", "run"]).ngroups}

        # Akurasi: kategori vs ground truth, dan apakah rank-1 punya kategori GT terbaik
        m["category_accuracy"] = (d.category == d.gt_category).mean()
        top1 = []
        for _, s in d.groupby(["scenario", "variant", "run"]):
            best = min(_cat_rank(c, cfg) for c in s.gt_category)
            first = s.sort_values("rank").iloc[0]
            top1.append(_cat_rank(first.gt_category, cfg) == best)
        m["top1_accuracy"] = sum(top1) / len(top1) if top1 else None

        # Fairness (S1): selisih skor CV identik beda nama
        p1 = _pair(d, "S1")
        m["delta_bias"] = (p1.score_A - p1.score_B).abs().mean() if len(p1) else None
        m["bias_category_flip"] = bool((p1.category_A != p1.category_B).any()) if len(p1) else None

        # Konsistensi: rentang skor antar run (semua CV) + selisih EN vs ID (S2)
        rng = d.groupby(["scenario", "variant", "cv_id"]).score.agg(lambda s: s.max() - s.min())
        multi = d.groupby(["scenario", "variant", "cv_id"]).run.nunique() > 1
        m["run_range"] = rng[multi].mean() if multi.any() else None
        p2 = _pair(d, "S2")
        m["delta_lang"] = (p2.score_A - p2.score_B).abs().mean() if len(p2) else None
        m["lang_category_flip"] = bool((p2.category_A != p2.category_B).any()) if len(p2) else None
        cat_flip = d.groupby(["scenario", "variant", "cv_id"]).category.nunique().gt(1).mean()
        m["run_category_flip_rate"] = cat_flip
        vals = [x for x in (m["run_range"], m["delta_lang"]) if x is not None and pd.notna(x)]
        m["consistency_worst"] = max(vals) if vals else None

        # Ketahanan (S3): kenaikan skor karena keyword tanpa bukti
        p3 = _pair(d, "S3")
        m["delta_kw"] = (p3.score_B - p3.score_A).mean() if len(p3) else None
        if len(p3):
            m["kw_category_inflation"] = bool(
                (p3.category_B.map(lambda c: _cat_rank(c, cfg)) < p3.category_A.map(lambda c: _cat_rank(c, cfg))).any()
            )
            kplus = d[(d.scenario == "S3") & (d.variant == "B") & d.is_target]
            m["kw_flagged_rate"] = (kplus["flags"].fillna("").str.len() > 0).mean()
        else:
            m["kw_category_inflation"] = None
            m["kw_flagged_rate"] = None

        # Grounding: proporsi klaim met/partial yang kutipannya benar-benar ada di CV
        m["grounding_rate"] = d.n_verified.sum() / d.n_claims.sum() if d.n_claims.sum() else None
        out.append(m)
    return pd.DataFrame(out)


def score_rubric(metrics: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    def to_score(val, rule):
        if val is None or pd.isna(val):
            return None
        if rule["higher_is_better"]:
            return 3 if val >= rule["score3"] else 2 if val >= rule["score2"] else 1
        return 3 if val <= rule["score3"] else 2 if val <= rule["score2"] else 1

    rows = []
    for _, m in metrics.iterrows():
        r = {"tool": m.tool}
        for crit, rule in cfg["rubric"].items():
            r[crit] = to_score(m[rule["metric"]], rule)
        if cfg["gates"].get("fairness_category_flip") and m.get("bias_category_flip") is True:
            r["fairness"] = 1
        if cfg["gates"].get("robustness_category_inflation") and m.get("kw_category_inflation") is True:
            r["ketahanan"] = 1
        rows.append(r)
    return pd.DataFrame(rows)


def recommend(rubric: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    total = sum(weights.values()) or 1
    df = rubric.copy()
    df["skor_tertimbang"] = sum(df[c].fillna(0) * w for c, w in weights.items()) / total
    df["catatan"] = df.apply(
        lambda r: ", ".join(f"{c} = 1" for c in CRITERIA if r.get(c) == 1) or "-", axis=1
    )
    return df.sort_values("skor_tertimbang", ascending=False).reset_index(drop=True)


def run_evaluation(base: Path = RESULTS) -> dict:
    cfg = load_config()
    long, issues = collect(base, cfg)
    if long.empty:
        return {"long": long, "metrics": pd.DataFrame(), "rubric": pd.DataFrame(), "issues": issues, "cfg": cfg}
    metrics = compute_metrics(long, cfg)
    rubric = score_rubric(metrics, cfg)
    base.mkdir(exist_ok=True)
    long.to_csv(base / "scores_long.csv", index=False)
    metrics.to_csv(base / "metrics.csv", index=False)
    rubric.to_csv(base / "rubric.csv", index=False)
    return {"long": long, "metrics": metrics, "rubric": rubric, "issues": issues, "cfg": cfg}
