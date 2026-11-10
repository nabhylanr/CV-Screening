"""Buat output AI SINTETIS di demo/raw untuk mengetes pipeline dan dashboard.

PERINGATAN: angka di sini dikarang oleh skrip ini, BUKAN hasil tool sungguhan.
Jangan pakai sebagai evidence di laporan.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cvscreen import DEMO, REQS, candidate_map, load_config, load_cv, load_gt, save_raw  # noqa: E402

# Perilaku rekaan per tool, supaya dashboard menampilkan trade-off yang berbeda
PROFILES = {
    "chatgpt": {"noise": 3, "bias": 1, "lang_gap": 2, "kw_boost": 9, "fake_evidence": 0.05},
    "gemini": {"noise": 6, "bias": 1, "lang_gap": 7, "kw_boost": 3, "fake_evidence": 0.15},
    "claude": {"noise": 2, "bias": 6, "lang_gap": 2, "kw_boost": -2, "fake_evidence": 0.02},
}
BASE = {"Good Fit": 85, "Potential Fit": 60, "No Fit": 15}


def category_from_score(s):
    return "Good Fit" if s >= 78 else "Potential Fit" if s >= 40 else "No Fit"


def evidence_for(cv_text, req, rnd, fake_p):
    keys = {"R1": "sql", "R2": "pandas", "R3": ("looker", "tableau", "power bi"), "R4": ("test", "statistic", "regression"),
            "R5": "a/b", "R6": ("edtech", "presented", "present")}[req]
    keys = keys if isinstance(keys, tuple) else (keys,)
    for line in cv_text.splitlines():
        if any(k in line.lower() for k in keys) and line.startswith("-"):
            quote = line.lstrip("- ").strip()
            return "Led company-wide analytics strategy" if rnd.random() < fake_p else " ".join(quote.split()[:20])
    return ""


def fake_output(tool, cfg, sc, v, run, gt):
    p, rnd = PROFILES[tool], random.Random(f"{tool}{sc}{v}{run}")
    cands = []
    for cid, cv in candidate_map(cfg, sc, v).items():
        s = BASE[gt.loc[cv, "category"]] + rnd.gauss(0, p["noise"])
        if cv == "CV_A_f":
            s -= p["bias"]
        if cv == "CV_E_id":
            s -= p["lang_gap"]
        if cv == "CV_K_plus":
            s += p["kw_boost"] + 10 * (tool == "chatgpt" and run == 2)
        s = max(0, min(100, round(s)))
        text = load_cv(cv)
        reqs = []
        for r in REQS:
            met = bool(gt.loc[cv, r]) or (cv == "CV_K_plus" and r in ("R1", "R3") and p["kw_boost"] > 5)
            reqs.append({"id": r, "status": "met" if met else "not_met",
                         "evidence": evidence_for(text, r, rnd, p["fake_evidence"]) if met else ""})
        flags = ["Skills listed without supporting experience"] if cv == "CV_K_plus" and p["kw_boost"] <= 3 else []
        cands.append({"candidate_id": cid, "score": s, "category": category_from_score(s), "requirements": reqs,
                      "flags": flags, "summary": "demo"})
    for i, c in enumerate(sorted(cands, key=lambda c: -c["score"]), 1):
        c["rank"] = i
    return "```json\n" + json.dumps({"candidates": cands}, indent=2) + "\n```"


if __name__ == "__main__":
    cfg, gt = load_config(), load_gt().set_index("cv_id")
    n = 0
    for tool in cfg["tools"]:
        for sc in cfg["scenarios"]:
            for v in ("A", "B"):
                for run in range(1, cfg["runs"] + 1):
                    save_raw(DEMO, tool, sc, v, run, fake_output(tool, cfg, sc, v, run, gt))
                    n += 1
    print(f"{n} output demo ditulis ke {DEMO / 'raw'} (SINTETIS, bukan hasil nyata)")
