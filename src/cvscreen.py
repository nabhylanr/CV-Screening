"""Fungsi bantu bersama: parsing JSON output AI, normalisasi kategori, dan cek bukti (grounding)."""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

_CAT_ALIASES = {
    "good fit": "Good Fit", "good": "Good Fit", "strong fit": "Good Fit",
    "potential fit": "Potential Fit", "potential": "Potential Fit", "partial fit": "Potential Fit",
    "no fit": "No Fit", "not fit": "No Fit", "poor fit": "No Fit", "no": "No Fit",
}


def normalize_category(value) -> str | None:
    key = re.sub(r"\s+", " ", str(value).strip().lower())
    return _CAT_ALIASES.get(key)


def extract_json(text: str) -> dict:
    """Ambil objek JSON dari teks jawaban AI (boleh ada ```json fence atau teks pengantar)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    if not candidate:
        raise ValueError("Tidak ditemukan objek JSON")
    return json.loads(candidate)


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
