from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Signals:
    text: str
    tokens: List[str]
    keywords: List[str]
    entities: List[Dict[str, str]]


def simple_normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_nltk(text: str) -> List[str]:
    """
    NLTK tokenizer with a safe fallback if punkt isn't downloaded.
    """
    try:
        import nltk
        from nltk.tokenize import word_tokenize

        return [t for t in word_tokenize(text) if t.strip()]
    except Exception:
        # Fallback: whitespace tokenization
        return [t for t in re.split(r"\s+", text) if t]


def extract_keywords(tokens: List[str], *, max_k: int = 12) -> List[str]:
    """
    Lightweight keyword extraction: keep longer alpha tokens, de-duplicate, preserve order.
    """
    seen = set()
    out: List[str] = []
    for t in tokens:
        t0 = re.sub(r"[^A-Za-z0-9_]", "", t).lower()
        if len(t0) < 4:
            continue
        if t0 in seen:
            continue
        seen.add(t0)
        out.append(t0)
        if len(out) >= max_k:
            break
    return out


def spacy_entities(text: str) -> List[Dict[str, str]]:
    """
    spaCy NER. If no model is installed, returns [].
    """
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            # No model installed; avoid hard-failing.
            return []
        doc = nlp(text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    except Exception:
        return []


def build_signals(text: str) -> Signals:
    text_n = simple_normalize(text)
    toks = tokenize_nltk(text_n)
    kws = extract_keywords(toks)
    ents = spacy_entities(text_n)
    return Signals(text=text_n, tokens=toks, keywords=kws, entities=ents)
