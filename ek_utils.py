# ek_utils.py  – helper functions for EKoder-4o
from __future__ import annotations
import re, pickle, numpy as np, pandas as pd
from numpy.linalg import norm
from pathlib import Path

# ---------- Clinical-impression extraction --------------------------------
def extract_impression(note: str) -> str:
    """Return the doctor's Impression / Assessment / Diagnosis section, or ''. """
    headers = [
        r"impression\s*(?:&|and)?\s*plan",
        r"assessment\s*(?:&|and)?\s*plan",
        r"primary diagnosis",
        r"diagnosis",
        r"differential diagnosis",
        r"impression",
        r"assessment",
        r"clinical impression",
        r"working diagnosis",
        r"discharge diagnosis",
    ]
    for h in headers:
        m = re.search(
            rf"(?:^|\n)\s*{h}\s*:?\s*(.*?)(?=\n[A-Z][a-z]+:|\Z)",
            note, flags=re.I | re.S)
        if m and len(m.group(1)) > 20:
            txt = re.sub(r"(plan|management|treatment)\s*:.*",
                         "", m.group(1), flags=re.I | re.S)
            lines = []
            for ln in txt.splitlines():
                clean = re.sub(r"^[\d\-\•\.]+\s*", "", ln).strip()
                if re.search(r"(diagnosis|impression|[-•])", clean, flags=re.I):
                    lines.append(clean)
                elif re.search(r"(plan|management|treatment)", clean, flags=re.I):
                    break
            return " ".join(lines).strip()
    return ""

# ---------- Acute-keyword boosting ---------------------------------------
ACUTE = {
    'infection': ['fever', 'sepsis', 'uti', 'urinary', 'infection'],
    'cardiac':   ['chest pain', 'stemi', 'nstemi', 'mi', 'acs'],
    'resp':      ['sob', 'dyspnea', 'dyspnoea', 'hypoxia', 'respiratory failure'],
    'abdo':      ['abdominal pain', 'epigastric', 'rlq', 'ruq', 'llq', 'luq'],
    'trauma':    ['fracture', 'injury', 'trauma', 'fall'],
}
def extract_keywords(note: str) -> list[str]:
    n = note.lower()
    return [kw for t in ACUTE.values() for kw in t if kw in n]

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (norm(u) * norm(v)))

def boost_scores(df: pd.DataFrame, note: str) -> pd.DataFrame:
    kws = extract_keywords(note)
    n_lower = note.lower()
    boosted = []
    for _, r in df.iterrows():
        s = r.Similarity
        term = r['ED Short List Term'].lower()
        for kw in kws:
            if kw in term:
                s *= 1.5
        if 'delirium' in n_lower and 'delirium' in term:
            s *= 2
        if 'uti' in n_lower and 'uti' in term:
            s *= 2
        if 'sepsis' in n_lower and 'sepsis' in term:
            s *= 2
        if kws and any(ch in term for ch in ['type 2 diabetes', 'hypertension', 'chronic']):
            s *= 0.5
        boosted.append(s)
    out = df.copy()
    out['Similarity'] = boosted
    return out.sort_values('Similarity', ascending=False)

# ---------- Code list + sentence-transformer embeddings -------------------
from sentence_transformers import SentenceTransformer
EMBED_MODEL = "intfloat/e5-small-v2"
DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE = DATA_DIR / "embeddings.pkl"

def load_codes(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "ED Short": "ED Short List code",
        "Diagnosis": "ED Short List Term",
        "Descriptor": "ED Short List Included conditions",
        "Scale": "Scale",
    })
    df["description"] = (
        df["ED Short List Term"] + ". " +
        df["ED Short List Included conditions"].fillna("")
    )
    return df

def build_embeddings(texts: list[str]) -> np.ndarray:
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            vecs = pickle.load(f)
        if len(vecs) == len(texts):
            return vecs
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    with open(CACHE, "wb") as f:
        pickle.dump(vecs, f)
    return vecs

# ---------- GPT-4o output parser -----------------------------------------
def parse_gpt(resp: str, df: pd.DataFrame) -> list[tuple]:
    valid = set(df['ED Short List code'].astype(str))
    term = dict(zip(df['ED Short List code'], df['ED Short List Term']))
    rows = []
    for line in resp.splitlines():
        m = re.match(r"\d+\.\s*([A-Z0-9\.]+)\s*[—-]\s*(.*)", line)
        if m:
            code, expl = m.groups()
            code = code.strip()
            expl = expl.strip().strip('"').strip("'")
            if code in valid and code != "R69":
                rows.append((code, term.get(code, "Unknown"), expl))
    return rows[:4]
