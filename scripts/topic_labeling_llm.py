from pathlib import Path
import os
from typing import Optional
import unicodedata
import pandas as pd
from openai import OpenAI

# ---------------- Paths ----------------
# Resolve paths relative to repo root (…/scripts/topic_labeling_llm.py -> repo root is parent of scripts)
REPO_ROOT = Path(__file__).resolve().parents[1]

# Input files produced by the main pipeline
BASE_DIR = REPO_ROOT / "outputs_textual_factors"
TRIMMED_PATH = BASE_DIR / "topics_words_trimmed.csv"
IMPORTANCE_PATH = BASE_DIR / "topic_importances.csv"

# Save "permanent" result (safe to commit)
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "topic_labels_with_importance_llm.csv"

# ---------------- Settings ----------------
# Only label the N most important topics (by overall importance). Set to None to label all.
TOP_N = 20

# Optional: if True, the output will include importance renormalized within TOP_N
ADD_TOPN_RENORM = True

# Model for labeling
LABEL_MODEL = "gpt-4.1-mini"  # change to "gpt-4.1" for a stronger (slower/more expensive) labeler

# A few generic words to downweight for heuristic fallback labels
GENERIC = {"bank", "banks", "financial", "market", "markets", "year", "data", "risk"}


# ---------------- OpenAI ----------------

def _load_env_file(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines). Does not override already-set env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _get_api_key() -> str:
    """Get API key from env (preferred) or from .env at repo root."""
    # 1) Preferred: environment variable (PyCharm Run Configuration sets this)
    api_key = os.getenv("OPENAI_API_KEY")

    # 2) Fallback: load .env from repo root (useful if you run from terminal)
    if not api_key:
        _load_env_file(REPO_ROOT / ".env")
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in PyCharm Run Configuration → Environment variables, "
            "or put OPENAI_API_KEY=... in a local .env file in the repo root (do NOT commit it)."
        )

    # Strip accidental whitespace/newlines
    api_key = api_key.strip()

    # Validate: keys should be ASCII and start with sk-
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as e:
        raise RuntimeError(
            "OPENAI_API_KEY contains non-ASCII characters (often from copying). "
            "Please re-paste the key in PyCharm and ensure there are no extra characters."
        ) from e

    if not api_key.startswith("sk-"):
        raise RuntimeError(
            "OPENAI_API_KEY does not look like a valid key (expected to start with 'sk-'). "
            "Please paste the correct key from the OpenAI dashboard."
        )

    if len(api_key) < 20:
        raise RuntimeError(
            "OPENAI_API_KEY looks too short. Please paste the full key value."
        )

    return api_key


api_key = _get_api_key()
client = OpenAI(api_key=api_key)


# ---------------- Helpers ----------------
def heuristic_label(top_words_pretty: str, max_words: int = 3) -> str:
    """Fallback if the API fails: pick a few non-generic keywords."""
    if not isinstance(top_words_pretty, str):
        return ""
    words = [w.strip() for w in top_words_pretty.split(",") if w.strip()]
    if not words:
        return ""
    specific = [w for w in words if w.lower() not in GENERIC]
    if not specific:
        specific = words
    return " ".join(specific[:max_words])


def sanitize_for_api(text: str) -> str:
    """Convert text to a safe ASCII-only form for environments that choke on Unicode."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Normalize common PDF ligatures/superscripts etc.
    # NFKD breaks ligatures like 'ﬁ' into 'fi' and decomposes accents.
    text = unicodedata.normalize("NFKD", text)

    # Drop any remaining non-ASCII characters (e.g., '¹', 'ø')
    text = text.encode("ascii", "ignore").decode("ascii")

    # Collapse repeated whitespace
    return " ".join(text.split())


def generate_llm_label(top_words_pretty: str) -> tuple[str, bool]:
    """Use an LLM to produce a risk-centric, emerging-risk style label.

    Returns:
        (label, ok) where ok=True means the label came from the LLM, ok=False means fallback was used.
    """
    if not isinstance(top_words_pretty, str) or not top_words_pretty.strip():
        return "", False

    instructions = (
        "You are labeling topics from bank annual reports with the goal of identifying EMERGING RISKS.\n"
        "You are given ordered topic keywords (most important first).\n"
        "Return ONLY a concise risk-centric label of 2–5 words that could appear in a risk report.\n"
        "Prefer specific themes (e.g., 'Commercial real estate stress', 'Cyber resilience', "
        "'Sanctions & compliance risk', 'Climate transition risk', 'Funding & liquidity pressure').\n"
        "Avoid generic labels like 'bank', 'financial', 'risk', 'market' unless needed for clarity.\n"
        "Do not add explanations, punctuation, quotes, or numbering."
    )

    safe_words = sanitize_for_api(top_words_pretty)
    input_text = f"Top keywords:\n{safe_words}\n\nLabel:"

    try:
        resp = client.responses.create(
            model=LABEL_MODEL,
            instructions=instructions,
            input=input_text,
        )
        label = resp.output_text.strip()
        # Safety trim in case the model outputs something longer
        label = " ".join(label.split()[:8])
        return label, True
    except Exception as e:
        print(f"LLM label failed -> {e}")
        return heuristic_label(top_words_pretty), False


# ---------------- Main ----------------
def main():
    if not TRIMMED_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {TRIMMED_PATH}. Run the trim script first (scripts/trim_topics_words.py)."
        )
    if not IMPORTANCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {IMPORTANCE_PATH}. Run the main pipeline first (Main_with_new embedding.py)."
        )

    trimmed = pd.read_csv(TRIMMED_PATH)
    imps = pd.read_csv(IMPORTANCE_PATH)

    merged = trimmed.merge(imps, left_on="topic", right_on="cluster", how="left")

    # Normalize importance across all topics (sums to 1 across all topics)
    total = merged["leading_importance"].sum()
    merged["importance_norm"] = merged["leading_importance"] / total
    merged["importance_raw"] = merged["leading_importance"]

    # Only label the TOP_N most important topics (fast)
    if TOP_N is not None:
        merged = merged.sort_values("importance_norm", ascending=False).head(TOP_N).copy()

    if ADD_TOPN_RENORM and TOP_N is not None:
        merged["importance_topN_norm"] = merged["importance_norm"] / merged["importance_norm"].sum()

    # LLM labels
    labels = []
    llm_ok = []
    n = len(merged)
    for i, (_, row) in enumerate(merged.iterrows(), start=1):
        # Debug: detect the first non-ASCII topic word string (helps diagnose \xf8/ø errors)
        s = row.get("top_words_pretty", "")
        try:
            str(s).encode("ascii")
        except UnicodeEncodeError:
            print("NON-ASCII top_words_pretty:", repr(s))
            for ch in str(s):
                if ord(ch) > 127:
                    print("  offending char:", ch, "codepoint:", ord(ch))
            # Continue; we still attempt labeling (or fallback) below

        label, ok = generate_llm_label(s)
        labels.append(label)
        llm_ok.append(ok)
        if i % 10 == 0 or i == n:
            print(f"Labeled {i}/{n}")

    merged["llm_label"] = labels
    merged["llm_ok"] = llm_ok
    merged["heuristic_label"] = merged["top_words_pretty"].apply(heuristic_label)

    cols = [
        "topic",
        "llm_label",
        "llm_ok",
        "heuristic_label",
        "importance_norm",
    ]
    if ADD_TOPN_RENORM and TOP_N is not None:
        cols.append("importance_topN_norm")
    cols += [
        "importance_raw",
        "top_words_pretty",
    ]

    out = merged[cols]
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
