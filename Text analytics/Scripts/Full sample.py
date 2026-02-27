import os
import sys
import re
import nltk
from itertools import chain
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from collections import Counter
from openai import OpenAI

# Read API key from environment variable (set in PyCharm Run Configuration).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise RuntimeError(
        "Missing OPENAI_API_KEY environment variable. "
        "Set it in Run → Edit Configurations → Environment variables."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------
# Directions
# -----------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                 # .../Text analytics/Scripts
TEXT_ANALYTICS_DIR = os.path.dirname(SCRIPT_DIR)
NLTK_DATA_DIR = os.path.join(TEXT_ANALYTICS_DIR, "nltk_data")
CONG_REP_DIR = os.path.join(TEXT_ANALYTICS_DIR, "Cong et al. rep")      # .../Text analytics/Cong et al. rep
nltk.data.path = [NLTK_DATA_DIR]

for p in (CONG_REP_DIR, TEXT_ANALYTICS_DIR, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# --- helper functions from engine.py ---
from engine import (
    clean_and_normalize_text,    # cleans and normalizes the 'content' text
    calculate_word_frequencies,  # tokenizes + counts words per document
)

# --- helper classes/functions from TextualFactors.py ---
from TextualFactors import (
    EmbeddingCluster,
    NeighborFinder,
    TextualFactors,
    transfer_document_topics,
    transfer_topic_words,
    transfer_sigular_values,
    transfer_topic_importances,
)

# ============================================================
# 0. SETTINGS: folders, page ranges, etc.
# ============================================================

# This script lives in: <repo>/Text analytics/Scripts/
# Reports live in:      <repo>/Text analytics/Reports/
reports_folder = os.path.join(TEXT_ANALYTICS_DIR, "Reports")
print("Reports folder:", reports_folder)

# Your own page_ranges (copied from your notebook)
# ============================================================
# PAGE RANGES — ORGANIZED BY YEAR → BANK
# NOTE: ranges are PDF page indices (0-based, end exclusive)
# ============================================================


# Page ranges by year (dedented to top-level, not inside norm_for_match)
page_ranges_2015 = {

}

page_ranges_2016 = {

}
page_ranges_2017 = {

}
page_ranges_2018 = {

}

page_ranges_2019 = {

}

page_ranges_2020 = {

}
page_ranges_2021 = {


}

page_ranges_2022 = {

}

page_ranges_2023 = {

}

page_ranges_2024 = {


}

# Combined mapping used by the pipeline
page_ranges = {}
for _d in (
    page_ranges_2015,
    page_ranges_2016,
    page_ranges_2017,
    page_ranges_2018,
    page_ranges_2019,
    page_ranges_2020,
    page_ranges_2021,
    page_ranges_2022,
    page_ranges_2023,
    page_ranges_2024,
):
    page_ranges.update(_d)


# ============================================================
# 1. LOAD TEXT FROM PDF´s
# ============================================================

def load_report_paragraphs(reports_folder, page_ranges, strict=True):
    report_paragraphs = []
    report_paragraphs_source = []

    print(f"Looking for PDFs in: {reports_folder}")
    # Exact lookup: filenames in Reports/ must match keys in page_ranges
    page_ranges_exact = page_ranges

    for path, dirs, files in os.walk(reports_folder):
        pdfs = [file for file in files if file.lower().endswith(".pdf")]
        if not pdfs:
            continue
        print("Found PDFs:", pdfs)

        for _file in pdfs:
            print(f"Processing {_file}...")
            full_path = os.path.join(path, _file)

            # Decide which pages to process (STRICT: require explicit page ranges)
            file_key = _file
            if file_key not in page_ranges_exact:
                if strict:
                    raise ValueError(
                        f"File '{_file}' not found in page_ranges (exact match required). "
                        "Add an explicit page range for this PDF (no defaults) or rename the PDF to match the key."
                    )
                else:
                    # When running a filtered subset (e.g., only one year), ignore other PDFs in the folder
                    continue

            pages_to_process = page_ranges_exact[file_key]

            with fitz.open(full_path) as doc:
                total_pages = len(doc)

                # If None -> all pages
                if pages_to_process is None:
                    pages_to_process = range(total_pages )

                # Handle possible negative page indices
                actual_pages = []
                for page_num in pages_to_process:
                    if isinstance(page_num, int):
                        if page_num < 0:
                            actual_page = total_pages + page_num
                        else:
                            actual_page = page_num

                        if 0 <= actual_page < total_pages:
                            actual_pages.append(actual_page)

                # Extract text blocks from chosen pages
                for page_num in actual_pages:
                    page = doc[page_num]
                    blocks = [x[4] for x in page.get_text("blocks")]
                    blocks = [block.strip() for block in blocks if block.strip()]

                    if blocks:
                        report_paragraphs.extend(blocks)
                        report_paragraphs_source.extend([_file] * len(blocks))

    return report_paragraphs, report_paragraphs_source

# Output:
# After this section we have two parallel lists:
# 1) report_paragraphs        -> all extracted text paragraphs (strings)
# 2) report_paragraphs_source -> which PDF each paragraph came from
# Both lists have the same length; each index represents one paragraph.

# ============================================================
# 2. BUILD DOCUMENT DATAFRAME
# ============================================================

# Short summary:
# - Combine paragraph-level text into one document per PDF (bank × year).
# - Parse bank and year from filenames (expects bank_YYYY.pdf; fallback YYYY_bank_group.pdf).
# - Validate that bank/year parsing succeeds (fail fast if not).
# - Assign a stable integer document ID for downstream analysis.


def build_document_dataframe(report_paragraphs, report_sources):
    df = pd.DataFrame({"file": report_sources, "content": report_paragraphs})

    df_grouped = df.groupby("file", as_index=False).agg({"content": lambda texts: "\n".join(texts)})
    if df_grouped.empty:
        return pd.DataFrame(columns=["file", "content", "year", "bank", "document"])

    def _parse_bank_year(fname: str):
        base = os.path.basename(fname)
        m = re.match(r"^(?P<bank>.+?)_(?P<year>\d{4})\.pdf$", base, flags=re.IGNORECASE)
        if m:
            return m.group("bank"), int(m.group("year"))
        m = re.match(r"^(?P<year>\d{4})_(?P<bank>.+?)_group\.pdf$", base, flags=re.IGNORECASE)
        if m:
            return m.group("bank"), int(m.group("year"))
        return None, None

    parsed = df_grouped["file"].apply(_parse_bank_year)
    df_grouped["bank"] = parsed.apply(lambda x: x[0])
    df_grouped["year"] = parsed.apply(lambda x: x[1]).astype("Int64")

    if df_grouped["bank"].isna().any() or df_grouped["year"].isna().any():
        bad = df_grouped[df_grouped["bank"].isna() | df_grouped["year"].isna()]["file"].tolist()
        raise ValueError(f"Could not parse bank/year from filenames: {bad}")

    # Optional: normalize bank names
    df_grouped["bank"] = df_grouped["bank"].str.lower()

    df_grouped = df_grouped.sort_values(["year", "bank", "file"]).reset_index(drop=True)
    df_grouped["document"] = np.arange(len(df_grouped))
    return df_grouped


# Output:
# - DataFrame with one row per PDF containing:
#   * document : integer document ID (0, 1, 2, ...)
#   * content  : full text of the PDF (all paragraphs joined)
#   * file     : source PDF filename
#   * bank     : bank identifier parsed from filename (lowercased)
#   * year     : year parsed from filename

# ============================================================
# 3. TEXT PREPROCESSING (engine.py) + TOKEN FILTERING
# ============================================================
# Short summary:
# - Clean and normalize the raw text in `content` (whitespace, encoding, formatting).
# - Tokenize text and compute per-document word counts.
# - Remove obvious junk tokens (short tokens, non-letters, boilerplate/bank identifiers).
# - Remove rare tokens across the corpus using a minimum document-frequency threshold.

# --- DF / vocab filtering settings ---
TOKEN_MIN_LEN = 3  # drop tokens shorter than 3 characters.
MIN_DF = 5            # token must appear in at least MIN_DF documents

EXTRA_DROP_WORDS = {
    # Generic report boilerplate
    "annual", "report", "reports", "group", "plc", "page", "pages", "section", "chapter",
    "table", "tables", "figure", "figures", "statement", "statements",
    "introduction", "overview", "note", "notes", "euro",

    # Bank names / identifiers (extend as needed)
    "barclays", "seb", "ubs", "ing", "danske", "deutschebank", "deutsche", "bank",
    "bnp", "paribas", "fortis", "oppohjola", "op", "pohjola",

    # Common legal entities
    "limited", "ltd", "ab", "asa", "as",
}


def _basic_token_filter(tokens: list[str]) -> list[str]:
    """Remove obvious junk tokens before df-based filtering."""
    out: list[str] = []
    for t in tokens:
        if not isinstance(t, str):
            continue
        t = t.strip().lower()
        if not t:
            continue
        if len(t) < TOKEN_MIN_LEN:
            continue
        # keep only alphabetic tokens -> removes '_' and mixed punctuation/nums
        if not t.isalpha():
            continue
        if t in EXTRA_DROP_WORDS:
            continue
        out.append(t)
    return out


def _df_filter_tokens(df: pd.DataFrame, tokens_col: str, min_df: int) -> pd.DataFrame:
    """Apply document-frequency filtering (min_df only) across the corpus."""
    df = df.copy()

    df_counter = Counter()
    for toks in df[tokens_col]:
        df_counter.update(set(toks))

    allowed = {tok for tok, dfi in df_counter.items() if dfi >= min_df}

    df[tokens_col] = df[tokens_col].apply(lambda toks: [t for t in toks if t in allowed])
    return df


def preprocess_text_and_tokens(
    df: pd.DataFrame,
    text_col: str = "content",
    tokens_col: str = "tokens",
    min_df: int = MIN_DF,
) -> pd.DataFrame:
    """
    Canonical preprocessing step used everywhere in this pipeline.

    1) Clean and normalize raw text (engine.clean_and_normalize_text)
    2) Tokenize + count word frequencies (engine.calculate_word_frequencies)
    3) Apply basic token cleanup (length/alpha/stop words)
    4) Apply document-frequency filtering (min_df only)

    Returns a new DataFrame with cleaned text, filtered tokens, and word_freq.
    """

    df = df.copy()

    # 1) Clean / normalize raw text
    df = clean_and_normalize_text(df, column_name=text_col)

    # 2) Tokenize + count word frequencies
    df = calculate_word_frequencies(df, text_column=text_col)

    # Preserve raw tokens before any filtering (for QC / before-vs-after comparisons)
    if "tokens_raw" not in df.columns:
        df["tokens_raw"] = df[tokens_col].apply(lambda x: list(x) if isinstance(x, list) else [])

    # 3) Per-document token cleanup
    df[tokens_col] = df[tokens_col].apply(_basic_token_filter)

    # 4) DF-based filtering across corpus
    df = _df_filter_tokens(df, tokens_col=tokens_col, min_df=min_df)

    return df

# Output:
# - Returns the same DataFrame with additional / updated columns:
#   * content   : cleaned & normalized text
#   * tokens    : filtered list of tokens per document
#   * word_freq : Counter/dict of word -> count per document
#   * tokens_raw: unfiltered tokens directly from tokenization

# Note: We only use a subset of functions from engine.py.
# The unused utilities (daily aggregation, long-format by date) are meant for true time-series text data,
# but our documents are grouped by bank-year, not by calendar dates, so these functions are not needed here.

# ============================================================
# 4. OPENAI EMBEDDING FUNCTION
# ============================================================

def train_openai_embeddings(df, model_name="text-embedding-3-large"):
    """
    Build word embeddings using OpenAI's embedding API.
    Trains on paragraph-level tokens.
    """
    vocab = sorted(set(chain.from_iterable(df["tokens"].tolist())))
    print(f"Vocabulary size: {len(vocab)} words")

    batch_size = 200
    embeddings = []

    for i in range(0, len(vocab), batch_size):
        batch = vocab[i:i+batch_size]
        response = client.embeddings.create(
            model=model_name,
            input=batch
        )

        # --- SAFETY CHECK: vocab ↔ embedding alignment ---
        if len(response.data) != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch in batch {i//batch_size + 1}: "
                f"got {len(response.data)} embeddings for {len(batch)} inputs"
            )

        batch_embs = [item.embedding for item in response.data]
        embeddings.extend(batch_embs)
        print(f"Processed batch {i//batch_size + 1}")

    # --- FINAL SAFETY CHECK ---
    if len(embeddings) != len(vocab):
        raise RuntimeError(
            f"Final embedding mismatch: {len(embeddings)} embeddings for {len(vocab)} vocab items"
        )

    embedding_matrix = np.array(embeddings, dtype=np.float32)

    # Save embeddings together with other textual-factor outputs
    out_dir = os.path.join(TEXT_ANALYTICS_DIR, "outputs_textual_factors")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "embedding_matrix.npy")
    np.save(out_path, embedding_matrix)
    print("Saved embedding_matrix.npy to:", out_path)

    return vocab, embedding_matrix

# ============================================================
# 5. WORD-CLUSTERING (NeighborFinder + EmbeddingCluster)
# ============================================================
# Purpose:
# - Group semantically similar words into clusters using embedding similarity.
# - These clusters define the word groups within which we later run SVD/LSA.
#
# Relevance:
# - Yes. This is the core step that turns word embeddings into interpretable semantic groups.
#
# LSH parameters (set from tune_lsh.py)
N_BITS = 256
N_TABLES = 32

# Neighbor search algorithm:
# - "lsh"   : fast approximate nearest neighbors (FAISS LSH)
# - "brute" : exact brute-force neighbors (slow but deterministic)
DEFAULT_NEIGHBOR_ALG = "lsh"

def cluster_words(
    embedding_matrix: np.ndarray,
    target_cluster_size: int = 50,
    neighbor_alg: str = DEFAULT_NEIGHBOR_ALG,
):
    """
    Cluster word embeddings into semantic groups.

    Steps:
    1) Build a NeighborFinder (creates brute-force index).
    2) If neighbor_alg == "lsh", create an LSH index with pre-tuned
       (N_BITS, N_TABLES) and attach it to the NeighborFinder.
    3) Build an EmbeddingCluster object.
    4) Run sequential clustering to group similar words.

    Inputs:
    - embedding_matrix : numpy array (V x D) from Word2Vec
    - target_cluster_size : target / soft cap for words per cluster (clusters may end up smaller)
    - neighbor_alg     : "lsh" (fast, uses FAISS LSH) or "brute" (exact)

    Outputs:
    - ec                : EmbeddingCluster object
    - clusters          : list of clusters (each cluster = list of word indices)
    - cluster_words_map : cluster → words mapping
    - word_cluster_map  : word index → cluster ID mapping
    """

    if neighbor_alg not in {"lsh", "brute"}:
        raise ValueError(f"neighbor_alg must be 'lsh' or 'brute', got: {neighbor_alg}")

    # 1) Build neighbor search engine (brute-force index always built inside)
    nf = NeighborFinder(
        embedding_matrix,
        random_state=42,
        num_queries=1000,   # used for their internal diagnostics if needed
    )

    # 2) If we use LSH, create the FAISS LSH index with tuned parameters
    if neighbor_alg == "lsh":
        print(f"Using FAISS LSH (bits={N_BITS}, tables={N_TABLES})")
        nf.lsh_index = nf.create_lsh_index(N_BITS, N_TABLES)
    else:
        print("Using brute-force neighbor search (exact).")

    # 3) Create clustering engine using chosen neighbor algorithm
    ec = EmbeddingCluster(nf, neighbor_alg=neighbor_alg)

    # 4) Perform clustering (Cong et al.'s sequential clustering)
    clusters = ec.sequentialcluster(cluster_size=target_cluster_size)

    # Map clusters <-> words
    cluster_words_map, word_cluster_map = ec.cluster_word_map(clusters)

    print(f"Number of clusters created: {len(clusters)}")

    return ec, clusters, cluster_words_map, word_cluster_map

# Output:
# - clusters          : list[list[int]] of word indices per cluster
# - word_cluster_map  : dict[int, int] mapping word_index -> cluster_id
# - cluster_words_map : dict[int, list[int]] mapping cluster_id -> word_indices


# ============================================================
# 6. BUILD TABLES FOR TEXTUAL FACTORS (document_word_data + word_cluster_data)
# ============================================================
# Purpose:
# - Convert filtered document word counts into the long format expected by TextualFactors.
# - Create a word->cluster lookup table to define cluster-specific SVD submatrices.

def build_document_word_data(df: pd.DataFrame, vocab: list[str]) -> pd.DataFrame:
    rows = []
    vocab_set = set(vocab)

    for doc_id, word_counts in zip(df["document"], df["word_freq"]):
        for word, count in word_counts.items():
            if word in vocab_set:
                rows.append({"document": doc_id, "ngram": word, "count": int(count)})

    document_word_data = pd.DataFrame(rows)
    print(f"document_word_data: {len(document_word_data)} rows, {document_word_data['document'].nunique()} documents")
    return document_word_data


def build_word_cluster_data(vocab: list[str], word_cluster_map: dict) -> pd.DataFrame:
    missing = [i for i in range(len(vocab)) if i not in word_cluster_map]
    if missing:
        raise ValueError(f"word_cluster_map missing {len(missing)} vocab indices (e.g. {missing[:10]})")

    word_cluster_data = pd.DataFrame({
        "ngram": vocab,
        "sequential_cluster": [word_cluster_map[i] for i in range(len(vocab))]
    })
    return word_cluster_data

# Output:
# - document_word_data : long table of document-word frequencies
# - word_cluster_data  : mapping of each word to its cluster


# Number of latent topics extracted per word cluster
N_TOPICS_PER_CLUSTER = 2

# ============================================================
# 7. CONSTRUCT TEXTUAL FACTORS (SVD / LSA)
# ============================================================

def compute_textual_factors(
    document_word_data: pd.DataFrame,
    word_cluster_data: pd.DataFrame,
    n_topics: int = 1
) -> dict:
    """
    Runs SVD/LSA inside each word cluster using TextualFactors.lsa_topics().

    Returns:
      - first_doc_topics_df
      - second_doc_topics_df (empty if n_topics < 2)
      - topics_words_df (TF1 word loadings)
      - topics_words2_df (TF2 word loadings; empty if n_topics < 2)
      - singular_values_df
      - topic_importances_df
    """
    tf_model = TextualFactors(
        document_word_data=document_word_data,
        word_cluster_data=word_cluster_data
    )

    (
        first_doc_topics,
        second_doc_topics,
        first_topics_words,
        second_topics_words,
        singular_values,
        topic_importances,
    ) = tf_model.lsa_topics(
        cluster_type="sequential_cluster",
        n_topics=n_topics
    )

    first_doc_topics_df = transfer_document_topics(first_doc_topics)

    if n_topics < 2:
        second_doc_topics_df = pd.DataFrame(columns=["cluster_id", "document", "topic_loading"])
        topics_words2_df = pd.DataFrame(columns=["topic", "word", "topic_loading"])
    else:
        second_doc_topics_df = transfer_document_topics(second_doc_topics)
        topics_words2_df = transfer_topic_words(second_topics_words)

    topics_words_df = transfer_topic_words(first_topics_words)

    singular_values_df = transfer_sigular_values(singular_values)
    topic_importances_df = transfer_topic_importances(topic_importances)

    return {
        "first_doc_topics_df": first_doc_topics_df,
        "second_doc_topics_df": second_doc_topics_df,
        "topics_words_df": topics_words_df,
        "topics_words2_df": topics_words2_df,
        "singular_values_df": singular_values_df,
        "topic_importances_df": topic_importances_df,
    }


# Output:
# A dictionary of DataFrames containing:
# - document-level factor loadings (for first and second topic)
# - word-level topic loadings
# - singular values from SVD
# - topic importance weights


# ============================================================
# QC HELPERS (extraction summary + optional TF diagnostics)
# ============================================================
# Relevance:
# - Yes. These helpers produce a lightweight QC table so we can verify that
#   page ranges, extraction, and token filtering behave as expected.


def _range_to_bounds(rng):
    """Convert a Python range (or None) to (start, end_exclusive, n_pages)."""
    if rng is None:
        return None, None, None
    try:
        start = int(rng.start)
        stop = int(rng.stop)
        n_pages = max(0, stop - start)
        return start, stop, n_pages
    except Exception:
        return None, None, None


def _parse_bank_year_from_filename(fname: str):
    """Parse (bank, year) from expected PDF filenames."""
    base = os.path.basename(fname)

    m = re.match(r"^(?P<bank>.+?)_(?P<year>\d{4})\.pdf$", base, flags=re.IGNORECASE)
    if m:
        return m.group("bank"), int(m.group("year"))

    m2 = re.match(r"^(?P<year>\d{4})_(?P<bank>.+?)_group\.pdf$", base, flags=re.IGNORECASE)
    if m2:
        return m2.group("bank"), int(m2.group("year"))

    return None, None


def build_extraction_summary(
    df_docs_before_filter: pd.DataFrame,
    df_docs_after_filter: pd.DataFrame,
    page_ranges_year: dict,
    year_label: str,
) -> pd.DataFrame:
    """Create a bank-year extraction QC table."""

    # Token counts: raw -> filtered
    tok_before_series = df_docs_before_filter.set_index("file").get("tokens_raw")
    if tok_before_series is None:
        tok_before = df_docs_before_filter.set_index("file")["tokens"].apply(lambda x: len(x) if isinstance(x, list) else 0).to_dict()
    else:
        tok_before = tok_before_series.apply(lambda x: len(x) if isinstance(x, list) else 0).to_dict()

    tok_after = (
        df_docs_after_filter.set_index("file")["tokens"].apply(lambda x: len(x) if isinstance(x, list) else 0).to_dict()
        if len(df_docs_after_filter)
        else {}
    )

    # Approximate paragraph count per file using joined content (newline-separated blocks)
    # This avoids needing the raw paragraph list at this stage.
    para_counts = {}
    if "content" in df_docs_before_filter.columns and "file" in df_docs_before_filter.columns:
        tmp = df_docs_before_filter.set_index("file")["content"].fillna("")
        para_counts = (tmp.str.count("\n") + 1).to_dict()

    rows = []
    for fname, rng in page_ranges_year.items():
        p_from, p_to, n_pages = _range_to_bounds(rng)
        bank, year_parsed = _parse_bank_year_from_filename(fname)

        n_tok_after = int(tok_after.get(fname, 0))
        status = "ok" if n_tok_after > 0 else "empty_or_missing"

        rows.append({
            "year": year_parsed,
            "run_label": year_label,
            "file": fname,
            "bank": bank,
            "pages_from": p_from,
            "pages_to": p_to,
            "n_pages": n_pages,
            "n_paragraphs": int(para_counts.get(fname, 0)),
            "n_tokens_before": int(tok_before.get(fname, 0)),
            "n_tokens_after": n_tok_after,
            "status": status,
        })

    summary = pd.DataFrame(rows)

    # Make it easier to scan: put ok first
    if not summary.empty and "status" in summary.columns:
        summary["status"] = pd.Categorical(summary["status"], categories=["ok", "empty_or_missing"], ordered=True)
        summary = summary.sort_values(["status", "n_tokens_after", "n_tokens_before"], ascending=[True, False, False]).reset_index(drop=True)

    return summary


def build_topic_loading_summary(
    first_doc_topics_df: pd.DataFrame,
    df_docs: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    OPTIONAL QC: summarize TF loadings at the file level.

    Note: This helper is not required for the core pipeline.
    It is useful for debugging / quick inspection.
    """
    if first_doc_topics_df is None or first_doc_topics_df.empty:
        return pd.DataFrame(columns=["file", "tf_l1", "tf_l2", "tf_max_abs", "tf_top_clusters", "tf_top_loadings"])

    wide = first_doc_topics_df.copy()
    if "document" not in wide.columns:
        raise ValueError(f"first_doc_topics_df missing 'document'. Available: {list(wide.columns)}")

    # Long format
    if ("topic_loading" in wide.columns) and (("cluster_id" in wide.columns) or ("topic" in wide.columns)):
        tmp = wide.copy()
        if "cluster_id" not in tmp.columns and "topic" in tmp.columns:
            tmp = tmp.rename(columns={"topic": "cluster_id"})

    else:
        # Wide format topic_loading_0, topic_loading_1, ...
        loading_cols = [c for c in wide.columns if c.startswith("topic_loading_")]
        if not loading_cols:
            raise ValueError(
                "first_doc_topics_df has no 'topic_loading' and no 'topic_loading_*' columns. "
                f"Available: {list(wide.columns)}"
            )

        tmp = wide.melt(
            id_vars=["document"],
            value_vars=loading_cols,
            var_name="cluster_id",
            value_name="topic_loading",
        )
        tmp["cluster_id"] = tmp["cluster_id"].str.replace("topic_loading_", "", regex=False).astype(int)

    tmp = tmp.merge(df_docs[["document", "file"]], on="document", how="left")
    tmp["abs_loading"] = tmp["topic_loading"].abs()

    def _topk_str(g, k):
        g2 = g.sort_values("abs_loading", ascending=False).head(k)
        clusters = ";".join(str(int(x)) for x in g2["cluster_id"].tolist())
        loads = ";".join(f"{float(x):.6f}" for x in g2["topic_loading"].tolist())
        return clusters, loads

    out_rows = []
    for f, g in tmp.groupby("file"):
        l1 = float(g["abs_loading"].sum())
        l2 = float(np.sqrt((g["topic_loading"] ** 2).sum()))
        mx = float(g["abs_loading"].max()) if len(g) else 0.0
        clusters, loads = _topk_str(g, top_n)

        out_rows.append({
            "file": f,
            "tf_l1": l1,
            "tf_l2": l2,
            "tf_max_abs": mx,
            "tf_top_clusters": clusters,
            "tf_top_loadings": loads,
        })

    return pd.DataFrame(out_rows)

def main():
    print("\n=== STEP 1: Load paragraphs from PDFs ===")
    # ------------------------------------------------------------
    # FULL-CORPUS RUN (all years)
    # ------------------------------------------------------------
    TARGET_YEAR = "ALL"
    page_ranges_year = page_ranges

    print("Running analysis on FULL corpus (all years)")
    print(f"Total files included: {len(page_ranges_year)}")

    report_paragraphs, report_sources = load_report_paragraphs(
        reports_folder,
        page_ranges_year,
        strict=True
    )
    print(f"Loaded {len(report_paragraphs)} paragraphs")
    if len(report_paragraphs) == 0:
        raise RuntimeError(
            f"Loaded 0 paragraphs. Check PDFs exist in: {reports_folder} and filenames match page_ranges keys."
        )

    # === NEW STEP 2A: Build paragraph-level DataFrame for Word2Vec ===
    df_paragraphs = pd.DataFrame({
        "content": report_paragraphs,
        "file": report_sources
    })

    # Clean + tokenize paragraphs for Word2Vec training
    df_paragraphs = preprocess_text_and_tokens(df_paragraphs, text_col="content", tokens_col="tokens")

    print("\n=== STEP 2: Build document-level DataFrame ===")
    df_docs = build_document_dataframe(report_paragraphs, report_sources)
    print(df_docs.head())

    print("\n=== STEP 3: Clean text + tokenize + count words ===")
    df_docs = preprocess_text_and_tokens(df_docs, text_col="content", tokens_col="tokens")

    # Keep a snapshot BEFORE dropping short documents, for QC reporting
    df_docs_before_doclen_filter = df_docs.copy()

    df_docs = df_docs[df_docs["tokens"].apply(len) >= 5].copy()
    # ------------------------------------------------------------
    # QC OUTPUT: Extraction summary table (ideal checkpoint output)
    # ------------------------------------------------------------
    out_folder = os.path.join(TEXT_ANALYTICS_DIR, "outputs_textual_factors")
    os.makedirs(out_folder, exist_ok=True)

    extraction_summary = build_extraction_summary(
        df_docs_before_filter=df_docs_before_doclen_filter,
        df_docs_after_filter=df_docs,
        page_ranges_year=page_ranges_year,
        year_label="ALL",
    )
    summary_path = os.path.join(out_folder, "extraction_summary_ALL.csv")
    extraction_summary.to_csv(summary_path, index=False)
    print(f"\nSaved extraction summary to: {summary_path}")
    print(extraction_summary.head(10))

    print(f"Documents kept after token+doc filtering: {len(df_docs)}")
    print("Example tokens after filtering:", df_docs["tokens"].iloc[0][:20])

    print("Example cleaned document:", df_docs["content"].iloc[0][:200])
    print("Example tokens:", df_docs["tokens"].iloc[0][:20])


    print("\n=== STEP 4: Create OpenAI Embeddings ===")
    vocab, embedding_matrix = train_openai_embeddings(df_paragraphs)
    print(f"Vocabulary size: {len(vocab)}")

    print("\n=== STEP 5: Cluster word embeddings (LSH sequential clustering; target_cluster_size=150) ===")
    ec, clusters, cluster_words_map, word_cluster_map = cluster_words(
        embedding_matrix,
        target_cluster_size=50,
        neighbor_alg="lsh"
    )
    print(f"Number of clusters: {len(clusters)}")

    print("\n=== STEP 6: Build document-word and word-cluster tables ===")
    document_word_data = build_document_word_data(df_docs, vocab)
    word_cluster_data  = build_word_cluster_data(vocab, word_cluster_map)

    print(document_word_data.head())
    print(word_cluster_data.head())

    print("\n=== STEP 7: Compute Textual Factors (SVD / LSA) ===")
    tf_results = compute_textual_factors(
        document_word_data,
        word_cluster_data,
        n_topics=N_TOPICS_PER_CLUSTER,
    )

    if N_TOPICS_PER_CLUSTER < 2:
        print("\nNote: N_TOPICS_PER_CLUSTER=1, so TF2 outputs are skipped.")

    # ------------------------------------------------------------
    # Merge ALL TF topic-loadings into extraction summary (wide)
    # ------------------------------------------------------------
    all_loadings = tf_results["first_doc_topics_df"].copy()

    # first_doc_topics_df is already wide: document + topic_loading_*
    # Attach file names
    all_loadings = all_loadings.merge(df_docs[["document", "file"]], on="document", how="left")

    # Keep 1 row per file with all topic_loading_* columns
    tf_cols = [c for c in all_loadings.columns if c.startswith("topic_loading_")]
    all_loadings = all_loadings[["file"] + tf_cols].drop_duplicates("file")

    # Merge into extraction summary
    extraction_summary_with_all_tfs = extraction_summary.merge(all_loadings, on="file", how="left")

    # Overwrite CSV
    extraction_summary_with_all_tfs.to_csv(summary_path, index=False)
    print(f"Updated extraction summary with ALL TF loadings: {summary_path}")

    print("\n=== Saving results ===")
    tf_results["first_doc_topics_df"].to_csv(os.path.join(out_folder, "first_doc_topics.csv"), index=False)

    # Only write TF2 if it exists (n_topics_per_cluster >= 2)
    if not tf_results["second_doc_topics_df"].empty:
        tf_results["second_doc_topics_df"].to_csv(os.path.join(out_folder, "second_doc_topics.csv"), index=False)

    tf_results["topics_words_df"].to_csv(os.path.join(out_folder, "topics_words.csv"), index=False)
    tf_results["singular_values_df"].to_csv(os.path.join(out_folder, "singular_values.csv"), index=False)
    tf_results["topic_importances_df"].to_csv(os.path.join(out_folder, "topic_importances.csv"), index=False)

    print("\nPipeline finished ✓")
    print("Outputs written to:", out_folder)

if __name__ == "__main__":
    main()

# Tuning