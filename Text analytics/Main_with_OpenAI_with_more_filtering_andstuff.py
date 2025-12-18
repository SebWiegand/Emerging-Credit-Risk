import os
from itertools import chain  # currently unused, but kept so you recognize it

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from collections import Counter
from openai import OpenAI

import ast
import re

# Read API key from environment variable (set in PyCharm Run Configuration)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise RuntimeError(
        "Missing OPENAI_API_KEY environment variable. "
        "Set it in Run → Edit Configurations → Environment variables."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

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

# Project root = folder where this Main_V1.py lives
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Folder with your annual reports (the Reports folder in your project)
reports_folder = os.path.join(PROJECT_ROOT, "Reports")
# If you prefer, you can hard-code:
# reports_folder = "/Users/sebastianwiegandmoller/PycharmProjects/Emerging-Credit-Risk/Reports"

# Your own page_ranges (copied from your notebook)
page_ranges = {

    # =========================
    # 2021
    # =========================
    "2021_Barclays_group.pdf": range(199, 288),
    "2021_BNPparibas_group.pdf": range(170, 192),
    "2021_Danske_group.pdf": range(159, 194),
    "2021_DeutscheBank_group.pdf": range(84, 201),
    "2021_ING_group.pdf": range(45, 150),
    "2021_OPPohjola_group.pdf": range(1, 22),
    "2021_SEB_group.pdf": range(89, 96),
    "2021_UBS_group.pdf": range(98, 150),

    # =========================
    # 2022
    # =========================
    "2022_Barclays_group.pdf": range(263, 369),
    "2022_BNPparibas_group.pdf": range(170, 192),
    "2022_Danske_group.pdf": range(169, 208),  # note double .pdf as in filename
    "2022_DeutscheBank_group.pdf": range(90, 213),
    "2022_ING_group.pdf": range(103, 185),
    "2022_OPPohjola_group.pdf": range(1, 22),
    "2022_SEB_group.pdf": range(82, 91),
    "2022_UBS_group.pdf": range(83, 134),

    # =========================
    # 2023
    # =========================
    "2023_Barclays_group.pdf": range(253, 362),
    "2023_BNPparibas_group.pdf": range(166, 190),
    "2023_Danske_group.pdf": range(175, 213),
    "2023_DeutscheBank_group.pdf": range(91, 208),
    "2023_ING_group.pdf": range(131, 204),
    "2023_OPPohjola_group.pdf": range(1, 34),
    "2023_SEB_group.pdf": range(50, 59),
    "2023_UBS_group.pdf": range(97, 153),

    # =========================
    # 2024
    # =========================
    "2024_Barclays_group.pdf": range(262, 382),
    "2024_BNPparibas_group.pdf": range(160, 184),
    "2024_Danske_group.pdf": range(208, 240),
    "2024_DeutscheBank_group.pdf": range(91, 208),
    "2024_ING_group.pdf": range(158, 222),
    "2024_OPPohjola_group.pdf": range(40, 81),
    "2024_SEB_group.pdf": range(50, 59),
    "2024_UBS_group.pdf": range(88, 136),
}

# IMPORTANT: No silent fallback. Every PDF must have an explicit page range.
# If a PDF filename is not in `page_ranges`, the pipeline will stop with an error.
default_pages = None

# ============================================================
# 0B. TOKEN / VOCAB FILTERING (IMPORTANT FOR INTERPRETABILITY)
# ============================================================

TOKEN_MIN_LEN = 3
MIN_DF = 2            # token must appear in at least MIN_DF documents
MAX_DF_RATIO = 0.70   # drop tokens that appear in more than this share of documents

EXTRA_DROP_WORDS = {
    # Generic report boilerplate
    "annual", "report", "reports", "group", "plc", "page", "pages", "section", "chapter",
    "table", "tables", "figure", "figures", "statement", "statements",
    "introduction", "overview", "note", "notes",

    # Bank names / identifiers (extend as needed)
    "barclays", "seb", "ubs", "ing", "danske", "deutschebank", "deutsche", "bank",
    "bnp", "paribas", "fortis", "oppohjola", "op", "pohjola",

    # Common legal entities
    "limited", "ltd", "ab", "asa", "as",
}

def _basic_token_filter(tokens):
    """Remove obvious junk tokens before df-based filtering."""
    out = []
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

def filter_tokens_with_df_rules(df, tokens_col="tokens", min_df=MIN_DF, max_df_ratio=MAX_DF_RATIO):
    """
    1) basic token filtering per document
    2) document-frequency filtering across corpus (min_df/max_df)

    Returns COPY of df with filtered tokens.
    """
    df = df.copy()

    # 1) per-doc filtering
    df[tokens_col] = df[tokens_col].apply(_basic_token_filter)

    # doc frequency
    doc_n = len(df)
    df_counter = Counter()
    for toks in df[tokens_col]:
        df_counter.update(set(toks))

    max_df = int(max_df_ratio * doc_n)
    allowed = {tok for tok, dfi in df_counter.items() if (dfi >= min_df) and (dfi <= max_df)}

    # 2) apply df rule
    df[tokens_col] = df[tokens_col].apply(lambda toks: [t for t in toks if t in allowed])
    return df

# ============================================================
# 1. LOAD TEXT FROM PDF´s
# ============================================================


def load_report_paragraphs(reports_folder, page_ranges, default_pages):
    report_paragraphs = []
    report_paragraphs_source = []
    report_pages_source = []

    print(f"Looking for PDFs in: {reports_folder}")

    for path, dirs, files in os.walk(reports_folder):
        pdfs = [file for file in files if file.endswith(".pdf")]
        if not pdfs:
            continue
        print("Found PDFs:", pdfs)

        for _file in pdfs:
            print(f"Processing {_file}...")
            full_path = os.path.join(path, _file)

            # Decide which pages to process (STRICT: require explicit page ranges)
            if _file not in page_ranges:
                raise ValueError(
                    f"File '{_file}' not found in page_ranges. "
                    "Add an explicit page range for this PDF (no defaults)."
                )
            pages_to_process = page_ranges[_file]

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
                        report_pages_source.extend([page_num] * len(blocks))
                        report_paragraphs_source.extend([_file] * len(blocks))

    return report_paragraphs, report_paragraphs_source, report_pages_source


# Output:
# After this section we have three parallel lists:
# 1) report_paragraphs        -> all extracted text paragraphs (strings)
# 2) report_paragraphs_source -> which PDF each paragraph came from
# 3) report_pages_source      -> which page number each paragraph came from
# All three lists have the same length; each index represents one paragraph.

# ============================================================
# 2. BUILD DOCUMENT DATAFRAME
# ============================================================

def build_document_dataframe(report_paragraphs, report_sources):
    """
    Combine all paragraphs belonging to the same file into one document.
    Each PDF becomes ONE document (bank × year).

    Also parses bank and year from the filename so outputs
    can be merged with bank-year panel data.
    """

    df = pd.DataFrame({
        "file": report_sources,
        "content": report_paragraphs,
    })

    # Group paragraphs into one document per file
    df_grouped = df.groupby("file", as_index=False)["content"].apply(
        lambda texts: "\n".join(texts)
    )

    # Parse year and bank from filenames like:
    # "2021_Danske_group.pdf" or "2021_Danske_group.pdf.pdf"
    pattern = r"(?P<year>\d{4})_(?P<bank>.+?)_group\.pdf(?:\.pdf)?"
    extracted = df_grouped["file"].str.extract(pattern)

    df_grouped["year"] = extracted["year"].astype("Int64")
    df_grouped["bank"] = extracted["bank"]

    # Sort for reproducible document IDs
    df_grouped = df_grouped.sort_values(
        ["year", "bank", "file"]
    ).reset_index(drop=True)

    # Stable internal document ID
    df_grouped["document"] = np.arange(len(df_grouped))

    return df_grouped

# Output:
# df with columns:
# - document : integer ID (0, 1, 2, ...)
# - content  : paragraph text
# - file     : source PDF filename
# (optional)
# - year     : year parsed from filename
# - bank     : bank parsed from filename


# ============================================================
# 3. CLEAN TEXT + WORD FREQUENCIES (engine.py)
# ============================================================

def preprocess_and_count_words(df):
    """
    Prepare the text for embedding using engine.py functions.
    Steps:
    1) Clean and normalize the 'content' column.
    2) Tokenize and count word frequencies per document.
    """

    # 1) Clean / normalize the text in 'content'
    df = clean_and_normalize_text(df, column_name="content")

    # 2) Tokenize + count word frequencies.
    #    calculate_word_frequencies expects a text column (default 'text'),
    #    so we tell it to use 'content'.
    df = calculate_word_frequencies(df, text_column="content")

    return df

# Output:
# df now has extra columns:
# - content   : cleaned & normalized text
# - tokens    : list of tokens (words) per document
# - word_freq : Counter/dict with word -> count for each document

# Note: We only use a subset of functions from engine.py.
# The unused utilities (daily aggregation, long-format by date) are meant for true time-series text data,
# but our documents are grouped by bank-year, not by calendar dates, so these functions are not needed here.



# ============================================================
# 4. OPENAI EMBEDDING FUNCTION
# ============================================================

def train_openai_embeddings(df, model_name="text-embedding-3-small"):
    """
    Build word embeddings using OpenAI's embedding API.
    Trains on paragraph-level tokens.
    """
    vocab = sorted(set(chain.from_iterable(df["tokens"].tolist())))
    print(f"Vocabulary size: {len(vocab)} words")

    batch_size = 500
    embeddings = []

    for i in range(0, len(vocab), batch_size):
        batch = vocab[i:i+batch_size]
        response = client.embeddings.create(
            model=model_name,
            input=batch
        )
        batch_embs = [item.embedding for item in response.data]
        embeddings.extend(batch_embs)
        print(f"Processed batch {i//batch_size + 1}")

    embedding_matrix = np.array(embeddings, dtype=np.float32)
    np.save("embedding_matrix.npy", embedding_matrix)
    print("Saved embedding_matrix.npy in:", os.getcwd())
    return vocab, embedding_matrix



# ============================================================
# 5. CLUSTER WORD EMBEDDINGS (NeighborFinder + EmbeddingCluster)
#    using pre-tuned FAISS LSH parameters
# ============================================================

# ⚠ Set these based on your separate tuning script (e.g. tune_lsh.py)
N_BITS = 256    # number of hash bits = hyperplanes per table  (example)
N_TABLES = 32   # number of hash tables                       (example)

# Global hyperparameter: number of LSA topics per clustera
N_TOPICS_PER_CLUSTER = 1


def cluster_words(
    embedding_matrix,
    cluster_size=100,
    neighbor_alg="lsh",
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
    - cluster_size     : approx. number of words per cluster
    - neighbor_alg     : "lsh" (fast, uses FAISS LSH) or "brutal" (exact)

    Outputs:
    - ec                : EmbeddingCluster object
    - clusters          : list of clusters (each cluster = list of word indices)
    - cluster_words_map : cluster → words mapping
    - word_cluster_map  : word index → cluster ID mapping
    """

    # 1) Build neighbor search engine (brute-force index always built inside)
    nf = NeighborFinder(
        embedding_matrix,
        random_state=42,
        num_queries=1000,   # used for their internal diagnostics if needed
    )

    # 2) If we use LSH, create the FAISS LSH index with tuned parameters
    if neighbor_alg == "lsh":
        print(
            f"Using FAISS LSH with tuned parameters: "
            f"bits={N_BITS}, tables={N_TABLES}"
        )
        nf.lsh_index = nf.create_lsh_index(N_BITS, N_TABLES)
    else:
        print("Using brute-force neighbor search (no LSH).")

    # 3) Create clustering engine using chosen neighbor algorithm
    ec = EmbeddingCluster(nf, neighbor_alg=neighbor_alg)

    # 4) Perform clustering (Cong et al.'s sequential clustering)
    clusters = ec.sequentialcluster(cluster_size=cluster_size)

    # Map clusters <-> words
    cluster_words_map, word_cluster_map = ec.cluster_word_map(clusters)

    print(f"Number of clusters created: {len(clusters)}")

    return ec, clusters, cluster_words_map, word_cluster_map


# Output:
# - clusters : semantic word clusters
# - word_cluster_map : tells you which cluster each word belongs to
# - cluster_words_map : tells you which words are in each cluster
#
# Note:
# We rely on Cong et al.'s NeighborFinder and EmbeddingCluster:
# - NeighborFinder.__init__() to build LSH / brute-force indices
# - NeighborFinder.create_lsh_index() for FAISS LSH construction
# - EmbeddingCluster.sequentialcluster() for semantic clustering
# - EmbeddingCluster.cluster_word_map() to map words to clusters
#
# LSH hyperparameters (N_BITS, N_TABLES) are chosen offline in a
# separate tuning script using their eval_index_accuracy diagnostics.




# ============================================================
# 6. BUILD DOCUMENT-WORD AND WORD-CLUSTER DATA FOR TEXTUAL FACTORS
# ============================================================

def build_document_word_data(df, vocab):
    """
    Create a long-format table with:
    - document (document ID)
    - ngram (word)
    - count (frequency of the word in that document)

    This is the format expected by TextualFactors.
    """

    rows = []
    vocab_set = set(vocab)

    # df["word_freq"] is a dict: word → count for each document
    for doc_id, word_counts in zip(df["document"], df["word_freq"]):
        for word, count in word_counts.items():
            if word in vocab_set:  # keep only words that exist in the embedding model
                rows.append(
                    {
                        "document": doc_id,
                        "ngram": word,
                        "count": int(count),
                    }
                )

    doc_word_df = pd.DataFrame(rows)

    print(
        f"document_word_data: {doc_word_df.shape[0]} rows, "
        f"{doc_word_df['document'].nunique()} documents"
    )

    return doc_word_df



def build_word_cluster_data(vocab, word_cluster_map):
    """
    Create a mapping:
    - ngram (word)
    - sequential_cluster (cluster ID)

    word_cluster_map: index → cluster_id
    vocab: list of words aligned with embedding_matrix
    """

    cluster_ids = [word_cluster_map[i] for i in range(len(vocab))]

    word_cluster_df = pd.DataFrame(
        {
            "ngram": vocab,
            "sequential_cluster": cluster_ids
        }
    )

    return word_cluster_df

# Output:
# - document_word_data : long table of document-word frequencies
# - word_cluster_data  : mapping of each word to its cluster


# ============================================================
# 7. CONSTRUCT TEXTUAL FACTORS (SVD / LSA)
# ============================================================


def compute_textual_factors(document_word_data, word_cluster_data, n_topics=1):
    """
    Compute textual factors using the TextualFactors class.
    This performs SVD (LSA) inside each word cluster.

    Inputs:
    - document_word_data : long-format table (document, ngram, count)
    - word_cluster_data  : mapping of each word to its cluster
    - n_topics           : number of latent topics to extract per cluster (typically 1–2)

    Outputs:
    Returns a dictionary of DataFrames:
    - first_doc_topics_df    : document-level factor loadings (topic 1)
    - second_doc_topics_df   : document-level factor loadings (topic 2)
    - topics_words_df        : word-level loadings for each topic
    - singular_values_df     : SVD singular values
    - topic_importances_df   : importance weights for each topic
    """

    # 1. Initialize the model with your two required data tables
    tf_model = TextualFactors(
        document_word_data=document_word_data,
        word_cluster_data=word_cluster_data
    )

    # 2. Compute the latent topics using SVD (LSA)
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

    # TF1
    first_doc_topics_df = transfer_document_topics(first_doc_topics)

    # TF2: kun hvis vi faktisk har bedt om 2 topics
    if n_topics < 2:
        second_doc_topics_df = pd.DataFrame(columns=["cluster_id", "document", "topic_loading"])
    else:
        second_doc_topics_df = transfer_document_topics(second_doc_topics)

    # Word-level topic loadings (TF1)
    topics_words_df = transfer_topic_words(first_topics_words)

    singular_values_df = transfer_sigular_values(singular_values)
    topic_importances_df = transfer_topic_importances(topic_importances)

    return {
        "first_doc_topics_df": first_doc_topics_df,
        "second_doc_topics_df": second_doc_topics_df,
        "topics_words_df": topics_words_df,
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
# 7B. EXPORT TOP-20 SUMMARIES (TOP CLUSTERS, TOP WORDS, TOP LOADINGS)
# ============================================================

def _pick_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_first_numeric_col(df, exclude_cols=None, prefer_substr=None):
    exclude_cols = set(exclude_cols or [])
    num_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return None
    if prefer_substr:
        for c in num_cols:
            if prefer_substr in c.lower():
                return c
    return num_cols[0]


def _load_first_doc_topics_as_long(path_csv):
    """
    Supports BOTH formats:
    (A) Long:  columns like [cluster_id, document, topic_loading]
    (B) Wide:  columns like [document, topic_loading_0, topic_loading_1, ...]
    Returns long df with columns: cluster_id, document, topic_loading
    """
    df = pd.read_csv(path_csv)

    # Long format already?
    if "cluster_id" in df.columns and "document" in df.columns and "topic_loading" in df.columns:
        return df[["cluster_id", "document", "topic_loading"]].copy()

    # Wide format: melt topic_loading_{cluster}
    if "document" not in df.columns:
        raise RuntimeError(f"first_doc_topics.csv has no 'document' column. Found columns: {list(df.columns)}")

    topic_cols = [c for c in df.columns if c.startswith("topic_loading_")]
    if not topic_cols:
        raise RuntimeError(
            "Could not find topic loading columns in first_doc_topics.csv. "
            f"Found columns: {list(df.columns)}"
        )

    long_df = df.melt(
        id_vars=["document"],
        value_vars=topic_cols,
        var_name="cluster_id",
        value_name="topic_loading",
    )
    long_df["cluster_id"] = long_df["cluster_id"].str.replace("topic_loading_", "", regex=False)
    long_df["cluster_id"] = pd.to_numeric(long_df["cluster_id"], errors="coerce")
    long_df = long_df.dropna(subset=["cluster_id"]).copy()
    long_df["cluster_id"] = long_df["cluster_id"].astype(int)

    return long_df[["cluster_id", "document", "topic_loading"]]


def _topics_words_to_long(tw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize topics_words output to a long DataFrame with columns:
      - cluster_id
      - word
      - word_loading

    Supports two formats:
      (A) already-long with cluster/word/loading columns
      (B) compact format with columns: ['topic', 'topic_distribution'] where
          topic_distribution is a stringified list of (word, loading) pairs.
    """

    # Case B: compact format from transfer_topic_words
    if "topic" in tw.columns and "topic_distribution" in tw.columns and len(tw.columns) == 2:
        rows = []
        for _, r in tw.iterrows():
            topic = r["topic"]

            # Extract numeric cluster id from topic (handles int, '12', 'cluster_12', etc.)
            cluster_id = None
            if pd.isna(topic):
                continue
            if isinstance(topic, (int, np.integer)):
                cluster_id = int(topic)
            else:
                s = str(topic)
                m = re.search(r"\d+", s)
                if m:
                    cluster_id = int(m.group())
            if cluster_id is None:
                continue

            dist = r["topic_distribution"]
            if pd.isna(dist):
                continue

            # Parse distribution into iterable of (word, loading)
            parsed = None
            if isinstance(dist, str):
                try:
                    parsed = ast.literal_eval(dist)
                except Exception:
                    parsed = None
            else:
                parsed = dist

            if parsed is None:
                continue

            if isinstance(parsed, dict):
                items = parsed.items()
            else:
                items = parsed

            for item in items:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                word = item[0]
                try:
                    loading = float(item[1])
                except Exception:
                    continue
                rows.append({
                    "cluster_id": cluster_id,
                    "word": str(word),
                    "word_loading": loading,
                })

        return pd.DataFrame(rows, columns=["cluster_id", "word", "word_loading"])

    # Case A: already-long format (try to detect common column names)
    tw_cluster_col = _pick_first_existing_col(tw, ["cluster_id", "sequential_cluster", "cluster"])
    word_col = _pick_first_existing_col(tw, ["ngram", "word", "token", "term"])
    loading_col = _pick_first_numeric_col(tw, exclude_cols=[c for c in [tw_cluster_col] if c], prefer_substr="loading")
    if loading_col is None:
        loading_col = _pick_first_numeric_col(tw, exclude_cols=[c for c in [tw_cluster_col] if c], prefer_substr="weight")

    if tw_cluster_col and word_col and loading_col:
        out = tw[[tw_cluster_col, word_col, loading_col]].copy()
        out = out.rename(columns={tw_cluster_col: "cluster_id", word_col: "word", loading_col: "word_loading"})
        out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce")
        out = out.dropna(subset=["cluster_id"]).copy()
        out["cluster_id"] = out["cluster_id"].astype(int)
        out["word_loading"] = pd.to_numeric(out["word_loading"], errors="coerce")
        out = out.dropna(subset=["word_loading"]).copy()
        return out[["cluster_id", "word", "word_loading"]]

    raise RuntimeError(
        "Could not normalize topics_words to long format. "
        f"Found columns: {list(tw.columns)}"
    )


def export_top20_summaries(out_folder="outputs_textual_factors", top_n=20, top_words_per_cluster=20):
    """
    Writes four CSVs into out_folder:
      1) top20_topic_importances.csv
      2) top20_first_doc_topics.csv          (top |loading| rows among top clusters)
      3) top20_words_per_top_cluster.csv     (top words per cluster by |word_loading|)
      4) top20_topic_words_overall.csv       (top words overall across top clusters)

    Robust to wide vs long first_doc_topics format.
    """
    imp_path = os.path.join(out_folder, "topic_importances.csv")
    words_path = os.path.join(out_folder, "topics_words.csv")
    doc_path = os.path.join(out_folder, "first_doc_topics.csv")

    if not (os.path.exists(imp_path) and os.path.exists(words_path) and os.path.exists(doc_path)):
        print("[export_top20_summaries] Skipping: required CSVs not found in", out_folder)
        return

    # ----------------------
    # 1) Topic importances
    # ----------------------
    imp = pd.read_csv(imp_path)

    cluster_col = _pick_first_existing_col(imp, ["cluster_id", "sequential_cluster", "cluster"])
    if cluster_col is None:
        cluster_col = imp.columns[0]  # fallback

    importance_col = _pick_first_numeric_col(imp, exclude_cols=[cluster_col], prefer_substr="importance")
    if importance_col is None:
        raise RuntimeError(
            f"Could not find a numeric importance column in {imp_path}. Found columns: {list(imp.columns)}"
        )

    imp_small = imp[[cluster_col, importance_col]].copy()
    imp_small = imp_small.rename(columns={cluster_col: "cluster_id", importance_col: "topic_importance"})
    imp_small["cluster_id"] = pd.to_numeric(imp_small["cluster_id"], errors="coerce")
    imp_small = imp_small.dropna(subset=["cluster_id"]).copy()
    imp_small["cluster_id"] = imp_small["cluster_id"].astype(int)

    top_imp = imp_small.sort_values("topic_importance", ascending=False).head(top_n)
    top_imp.to_csv(os.path.join(out_folder, "top20_topic_importances.csv"), index=False)
    top_clusters = top_imp["cluster_id"].tolist()

    # ----------------------
    # 2) First doc topics (TF1 loadings)
    # ----------------------
    doc_long = _load_first_doc_topics_as_long(doc_path)
    doc_long = doc_long[doc_long["cluster_id"].isin(top_clusters)].copy()

    doc_long["abs_loading"] = doc_long["topic_loading"].abs()
    top_doc = doc_long.sort_values("abs_loading", ascending=False).head(top_n).drop(columns=["abs_loading"])
    top_doc.to_csv(os.path.join(out_folder, "top20_first_doc_topics.csv"), index=False)

    # ----------------------
    # 3) Topic words (TF1 word loadings)
    # ----------------------
    tw_raw = pd.read_csv(words_path)
    tw_small = _topics_words_to_long(tw_raw)

    tw_top = tw_small[tw_small["cluster_id"].isin(top_clusters)].copy()
    tw_top["abs_word_loading"] = tw_top["word_loading"].abs()

    # (a) Top words per top cluster
    rows = []
    for cid in top_clusters:
        sub = tw_top[tw_top["cluster_id"] == cid].sort_values("abs_word_loading", ascending=False)
        rows.append(sub.head(top_words_per_cluster))
    top_words_per_cluster_df = (
        pd.concat(rows, ignore_index=True).drop(columns=["abs_word_loading"])
        if rows else
        pd.DataFrame(columns=["cluster_id", "word", "word_loading"])
    )
    top_words_per_cluster_df.to_csv(os.path.join(out_folder, "top20_words_per_top_cluster.csv"), index=False)

    # (b) Top words overall across top clusters
    top_words_overall = tw_top.sort_values("abs_word_loading", ascending=False).head(top_n).drop(columns=["abs_word_loading"])
    top_words_overall.to_csv(os.path.join(out_folder, "top20_topic_words_overall.csv"), index=False)

    print(f"[export_top20_summaries] Wrote top-20 summaries to: {out_folder} (clusters={len(top_clusters)})")

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("\n=== STEP 1: Load paragraphs from PDFs ===")
    report_paragraphs, report_sources, report_pages = load_report_paragraphs(
        reports_folder,
        page_ranges,
        default_pages
    )
    print(f"Loaded {len(report_paragraphs)} paragraphs")

    # === NEW STEP 2A: Build paragraph-level DataFrame for Word2Vec ===
    df_paragraphs = pd.DataFrame({
        "content": report_paragraphs,
        "file": report_sources
    })

    # Clean + tokenize paragraphs for Word2Vec training
    df_paragraphs = preprocess_and_count_words(df_paragraphs)
    df_paragraphs = filter_tokens_with_df_rules(df_paragraphs, tokens_col="tokens")

    print("\n=== STEP 2: Build document-level DataFrame ===")
    df_docs = build_document_dataframe(report_paragraphs, report_sources)
    print(df_docs.head())

    print("\n=== STEP 3: Clean text + tokenize + count words ===")
    df_docs = preprocess_and_count_words(df_docs)
    df_docs = filter_tokens_with_df_rules(df_docs, tokens_col="tokens")

    df_docs = df_docs[df_docs["tokens"].apply(len) >= 5].copy()
    print(f"Documents kept after token+doc filtering: {len(df_docs)}")
    print("Example tokens after filtering:", df_docs["tokens"].iloc[0][:20])

    print("Example cleaned document:", df_docs["content"].iloc[0][:200])
    print("Example tokens:", df_docs["tokens"].iloc[0][:20])


    print("\n=== STEP 4: Create OpenAI Embeddings ===")
    vocab, embedding_matrix = train_openai_embeddings(df_paragraphs)
    print(f"Vocabulary size: {len(vocab)}")

    print("\n=== STEP 5: Cluster word embeddings (LSH sequential clustering) ===")
    ec, clusters, cluster_words_map, word_cluster_map = cluster_words(
        embedding_matrix,
        cluster_size=100,
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

    if N_TOPICS_PER_CLUSTER == 1:
        print("\nNote: N_TOPICS_PER_CLUSTER=1, so TF2 outputs are skipped.")

    # Create output folder
    out_folder = "outputs_textual_factors"
    os.makedirs(out_folder, exist_ok=True)

    print("\n=== Saving results ===")
    tf_results["first_doc_topics_df"].to_csv(f"{out_folder}/first_doc_topics.csv", index=False)

    # Only write TF2 if it exists (n_topics_per_cluster >= 2)
    if not tf_results["second_doc_topics_df"].empty:
        tf_results["second_doc_topics_df"].to_csv(f"{out_folder}/second_doc_topics.csv", index=False)

    tf_results["topics_words_df"].to_csv(f"{out_folder}/topics_words.csv", index=False)
    tf_results["singular_values_df"].to_csv(f"{out_folder}/singular_values.csv", index=False)
    tf_results["topic_importances_df"].to_csv(f"{out_folder}/topic_importances.csv", index=False)
    export_top20_summaries(out_folder=out_folder)

    print("\nPipeline finished ✓")
    print("Outputs written to:", out_folder)


if __name__ == "__main__":
    main()

# Improvements:
# - Cluster size:
#     * Currently cluster_size = 50, which yields ~600 small clusters (≈5–6 words each).
#     * Possible improvement: run a small sensitivity analysis on cluster_size
#       (e.g. 30, 50, 80) and see how robust the main TFs are.
#
# - Ingestion of documents:
#     * Filenames must follow the pattern "YYYY_<Bank>_group.pdf" (or .pdf.pdf as now),
#       because year/bank are parsed from the name.
#     * page_ranges keys must match the exact filenames in Reports/.
#       If the naming convention changes, update both the regex in
#       build_document_dataframe() and the page_ranges dict.
#
# - Number of topics per cluster:
#     * Currently N_TOPICS_PER_CLUSTER = 2 (TF1 + TF2 as a robustness check).
#     * For the final empirical analysis, consider using only TF1
#       (set N_TOPICS_PER_CLUSTER = 1) and treat TF2 as diagnostics.
#
# - Filtering clusters and topics:
#     * Not all clusters / topics will be relevant economically.
#     * After running the pipeline, use topic_importances.csv and singular_values.csv
#       to filter out weak or noisy components:
#           - Drop clusters where overall topic_importance is very low.
#           - Optionally drop TF2 if its singular value is much smaller than TF1
#             or if the associated words are not interpretable as a clear risk theme.
#     * This filtering is done downstream (in a separate analysis script / notebook),
#       but it is an explicit modelling choice and should be documented in the thesis.


# Manual modelling settings / hyperparameters:
# - Sample design:
#     * Which banks and years are included (which PDFs in Reports/)
#     * Which pages per PDF (page_ranges, default_pages)
#     * Filename convention for parsing identifiers:
#         - Expected: "YYYY_<Bank>_group.pdf" (or ".pdf.pdf" variants)
#         - Parsed fields: year, bank (used for bank-year panel merges)
#
# - Text preprocessing (engine.py):
#     * Stopword list (English)
#     * Lemmatization (WordNetLemmatizer)
#     * Removal of digits and punctuation
#     * Currently unigrams only (no bigrams yet)
#
# - Token quality filter (main pipeline):
#     * Report-level documents are filtered after cleaning
#     * Current rule: keep only documents where len(tokens) >= 5
#     * Motivation: some PDF content (tables/headers) can be cleaned to empty/near-empty text
#
# - Embeddings (OpenAI):
#     * model_name = "text-embedding-3-small"
#     * Vocabulary built from cleaned tokens across all paragraphs (df_paragraphs)
#     * Embeddings retrieved via OpenAI API (no local training hyperparameters)
#     * Batch size for embedding calls: batch_size = 500
#
# - Neighbor search / LSH:
#     * neighbor_alg = "lsh" (vs "brutal")
#     * N_BITS = 128, N_TABLES = 64  (FAISS LSH hyperparameters; should match tune_lsh.py output)
#     * random_state = 42, num_queries = 1000 (NeighborFinder diagnostics)
#
# - Clustering:
#     * Sequential clustering (Cong et al.)
#     * cluster_size = 50 (target words per cluster; affects number of clusters)
#
# - Textual factors (SVD / LSA):
#     * cluster_type = "sequential_cluster"
#     * N_TOPICS_PER_CLUSTER = 2 (TF1 and TF2; TF2 mainly as robustness check)


# Note on unused functions:
# Several functions in engine.py and TextualFactors.py are not used in this project.

# - Daily/text-based utilities (aggregate_daily_word_frequencies, convert_to_long_format)
#   are for timestamped text data, not annual report paragraphs.
#   We work with bank-year documents, not daily sequences, so these transformations are irrelevant.

# - LSH evaluation/tuning (eval_index_accuracy, optimize_lsh_hyperparameters)
#   is unnecessary for the Cong et al. pipeline.
#   Default FAISS LSH is sufficient for clustering, and the pipeline does not require accuracy optimization.

# - Alternative clustering methods (heuristic_cluster, hierarchical_cluster)
#   are not part of the standard sequential LSH clustering used here.
#   Cong et al. use a simple sequential LSH-based clustering, so alternative algorithms add no value.

# - Internal TextualFactors helpers (normalization/diagnostics) are not required.
#   Only the SVD-based lsa_topics() method is needed to construct textual factors from cl