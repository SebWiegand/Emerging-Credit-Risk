import os
from itertools import chain  # currently unused, but kept so you recognize it

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from openai import OpenAI

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
    # 2024 files
    "2024_Danske_group.pdf": range(208, 240),
    "2024_UBS_group.pdf": range(88, 136),
    "2024_DeutscheBank_group.pdf": range(91, 208),
    "2024_ING_group.pdf": range(158, 222),

    # 2023 files
    "2023_Danske_group.pdf": range(175, 213),
    "2023_UBS_group.pdf": range(97, 153),
    "2023_DeutscheBank_group.pdf": range(91, 208),
    "2023_ING_group.pdf": range(131, 204),

    # 2022 files
    "2022_Danske_group.pdf": range(169, 208),
    "2022_UBS_group.pdf": range(83, 134),
    "2022_DeutscheBank_group.pdf": range(90, 213),
    "2022_ING_group.pdf": range(103, 185),

    # 2021 files
    "2021_Danske_group.pdf": range(159, 194),
    "2021_UBS_group.pdf": range(98, 150),
    "2021_DeutscheBank_group.pdf": range(84, 201),
    "2021_ING_group.pdf": range(45, 150),
}

# Default pages if a file is not in page_ranges
default_pages = range(0, 10)  # first 10 pages

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

            # Decide which pages to process
            if _file in page_ranges:
                pages_to_process = page_ranges[_file]
            else:
                pages_to_process = default_pages

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
    np.save("../Text analytics/embedding_matrix.npy", embedding_matrix)
    print("Saved embedding_matrix.npy in:", os.getcwd())
    return vocab, embedding_matrix



# ============================================================
# 5. CLUSTER WORD EMBEDDINGS (NeighborFinder + EmbeddingCluster)
#    using pre-tuned FAISS LSH parameters
# ============================================================

# ⚠ Set these based on your separate tuning script (e.g. tune_lsh.py)
N_BITS = 128    # number of hash bits = hyperplanes per table  (example)
N_TABLES = 64   # number of hash tables                       (example)

# Global hyperparameter: number of LSA topics per clustera
N_TOPICS_PER_CLUSTER = 2


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


def compute_textual_factors(document_word_data, word_cluster_data, n_topics=2):
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

    # 3. Convert numpy outputs to DataFrames for easy use/export
    first_doc_topics_df  = transfer_document_topics(first_doc_topics)
    second_doc_topics_df = transfer_document_topics(second_doc_topics)
    topics_words_df      = transfer_topic_words(first_topics_words)
    singular_values_df   = transfer_sigular_values(singular_values)
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

    print("\n=== STEP 2: Build document-level DataFrame ===")
    df_docs = build_document_dataframe(report_paragraphs, report_sources)
    print(df_docs.head())

    print("\n=== STEP 3: Clean text + tokenize + count words ===")
    df_docs = preprocess_and_count_words(df_docs)

    df_docs = df_docs[df_docs["tokens"].apply(len) >= 5].copy()

    print("Example cleaned document:", df_docs["content"].iloc[0][:200])
    print("Example tokens:", df_docs["tokens"].iloc[0][:20])


    print("\n=== STEP 4: Create OpenAI Embeddings ===")
    vocab, embedding_matrix = train_openai_embeddings(df_paragraphs)
    print(f"Vocabulary size: {len(vocab)}")

    print("\n=== STEP 5: Cluster word embeddings (LSH sequential clustering) ===")
    ec, clusters, cluster_words_map, word_cluster_map = cluster_words(
        embedding_matrix,
        cluster_size=50,
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

    # Create output folder
    out_folder = "outputs_textual_factors"
    os.makedirs(out_folder, exist_ok=True)

    print("\n=== Saving results ===")
    tf_results["first_doc_topics_df"].to_csv(f"{out_folder}/first_doc_topics.csv", index=False)
    tf_results["second_doc_topics_df"].to_csv(f"{out_folder}/second_doc_topics.csv", index=False)
    tf_results["topics_words_df"].to_csv(f"{out_folder}/topics_words.csv", index=False)
    tf_results["singular_values_df"].to_csv(f"{out_folder}/singular_values.csv", index=False)
    tf_results["topic_importances_df"].to_csv(f"{out_folder}/topic_importances.csv", index=False)

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