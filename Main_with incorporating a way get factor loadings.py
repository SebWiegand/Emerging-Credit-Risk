import os
from itertools import chain  # currently unused, but kept so you recognize it

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

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
#
# ============================================================
# 0. SETTINGS: folders, page ranges, etc.
# ============================================================

# Project root = folder where this Main_with incorporating a way get factor loadings.py lives
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
                    pages_to_process = range(total_pages)

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

# Output
# Output:
# After this section we have three parallel lists:
# 1) report_paragraphs        -> all extracted text paragraphs (strings)
# 2) report_paragraphs_source -> which PDF each paragraph came from
# 3) report_pages_source      -> which page number each paragraph came from
# All three lists have the same length; each index represents one paragraph.

# ============================================================
# 2. BUILD DOCUMENT DATAFRAME
# ============================================================

import numpy as np
import pandas as pd

def build_document_dataframes(report_paragraphs, report_sources, report_pages):
    """
    Build two DataFrames:

    1) df_paragraphs: one row per paragraph (good for Word2Vec + LSH)
       - paragraph_id, content, file, page

    2) df_reports: one row per annual report (good for TF loadings / regressions)
       - document (0..num_reports-1), content (full report text),
         file, year, bank
    """

    # -----------------------------
    # A. Paragraph-level dataframe
    # -----------------------------
    df_paragraphs = pd.DataFrame({
        "paragraph_id": np.arange(len(report_paragraphs)),
        "content": report_paragraphs,
        "file": report_sources,
        "page": report_pages,
    })

    df_paragraphs["file"] = df_paragraphs["file"].astype(str)

    # -----------------------------
    # B. Annual-report-level dataframe
    # -----------------------------
    combined = (
        df_paragraphs
        .groupby("file")["content"]
        .apply(lambda x: "\n".join(x))
        .reset_index()
    )

    combined["document"] = combined.index

    combined["year"] = combined["file"].str.extract(r"(^\d{4})", expand=False)
    combined["bank"] = combined["file"].str.extract(r"^\d{4}_(.*?)_", expand=False)

    df_reports = combined[["document", "content", "file", "year", "bank"]]

    return df_paragraphs, df_reports


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
# 4. TRAIN WORD2VEC ON CLEANED TOKENS
# ============================================================

def train_word2vec(df):
    """
    Train a Word2Vec model on the tokenized documents.

    Input:
    - df : DataFrame with a 'tokens' column
           (each row is a list of words for one document)

    Output:
    - w2v_model       : the trained gensim Word2Vec model
    - vocab           : list of words in the vocabulary
    - embedding_matrix: numpy array of shape (V, D),
                        where V = vocab size, D = embedding dimension
    """

    # df['tokens'] is a list of tokens per document
    tokenized_docs = df["tokens"].tolist()

    print("Number of documents going into Word2Vec:", len(tokenized_docs))
    if tokenized_docs:
        print("Example doc tokens:", tokenized_docs[0][:20])

    if len(tokenized_docs) == 0:
        raise RuntimeError("No documents available for Word2Vec training.")

    # ------------------------------------------------------------
    # Train the Word2Vec model
    # ------------------------------------------------------------
    w2v_model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=100,   # embedding dimension
        window=5,          # context window size
        min_count=5,       # ignore rare words
        workers=4,         # CPU cores
        sg=1,              # skip-gram model
    )

    # Extract vocabulary and vectors
    word_vectors = w2v_model.wv
    vocab = list(word_vectors.key_to_index.keys())
    embedding_dim = word_vectors.vector_size

    # ------------------------------------------------------------
    # Build embedding matrix aligned to vocabulary
    # ------------------------------------------------------------
    embedding_matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    for i, w in enumerate(vocab):
        embedding_matrix[i, :] = word_vectors[w]

    print(f"Trained Word2Vec: {len(vocab)} words, dim={embedding_dim}")

    # ------------------------------------------------------------
    # SAVE embedding_matrix so tune_lsh.py can load it
    # ------------------------------------------------------------
    np.save("embedding_matrix.npy", embedding_matrix)
    print("Saved embedding_matrix.npy in:", os.getcwd())

    return w2v_model, vocab, embedding_matrix


# ============================================================
# 5. CLUSTER WORD EMBEDDINGS (NeighborFinder + EmbeddingCluster)
#    using pre-tuned FAISS LSH parameters
# ============================================================

# ⚠ Set these based on your separate tuning script (e.g. tune_lsh.py)
N_BITS = 128    # number of hash bits = hyperplanes per table  (example)
N_TABLES = 16   # number of hash tables                       (example)


def cluster_words(
    embedding_matrix,
    cluster_size=50,
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
    - document (paragraph ID)
    - ngram (word)
    - count (frequency of the word in that document)

    This is the format expected by TextualFactors.
    """

    rows = []

    # df["word_freq"] is a dict: word → count for each paragraph/document
    for doc_id, word_counts in zip(df["document"], df["word_freq"]):
        for word, count in word_counts.items():
            if word in vocab:  # keep only words that exist in the embedding model
                rows.append(
                    {
                        "document": doc_id,
                        "ngram": word,
                        "count": int(count)
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

    # ---------------------------------------------------------
    # STEP 2: Build two DataFrames
    # df_paragraphs → paragraph-level (for Word2Vec + LSH)
    # df_docs       → report-level  (for final TFs)
    # ---------------------------------------------------------
    df_paragraphs, df_docs = build_document_dataframes(
        report_paragraphs, report_sources, report_pages
    )

    print("=== STEP 2: Build document-level DataFrames ===")
    print("Paragraph-level docs (for Word2Vec):", len(df_paragraphs))
    print("Report-level docs (for TF regressions):", len(df_docs))
    print(df_docs.head())

    # ---------------------------------------------------------
    # STEP 3: Clean + tokenize paragraphs
    # (NOT report-level!)
    # ---------------------------------------------------------
    print("\n=== STEP 3: Clean text + tokenize paragraphs ===")
    df_paragraphs = preprocess_and_count_words(df_paragraphs)

    print("Example cleaned paragraph:", df_paragraphs['content'].iloc[0][:200])
    print("Example tokens:", df_paragraphs['tokens'].iloc[0][:20])

    # ---------------------------------------------------------
    # STEP 4: Train Word2Vec on PARAGRAPHS
    # ---------------------------------------------------------
    print("\n=== STEP 4: Train Word2Vec on paragraph-level tokens ===")
    w2v_model, vocab, embedding_matrix = train_word2vec(df_paragraphs)
    print(f"Vocabulary size: {len(vocab)}")

    # ---------------------------------------------------------
    # STEP 5: Cluster word embeddings
    # ---------------------------------------------------------
    print("\n=== STEP 5: Cluster word embeddings (LSH sequential clustering) ===")
    ec, clusters, cluster_words_map, word_cluster_map = cluster_words(
        embedding_matrix,
        cluster_size=50,
        neighbor_alg="lsh"
    )
    print(f"Number of clusters: {len(clusters)}")

    # ---------------------------------------------------------
    # STEP 6: Build document-word & cluster-word tables
    # Use PARAGRAPH-LEVEL doc-word matrix, then aggregate to reports
    # ---------------------------------------------------------
    print("\n=== STEP 6: Build document-word and word-cluster tables ===")

    df_paragraphs = df_paragraphs.rename(columns={"paragraph_id": "document"})
    document_word_data = build_document_word_data(df_paragraphs, vocab)
    word_cluster_data  = build_word_cluster_data(vocab, word_cluster_map)

    print(document_word_data.head())
    print(word_cluster_data.head())

    # ---------------------------------------------------------
    # STEP 6b: Aggregate paragraph-level document-word to REPORT level
    # (1 document = 1 annual report)
    # ---------------------------------------------------------
    print("\n=== Aggregating paragraph-level counts to report-level ===")

    # Map each paragraph to its report
    paragraph_to_report = df_paragraphs[["document", "file"]]

    merged = document_word_data.merge(paragraph_to_report, on="document", how="left")

    report_word_counts = (
        merged.groupby(["file", "ngram"])["count"]
        .sum()
        .reset_index()
    )

    # assign document ids matching df_docs
    report_word_counts = report_word_counts.merge(
        df_docs[["file", "document"]],
        on="file",
        how="left"
    )

    report_word_counts = report_word_counts.rename(columns={"document": "report_document"})

    # rebuild document_word_data FOR REPORTS
    document_word_data_reports = report_word_counts[["report_document", "ngram", "count"]]
    document_word_data_reports = document_word_data_reports.rename(
        columns={"report_document": "document"}
    )

    # ---------------------------------------------------------
    # STEP 7: Compute Textual Factors (on report-level doc-word matrix)
    # ---------------------------------------------------------
    print("\n=== STEP 7: Compute Textual Factors (SVD / LSA) ===")
    tf_results = compute_textual_factors(document_word_data_reports, word_cluster_data)

    # ---------------------------------------------------------
    # SAVE OUTPUT
    # ---------------------------------------------------------
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
#   Only the SVD-based lsa_topics() method is needed to construct textual factors from clustered words.

