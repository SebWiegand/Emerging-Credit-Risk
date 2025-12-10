import os
from itertools import chain  # currently unused, but kept so you recognize it

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# --- helper functions from engine.py ---
from engine import (
    clean_and_normalize_text,
    calculate_word_frequencies,
)

# --- helper classes/functions from TextualFactors.py ---
from TextualFactors import (
    NeighborFinder,
    EmbeddingCluster,
    TextualFactors,
    transfer_document_topics,
    transfer_topic_words,
    transfer_sigular_values,
    transfer_topic_importances,
)

# ============================================================
# 0. SETTINGS: folders, page ranges, etc.
# ============================================================

# Project root = folder where this Main.py lives
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
# 1. LOAD TEXT FROM PDFS (your existing logic)
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


# ============================================================
# 2. BUILD DOCUMENT DATAFRAME
# ============================================================


def build_document_dataframe(report_paragraphs, report_sources, report_pages):
    """
    Build a DataFrame where each row is one paragraph (document).
    """
    df = pd.DataFrame(
        {
            "document": np.arange(len(report_paragraphs)),  # simple id
            "content": report_paragraphs,
            "file": report_sources,
            "page": report_pages,
        }
    )

    # make sure 'file' is actually string, then extract year/bank
    df["file"] = df["file"].astype(str)

    # Example: "2024_Danske_group.pdf" -> year="2024", bank="Danske"
    df["year"] = df["file"].str.extract(r"(^\d{4})", expand=False)
    df["bank"] = df["file"].str.extract(r"^\d{4}_(.*?)_", expand=False)

    return df


# ============================================================
# 3. CLEAN TEXT + WORD FREQUENCIES
# ============================================================


def preprocess_and_count_words(df):
    """
    Uses engine.clean_and_normalize_text and engine.calculate_word_frequencies.
    """
    # Clean & normalize the 'content' column
    df = clean_and_normalize_text(df, column_name="content")

    # calculate_word_frequencies expects a column name (default 'text'), so we pass 'content'
    df = calculate_word_frequencies(df, text_column="content")

    return df


# ============================================================
# 4. TRAIN WORD2VEC ON CLEANED TEXT
# ============================================================


def train_word2vec(df):
    """
    Train Word2Vec on your cleaned tokens from engine.calculate_word_frequencies.
    Uses df['tokens'] directly.
    """
    # df['tokens'] is a list of tokens per document (created by calculate_word_frequencies)
    tokenized_docs = df["tokens"].tolist()

    # Optional: quick sanity check
    print("Number of documents going into Word2Vec:", len(tokenized_docs))
    if tokenized_docs:
        print("Example doc tokens:", tokenized_docs[0][:20])

    if len(tokenized_docs) == 0:
        raise RuntimeError("No documents available for Word2Vec training.")

    w2v_model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=100,
        window=5,
        min_count=5,  # set to 1 so we don't end up with empty vocab
        workers=4,
        sg=1,  # skip-gram
    )

    word_vectors = w2v_model.wv
    vocab = list(word_vectors.key_to_index.keys())
    embedding_dim = word_vectors.vector_size

    embedding_matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    for i, w in enumerate(vocab):
        embedding_matrix[i, :] = word_vectors[w]

    print(f"Trained Word2Vec: {len(vocab)} words, dim={embedding_dim}")
    return w2v_model, vocab, embedding_matrix


# ============================================================
# 5. CLUSTER WORD EMBEDDINGS WITH NeighborFinder + EmbeddingCluster
# ============================================================


def cluster_words(embedding_matrix, cluster_size=50, neighbor_alg="lsh"):
    """
    Use NeighborFinder + EmbeddingCluster from TextualFactors.py
    to get word clusters.
    """
    nf = NeighborFinder(embedding_matrix, random_state=42, num_queries=1000)
    ec = EmbeddingCluster(nf, neighbor_alg=neighbor_alg)

    # You can switch to hierarchical_cluster if you want a fixed K
    clusters = ec.sequentialcluster(cluster_size=cluster_size)
    cluster_words_map, word_cluster_map = ec.cluster_word_map(clusters)

    print(f"Number of clusters: {len(clusters)}")
    return ec, clusters, cluster_words_map, word_cluster_map


# ============================================================
# 6. BUILD document_word_data & word_cluster_data FOR TextualFactors
# ============================================================


def build_document_word_data(df, vocab):
    """
    Build a long-format document-word DataFrame with columns:
    'document', 'ngram', 'count', as TextualFactors expects.
    We use df['word_freq'] which is a Counter for each document.
    """
    rows = []
    for doc_id, word_counts in zip(df["document"], df["word_freq"]):
        for word, count in word_counts.items():
            if word in vocab:  # keep only words that are in embedding vocab
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
    Build DataFrame with columns:
    'ngram' and 'sequential_cluster' (cluster id).
    """
    cluster_ids = [word_cluster_map[i] for i in range(len(vocab))]
    word_cluster_df = pd.DataFrame(
        {
            "ngram": vocab,
            "sequential_cluster": cluster_ids,
        }
    )
    return word_cluster_df


# ============================================================
# 7. CONSTRUCT TEXTUAL FACTORS (LSA topics via SVD)
# ============================================================


def compute_textual_factors(document_word_data, word_cluster_data):
    tf_model = TextualFactors(
        document_word_data=document_word_data, word_cluster_data=word_cluster_data
    )

    # LSA topics (SVD-based textual factors)
    (
        first_doc_topics,
        second_doc_topics,
        first_topics_words,
        second_topics_words,
        singular_values,
        topic_importances,
    ) = tf_model.lsa_topics(cluster_type="sequential_cluster", n_topics=2)

    # Convert to DataFrames
    first_doc_topics_df = transfer_document_topics(first_doc_topics)
    second_doc_topics_df = transfer_document_topics(second_doc_topics)
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


# ============================================================
# MAIN PIPELINE
# ============================================================


def main():
    # --- 1. Load raw paragraphs from PDFs ---
    report_paragraphs, report_sources, report_pages = load_report_paragraphs(
        reports_folder, page_ranges, default_pages
    )

    if not report_paragraphs:
        print("⚠️ No paragraphs loaded. Check that 'reports_folder' and 'page_ranges' match your PDF files.")
        return

    # --- 2. Build document-level DataFrame ---
    df_docs = build_document_dataframe(report_paragraphs, report_sources, report_pages)
    print("Paragraph-level docs:", df_docs.shape)

    # --- 3. Clean text + word frequencies ---
    df_docs = preprocess_and_count_words(df_docs)
    print("After preprocessing:", df_docs[["document", "content"]].head())

    # --- 4. Train Word2Vec on cleaned content ---
    w2v_model, vocab, embedding_matrix = train_word2vec(df_docs)

    # --- 5. Cluster word embeddings ---
    ec, clusters, cluster_words_map, word_cluster_map = cluster_words(
        embedding_matrix, cluster_size=50, neighbor_alg="lsh"
    )

    # --- 6. Build document_word_data + word_cluster_data ---
    document_word_data = build_document_word_data(df_docs, vocab)
    word_cluster_data = build_word_cluster_data(vocab, word_cluster_map)

    # --- 7. Compute textual factors (LSA/SVD) ---
    tf_results = compute_textual_factors(document_word_data, word_cluster_data)

    # Example: save key outputs to CSV so you can inspect / use in regressions
    out_folder = "outputs_textual_factors"
    os.makedirs(out_folder, exist_ok=True)

    tf_results["first_doc_topics_df"].to_csv(
        os.path.join(out_folder, "first_doc_topics.csv"), index=False
    )
    tf_results["second_doc_topics_df"].to_csv(
        os.path.join(out_folder, "second_doc_topics.csv"), index=False
    )
    tf_results["topics_words_df"].to_csv(
        os.path.join(out_folder, "topics_words.csv"), index=False
    )
    tf_results["singular_values_df"].to_csv(
        os.path.join(out_folder, "singular_values.csv"), index=False
    )
    tf_results["topic_importances_df"].to_csv(
        os.path.join(out_folder, "topic_importances.csv"), index=False
    )

    print("Pipeline finished. Textual factor files written to:", out_folder)


if __name__ == "__main__":
    main()
