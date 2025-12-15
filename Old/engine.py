# ============================================================
# engine.py  –  text cleaning + word frequencies (no NLTK)
# ============================================================

import re
from collections import Counter

import pandas as pd

# ------------------------------------------------------------
# Simple English stopword list (no downloads needed)
# ------------------------------------------------------------
BASIC_STOPWORDS = {
    "the", "and", "of", "to", "in", "a", "for", "on", "at", "by", "with",
    "is", "are", "was", "were", "be", "been", "am", "it", "this", "that",
    "as", "an", "or", "from", "we", "our", "their", "they", "you", "your",
    "he", "she", "his", "her", "its", "but", "not", "into", "about", "than",
    "then", "which", "who", "what", "when", "where", "why", "how", "also",
    "such", "may", "can", "could", "would", "should", "do", "does", "did",
    "has", "have", "had", "will", "shall", "been", "being"
}


# ============================================================
# clean_and_normalize_text
# ============================================================

def clean_and_normalize_text(df: pd.DataFrame, column_name: str = "content") -> pd.DataFrame:
    """
    Clean and normalize text in df[column_name].

    Steps (kept conceptually similar to your NLTK version):
      1. Remove digits and punctuation.
      2. Lowercase the text.
      3. Tokenize by simple whitespace split.
      4. Remove basic English stopwords.
      5. Join tokens back into a cleaned string.

    No NLTK -> no downloads / SSL issues.
    """

    def process_document(doc):
        # Make sure we always work with a string
        if not isinstance(doc, str):
            doc = str(doc)

        # Remove digits and punctuation, keep letters + whitespace
        doc = re.sub(r"[\d]|[^\w\s]", " ", doc)

        # Lowercase and split into tokens
        tokens = doc.lower().split()

        # Remove stopwords
        tokens = [w for w in tokens if w not in BASIC_STOPWORDS]

        # Join back into a single cleaned string
        return " ".join(tokens)

    df[column_name] = df[column_name].apply(process_document)
    return df


# ============================================================
# calculate_word_frequencies
# ============================================================

def calculate_word_frequencies(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    From a column with cleaned text, build:

      - df['tokens']    : list of tokens per document
      - df['word_freq'] : Counter dict(word -> count) per document

    This replaces the old NLTK-based version, but keeps the same
    function name and output structure so the rest of your code
    (Word2Vec + TextualFactors) still works.
    """

    def tokenize(text):
        # Ensure string
        if not isinstance(text, str):
            text = str(text)

        # Simple whitespace tokenization
        tokens = text.lower().split()

        # Remove stopwords again (double-safety)
        tokens = [w for w in tokens if w not in BASIC_STOPWORDS]

        return tokens

    # Apply tokenization
    df["tokens"] = df[text_column].apply(tokenize)

    # Count word frequencies per document
    df["word_freq"] = df["tokens"].apply(Counter)

    return df


# ============================================================
# aggregate_daily_word_frequencies
# (unchanged, just cleaned up a bit)
# ============================================================

def aggregate_daily_word_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by 'date' and sum the word_freq dictionaries for each day.
    Expects:
        - 'date' column
        - 'word_freq' column containing dicts (word -> count)
    """
    grouped = df.groupby("date")

    def aggregate_dicts(series):
        total_count = Counter()
        for dictionary in series:
            total_count.update(dictionary)
        return dict(total_count)

    daily_word_freq = grouped["word_freq"].agg(aggregate_dicts).reset_index()
    return daily_word_freq


# ============================================================
# convert_to_long_format
# (unchanged in spirit)
# ============================================================

def convert_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a DataFrame from wide format (date, word_freq dictionary) to
    long format (date, word_id, word_freq).

    Parameters
    ----------
    df : DataFrame
        Must have columns:
            - 'date'
            - 'word_freq' (dict with words as keys and counts as values)

    Returns
    -------
    DataFrame
        Columns: 'date', 'word_id', 'word_freq'
    """
    long_df = (
        df.set_index("date")["word_freq"]
          .apply(pd.Series)      # dict -> columns
          .stack()               # wide -> long
          .reset_index()
    )
    long_df.columns = ["date", "word_id", "word_freq"]

    return long_df
