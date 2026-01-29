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
from urllib.parse import unquote

# Read API key from environment variable (set in PyCharm Run Configuration)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise RuntimeError(
        "Missing OPENAI_API_KEY environment variable. "
        "Set it in Run → Edit Configurations → Environment variables."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------
# Directions
# ------------------------------------------------------------
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

def norm_for_match(name: str) -> str:
    """Normalize filenames for robust matching between PDFs on disk and page_ranges keys."""
    name = unquote(name)
    name = os.path.basename(name).lower()
    # normalize apostrophes and whitespace
    name = name.replace("’", "'")
    name = name.replace("'", "")
    name = re.sub(r"\s+", " ", name).strip()
    return name

# Page ranges by year (dedented to top-level, not inside norm_for_match)
page_ranges_2015 = {
    "barclays_2015.pdf": range(116, 216),  # was LSE_BARC_2015_...
    "Deutsche_2015.pdf": range(79, 188),  # was NYSE_DB_2015.pdf
    "IDG_2015.pdf": range(50, 56),  # was NYSE_IDG_2015.pdf
    "danskebank_2015.pdf": range(134, 159),  # was danske-bank_2015.pdf
    "KBC_2015.pdf": range(82, 118),  # was KBC Group NV 2015.pdf
    "commerzbank_2015.pdf": range(106, 146),  # unchanged (but now consistent)
    "amrobank_2015.pdf": range(124, 248),  # was abn-amro-bank_2015.pdf
    "seb_2015.pdf": range(154, 170),  # was seb-group_2015.pdf
    "CreditAgricole_2015.pdf": range(44, 80),  # was Cr%c3%a9dit_Agricole_...(Mar-30-2015).pdf
    "Santander_2015.pdf": range(158, 275),  # was NYSE_STD_2015.pdf (very likely Santander)
    "BBVA_2015.pdf": range(35, 64),  # was NYSE_BBVA_2015_...
    "Erste_2015.pdf": range(196, 241),  # was Erste Group Bank A 2015.pdf
    "nordea_2015.pdf": range(33, 46),  # unchanged
    "DNB_2015.pdf": range(29, 55),  # was DNB's Annual Report 2015.pdf (corrected risk section)
    "ubs_2015.pdf": range(164, 232),  # 3569.ar.en.2015.pdf
    "swedbank_2015.pdf": range(144, 224),  # Annual Report2015.pdf
    "unicredit_2015.pdf": range(314, 462),  # 13370.ar.en.2015.pdf
    "Raiffeisen_2015.pdf": range(144, 224),
}

page_ranges_2016 = {
    "barclays_2016.pdf": range(133, 228),  # was LSE_BARC_2016.pdfj
    "Deutsche_2016.pdf": range(88, 199),  # was NYSE_DB_2016fff.pdf
    "IDG_2016.pdf": range(54, 62),  # was NYSE_IDG_2016.pdf
    "danskebank_2016.pdf": range(135, 160),  # was danske-bank_2016.pdf
    "KBC_2016.pdf": range(85, 122),  # was KBC Group NV 2016.pdf
    "commerzbank_2016.pdf": range(90, 124),  # unchanged (but now consistent)
    "amrobank_2016.pdf": range(95, 189),  # was abn-amro-bank_2016.pdf
    "seb_2016.pdf": range(158, 174),  # was seb-group_2016.pdf
    "CreditAgricole_2016.pdf": range(43, 83),  # was Cr%c3%a9dit_Agricole_...(Apr-01-2016).pdf
    "Santander_2016.pdf": range(174, 290),  # was NYSE_STD_2016.pdf (likely Santander)
    "handelsbanken_2016.pdf": range(24, 29),  # unchanged
    "nordea_2016.pdf": range(43, 59),  # unchanged
    "DNB_2016.pdf": range(29, 55),  # was DNB's Annual Report 2016.pdf (corrected risk section)
    "BBVA_2016.pdf": range(35, 73),  # was NYSE_BBVA_2016_...
    "Erste_2016.pdf": range(186, 231),  # was Erste Group Bank A2016.pdf
    "ubs_2016.pdf": range(118, 166),               # 3569.ar.en.2016.pdf
    "swedbank_2016.pdf": range(90, 122),           # 2016 Annual Report.pdf
    "unicredit_2016.pdf": range(282, 438),         # 13370.ar.en.2016.pdf
    "Raiffeisen_2016.pdf": range(90, 122),
}
page_ranges_2017 = {
    "barclays_2017.pdf": range(116, 196),  # was LSE_BARC_2017.pdf
    "Deutsche_2017.pdf": range(41, 137),  # was NYSE_DB_2017.pdf
    "IDG_2017.pdf": range(42, 51),  # was NYSE_IDG_2017.pdf
    "danskebank_2017.pdf": range(140, 167),  # was danske-bank_2017.pdf
    "KBC_2017.pdf": range(88, 126),  # was KBC Group NV 2017.pdf
    "commerzbank_2017.pdf": range(98, 134),  # unchanged (but now consistent)
    "amrobank_2017.pdf": range(47, 129),  # was abn-amro-bank_2017.pdf
    "seb_2017.pdf": range(160, 176),  # was seb-group_2017.pdf
    "handelsbanken_2017.pdf": range(24, 29),  # unchanged
    "nordea_2017.pdf": range(43, 59),  # unchanged
    "DNB_2017.pdf": range(29, 55),  # was DNB Bank annual report 2017.pdf
    "BBVA_2017.pdf": range(37, 78),  # was NYSE_BBVA_2017_...
    "Erste_2017.pdf": range(185, 227),  # was Erste Group Bank A 2017.pdf
    "ubs_2017.pdf": range(114, 165),  # 3569.ar.en.2017.pdf
    "swedbank_2017.pdf": range(146, 248),  # 2017 Annual Report .pdf
    "unicredit_2017.pdf": range(178, 262),  # 13370.ar.en.2017.pdf
    "Raiffeisen_2017.pdf": range(146, 248),
}
page_ranges_2018 = {
    "barclays_2018.pdf": range(126, 214),  # LSE_BARC_2018.pdf
    "Deutsche_2018.pdf": range(44, 154),  # NYSE_DB_2018.pdf
    "IDG_2018.pdf": range(61, 75),  # NYSE_IDG_2018.pdf
    "danskebank_2018.pdf": range(171, 203),  # danske-bank_2018.pdf
    "KBC_2018.pdf": range(90, 132),  # KBC Group NV 2018.pdf
    "commerzbank_2018.pdf": range(106, 142),  # commerzbank_2018.pdf
    "amrobank_2018.pdf": range(37, 116),  # abn-amro-bank_2018.pdf
    "seb_2018.pdf": range(162, 179),  # seb-group_2018.pdf
    "CreditAgricole_2018.pdf": range(50, 162),  # Crédit Agricole Form Annual Report (Apr-05-2018)
    "handelsbanken_2018.pdf": range(25, 30),  # handelsbanken_2018.pdf
    "nordea_2018.pdf": range(67, 76),  # nordea_2018.pdf
    "DNB_2018.pdf": range(139, 162),  # DNB's Annual Report 2018.pdf (corrected risk section)
    "BBVA_2018.pdf": range(70, 136),  # NYSE_BBVA_2018_...
    "Erste_2018.pdf": range(200, 245),  # Erste Group Bank A2018.pdf
    "ubs_2018.pdf": range(120, 171),  # 3569.ar.en.2018.pdf
    "unicredit_2018.pdf": range(279, 418),  # 13370.ar.en.2018.pdf
    "swedbank_2018.pdf": range(180, 268),  # 2018 Annual Report .pdf
    "Raiffeisen_2018.pdf": range(180, 268),
}
# 2019 (renamed to new bank_YYYY.pdf convention)
page_ranges_2019 = {
    "barclays_2019.pdf": range(124, 203),  # LSE_BARC_2019.pdf
    "Deutsche_2019.pdf": range(49, 162),  # NYSE_DB_2019.pdf
    "IDG_2019.pdf": range(162, 252),  # NYSE_IDG_2019.pdf
    "danskebank_2019.pdf": range(176, 209),  # danske-bank_2019.pdf
    "KBC_2019.pdf": range(92, 136),  # KBC Group NV 2019.pdf
    "commerzbank_2019.pdf": range(96, 134),  # commerzbank_2019.pdf
    "amrobank_2019.pdf": range(41, 129),  # abn-amro-bank_2019.pdf
    "seb_2019.pdf": range(163, 181),  # seb-group_2019.pdf
    "CreditAgricole_2019.pdf": range(50, 108),  # Crédit Agricole Form Annual Report (Mar-26-2019)
    "handelsbanken_2019.pdf": range(25, 30),  # handelsbanken_2019.pdf
    "nordea_2019.pdf": range(73, 110),  # nordea_2019.pdf
    "DNB_2019.pdf": range(30, 53),  # DNB's Annual Report 2019.pdf (confirmed)
    "BBVA_2019.pdf": range(123, 142),  # NYSE_BBVA_2019_...
    "Erste_2019.pdf": range(201, 251),  # Erste Group Bank A 2019.pdf
    "ubs_2019.pdf": range(106, 154),  # 3569.ar.en.2019.pdf
    "swedbank_2019.pdf": range(178, 211),  # 2019 Annual Report .pdf
    "unicredit_2019.pdf": range(250, 381),  # 13370.ar.en.2019.pdf
    "Raiffeisen_2019.pdf": range(178, 211),
}

# 2020 (renamed to new bank_YYYY.pdf convention)
page_ranges_2020 = {
    "barclays_2020.pdf": range(142, 232),  # LSE_BARC_2020.pdf
    "Deutsche_2020.pdf": range(76, 178),  # NYSE_DB_2020.pdf
    "IDG_2020.pdf": range(82, 184),  # NYSE_IDG_2020.pdf
    "danskebank_2020.pdf": range(176, 209),  # danske-bank_2020.pdf
    "KBC_2020.pdf": range(93, 138),  # KBC Group NV 2020.pdf
    "commerzbank_2020.pdf": range(120, 162),  # commerzbank_2020.pdf
    "amrobank_2020.pdf": range(61, 146),  # abn-amro-bank_2020.pdf
    "seb_2020.pdf": range(165, 178),  # seb-group_2020.pdf
    "handelsbanken_2020.pdf": range(25, 31),  # handelsbanken_2020.pdf
    "nordea_2020.pdf": range(105, 135),  # nordea_2020.pdf (as in original)
    "DNB_2020.pdf": range(32, 66),  # DNB's Annual Report 2020.pdf (confirmed)
    "BBVA_2020.pdf": range(178, 199),  # NYSE_BBVA_2020_...
    "Erste_2020.pdf": range(205, 256),  # Erste Group Bank A 2020.pdf
    "ubs_2020.pdf": range(91, 141),  # 3569.ar.en.2020.pdf
    "swedbank_2020.pdf": range(196, 228),  # 2020 Annual Report .pdf
    "unicredit_2020.pdf": range(263, 400),  # 13370.ar.en.2020.pdf
    "Raiffeisen_2020.pdf": range(196, 228),
}
page_ranges_2021 = {
    "barclays_2021.pdf": range(25, 60),  # 2021_Barclays_group.pdf
    "danskebank_2021.pdf": range(159, 194),  # 2021_Danske_group.pdf
    "Deutsche_2021.pdf": range(84, 201),  # 2021_DeutscheBank_group.pdf
    "seb_2021.pdf": range(140, 162),  # 2021_SEB_group.pdf
    "ubs_2021.pdf": range(98, 150),  # 2021_UBS_group.pdf
    "KBC_2021.pdf": range(94, 139),  # KBC Group NV 2021.pdf
    "commerzbank_2021.pdf": range(100, 144),  # commerzbank_2021.pdf
    "amrobank_2021.pdf": range(92, 180),  # abn-amro-bank_2021.pdf
    "handelsbanken_2021.pdf": range(26, 32),  # handelsbanken_2021.pdf
    "nordea_2021.pdf": range(160, 163),  # nordea_2021.pdf
    "DNB_2021.pdf": range(149, 174),  # DNB's Annual Report 2021.pdf (confirmed)
    "BBVA_2021.pdf": range(174, 192),  # NYSE_BBVA_2021.pdf
    "IDG_2021.pdf": range(45, 151),  # NYSE_IDG_2021 (1).pdf
    "Raiffeisen_2021.pdf": range(174, 202),
}
# 2022 (renamed to new bank_YYYY.pdf convention)
page_ranges_2022 = {
    "barclays_2022.pdf": range(263, 369),  # 2022_Barclays_group.pdf
    "danskebank_2022.pdf": range(169, 208),  # 2022_Danske_group.pdf
    "Deutsche_2022.pdf": range(90, 213),  # 2022_DeutscheBank_group.pdf
    "seb_2022.pdf": range(145, 168),  # 2022_SEB_group.pdf
    "ubs_2022.pdf": range(83, 134),  # 2022_UBS_group.pdf
    "KBC_2022.pdf": range(94, 140),  # KBC Group NV 2022.pdf
    "commerzbank_2022.pdf": range(105, 150),  # commerzbank_2022.pdf
    "amrobank_2022.pdf": range(64, 155),  # abn-amro-bank_2022.pdf
    "handelsbanken_2022.pdf": range(26, 32),  # handelsbanken_2022.pdf
    "nordea_2022.pdf": range(227, 229),  # nordea_2022.pdf
    "DNB_2022.pdf": range(149, 174),  # DNB's Annual Report 2022.pdf (confirmed)
    "CreditAgricole_2022.pdf": range(39, 101),  # Crédit Agricole Form Annual Report (Apr-04-2022)
    "BBVA_2022.pdf": range(183, 205),  # NYSE_BBVA_2022_...
    "Erste_2022.pdf": range(215, 263),  # Erste Group Bank 2022.pdf
    "IDG_2022.pdf": range(103, 188),  # NYSE_IDG_2022 (1).pdf
    "Raiffeisen_2022.pdf": range(193, 224),
}
# 2023 (renamed to new bank_YYYY.pdf convention)
page_ranges_2023 = {
    "barclays_2023.pdf": range(253, 362),  # 2023_Barclays_group.pdf
    "danskebank_2023.pdf": range(175, 213),  # 2023_Danske_group.pdf
    "Deutsche_2023.pdf": range(91, 208),  # 2023_DeutscheBank_group.pdf
    "seb_2023.pdf": range(148, 167),  # 2023_SEB_group.pdf
    "ubs_2023.pdf": range(97, 153),  # 2023_UBS_group.pdf
    "IDG_2023.pdf": range(131, 207),  # NYSE_IDG_2023 (1).pdf
    "KBC_2023.pdf": range(96, 145),  # KBC Group NV 2023.pdf
    "commerzbank_2023.pdf": range(215, 263),  # commerzbank_2023.pdf
    "amrobank_2023.pdf": range(54, 161),  # abn-amro-bank_2023.pdf
    "handelsbanken_2023.pdf": range(22, 27),  # handelsbanken_2023.pdf
    "nordea_2023.pdf": range(209, 251),  # nordea_2023.pdf
    "DNB_2023.pdf": range(150, 174),  # DNB's Annual Report 2023.pdf (confirmed)
    "BBVA_2023.pdf": range(284, 307),  # NYSE_BBVA_2023.pdf
    "Erste_2023.pdf": range(335, 391),  # Erste Group Bank 2023.pdf
    "ubs_2023.pdf": range(98, 156),  # 3569.ar.en.2023.pdf (overlaps UBS_2023)
    "Raiffeisen_2023.pdf": range(188, 220),
}
# 2024 (renamed to new bank_YYYY.pdf convention)
page_ranges_2024 = {
    "barclays_2024.pdf": range(262, 382),  # 2024_Barclays_group.pdf
    "danskebank_2024.pdf": range(208, 240),  # 2024_Danske_group.pdf
    "Deutsche_2024.pdf": range(91, 208),  # 2024_DeutscheBank_group.pdf
    "amrobank_2024.pdf": range(49, 161),  # abn amro bank n.v. annual report 2024.pdf
    "Santander_2024.pdf": range(501, 561),  # Banco Santander S.A. annual report 2024.pdf
    "commerzbank_2024.pdf": range(333, 382),  # commerzbank ag annual report 2024.pdf
    "KBC_2024.pdf": range(62, 97),  # KBC Group NV annual report 2024.pdf
    "nordea_2024.pdf": range(280, 375),  # nordea_2024.pdf
    "swedbank_2024.pdf": range(86, 120),  # swedbank-ab_2024.pdf
    "DNB_2024.pdf": range(152, 178),  # 2024_DNB_group.pdf / DNB Bank ASA annual report 2024.pdf (confirmed)
    "Erste_2024.pdf": range(285, 350),  # Erste Group Bank AG annual report 2024.pdf
    "BBVA_2024.pdf": range(418, 442),  # BBVA2024.pdf
    "handelsbanken_2024.pdf": range(80, 130),  # handelsbanken_2024.pdf
    "ubs_2024.pdf": range(89, 134),  # 3569.ar.en.2024.pdf
    "Raiffeisen_2024.pdf": range(536, 570),
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
# 0B. TOKEN / VOCAB FILTERING (IMPORTANT FOR INTERPRETABILITY)
# ============================================================

TOKEN_MIN_LEN = 3
MIN_DF = 2            # token must appear in at least MIN_DF documents
MAX_DF_RATIO = 1   # drop tokens that appear in more than this share of documents

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

def load_report_paragraphs(reports_folder, page_ranges, strict=True):
    report_paragraphs = []
    report_paragraphs_source = []

    print(f"Looking for PDFs in: {reports_folder}")
    # Build a normalized lookup so minor filename differences don't break matching
    page_ranges_norm = {norm_for_match(k): v for k, v in page_ranges.items()}

    for path, dirs, files in os.walk(reports_folder):
        pdfs = [file for file in files if file.endswith(".pdf")]
        if not pdfs:
            continue
        print("Found PDFs:", pdfs)

        for _file in pdfs:
            print(f"Processing {_file}...")
            full_path = os.path.join(path, _file)

            # Decide which pages to process (STRICT: require explicit page ranges)
            file_key = norm_for_match(_file)
            if file_key not in page_ranges_norm:
                if strict:
                    raise ValueError(
                        f"File '{_file}' not found in page_ranges (after normalization). "
                        "Add an explicit page range for this PDF (no defaults)."
                    )
                else:
                    # When running a filtered subset (e.g., only one year), ignore other PDFs in the folder
                    continue

            pages_to_process = page_ranges_norm[file_key]

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

    # Group paragraphs into one document per file (robust: always returns a DataFrame)
    df_grouped = df.groupby("file", as_index=False).agg({"content": lambda texts: "\n".join(texts)})
    if df_grouped.empty:
        return pd.DataFrame(columns=["file", "content", "year", "bank", "document"])

    # Parse year and bank from filenames.
    # Preferred: <bank>_<year>.pdf  (e.g., barclays_2015.pdf)
    # Fallback:  <year>_<bank>_group.pdf
    def _parse_bank_year(fname: str):
        base = os.path.basename(fname)
        m = re.match(r"^(?P<bank>.+?)_(?P<year>\d{4})\.pdf$", base, flags=re.IGNORECASE)
        if m:
            return m.group("bank"), int(m.group("year"))
        m = re.match(r"^(?P<year>\d{4})_(?P<bank>.+?)_group\.pdf(?:\.pdf)?$", base, flags=re.IGNORECASE)
        if m:
            return m.group("bank"), int(m.group("year"))
        return None, None

    parsed = df_grouped["file"].apply(_parse_bank_year)
    df_grouped["bank"] = parsed.apply(lambda x: x[0])
    df_grouped["year"] = parsed.apply(lambda x: x[1]).astype("Int64")

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

    # Save embeddings next to the repo in a stable location
    out_dir = os.path.join(TEXT_ANALYTICS_DIR, "Noter")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "embedding_matrix.npy")
    np.save(out_path, embedding_matrix)
    print("Saved embedding_matrix.npy to:", out_path)

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
# Extraction summary helpers
# ============================================================

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


def build_extraction_summary(df_docs_before_filter, df_docs_after_filter, page_ranges_year, year_label):
    """Create a bank-year extraction QC table and return it as a DataFrame."""
    # Token counts (before/after filtering)
    tok_before = df_docs_before_filter.set_index("file")["tokens"].apply(lambda x: len(x) if isinstance(x, list) else 0).to_dict()
    tok_after = df_docs_after_filter.set_index("file")["tokens"].apply(lambda x: len(x) if isinstance(x, list) else 0).to_dict() if len(df_docs_after_filter) else {}

    rows = []
    for fname, rng in page_ranges_year.items():
        p_from, p_to, n_pages = _range_to_bounds(rng)
        bank, year_parsed = (None, None)
        m = re.match(r"^(?P<bank>.+?)_(?P<year>\d{4})\.pdf$", fname, flags=re.IGNORECASE)
        if m:
            bank = m.group("bank")
            year_parsed = int(m.group("year"))
        else:
            m2 = re.match(r"^(?P<year>\d{4})_(?P<bank>.+?)_group\.pdf(?:\.pdf)?$", fname, flags=re.IGNORECASE)
            if m2:
                bank = m2.group("bank")
                year_parsed = int(m2.group("year"))
        rows.append({
            "year": year_parsed,
            "run_label": year_label,
            "file": fname,
            "bank": bank,
            "pages_from": p_from,
            "pages_to": p_to,
            "n_pages": n_pages,
            "n_paragraphs": int(df_docs_before_filter.loc[df_docs_before_filter["file"] == fname, "content"].shape[0]) if fname in df_docs_before_filter["file"].values else 0,
            "n_tokens_before": int(tok_before.get(fname, 0)),
            "n_tokens_after": int(tok_after.get(fname, 0)),
            "status": "ok" if tok_after.get(fname, 0) > 0 else "empty_or_missing",
        })

    summary = pd.DataFrame(rows)
    # Make it easier to scan
    summary = summary.sort_values(["status", "n_tokens_after", "n_tokens_before"], ascending=[True, False, False]).reset_index(drop=True)
    return summary


# ============================================================
# Helper: Build TF topic-loading summary at the file/document level
# ============================================================
def build_topic_loading_summary(first_doc_topics_df: pd.DataFrame, df_docs: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Summarize TF topic-loadings at the (file) document level.

    Supports both:
      - long format: columns [cluster_id/topic, document, topic_loading]
      - wide format: columns [document, topic_loading_0, topic_loading_1, ...]

    Returns: DataFrame keyed by file with compact loading summaries.
    """
    if first_doc_topics_df is None or first_doc_topics_df.empty:
        return pd.DataFrame(columns=["file", "tf_l1", "tf_l2", "tf_max_abs", "tf_top_clusters", "tf_top_loadings"])

    wide = first_doc_topics_df.copy()

    # must have document
    if "document" not in wide.columns:
        raise ValueError(f"first_doc_topics_df missing 'document'. Available: {list(wide.columns)}")

    # Case A: long format
    if ("topic_loading" in wide.columns) and (("cluster_id" in wide.columns) or ("topic" in wide.columns)):
        tmp = wide.copy()
        if "cluster_id" not in tmp.columns and "topic" in tmp.columns:
            tmp = tmp.rename(columns={"topic": "cluster_id"})

    else:
        # Case B: wide format (topic_loading_0, topic_loading_1, ...)
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
        # cluster_id: "topic_loading_326" -> 326
        tmp["cluster_id"] = tmp["cluster_id"].str.replace("topic_loading_", "", regex=False).astype(int)

    # attach filename
    tmp = tmp.merge(df_docs[["document", "file"]], on="document", how="left")

    tmp["abs_loading"] = tmp["topic_loading"].abs()

    def _topk_str(g, k):
        g2 = g.sort_values("abs_loading", ascending=False).head(k)
        clusters = ";".join(str(int(x)) for x in g2["cluster_id"].tolist())
        loads = ";".join(f"{float(x):.6f}" for x in g2["topic_loading"].tolist())
        return clusters, loads

    rows = []
    for f, g in tmp.groupby("file"):
        l1 = float(g["abs_loading"].sum())
        l2 = float(np.sqrt((g["topic_loading"] ** 2).sum()))
        mx = float(g["abs_loading"].max()) if len(g) else 0.0
        clusters, loads = _topk_str(g, top_n)

        rows.append({
            "file": f,
            "tf_l1": l1,
            "tf_l2": l2,
            "tf_max_abs": mx,
            "tf_top_clusters": clusters,
            "tf_top_loadings": loads,
        })

    return pd.DataFrame(rows)
# ============================================================
# MAIN PIPELINE
# ============================================================

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
        strict=False
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
    df_paragraphs = preprocess_and_count_words(df_paragraphs)
    df_paragraphs = filter_tokens_with_df_rules(df_paragraphs, tokens_col="tokens")

    print("\n=== STEP 2: Build document-level DataFrame ===")
    df_docs = build_document_dataframe(report_paragraphs, report_sources)
    print(df_docs.head())

    print("\n=== STEP 3: Clean text + tokenize + count words ===")
    df_docs = preprocess_and_count_words(df_docs)
    df_docs = filter_tokens_with_df_rules(df_docs, tokens_col="tokens")

    # Keep a snapshot BEFORE dropping short documents, for QC reporting
    df_docs_before_doclen_filter = df_docs.copy()

    df_docs = df_docs[df_docs["tokens"].apply(len) >= 5].copy()
    # ------------------------------------------------------------
    # QC OUTPUT: Extraction summary table (ideal checkpoint output)
    # ------------------------------------------------------------
    out_folder = "outputs_textual_factors"
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

    print("\n=== STEP 5: Cluster word embeddings (LSH sequential clustering) ===")
    ec, clusters, cluster_words_map, word_cluster_map = cluster_words(
        embedding_matrix,
        cluster_size=80,
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

    if N_TOPICS_PER_CLUSTER == 2:
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
    tf_results["first_doc_topics_df"].to_csv(f"{out_folder}/first_doc_topics.csv", index=False)

    # Only write TF2 if it exists (n_topics_per_cluster >= 2)
    if not tf_results["second_doc_topics_df"].empty:
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
#             or if the associated words are not interpret<truncated__content/>