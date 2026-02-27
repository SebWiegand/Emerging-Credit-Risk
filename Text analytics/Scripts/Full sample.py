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

# ===========================================================
# 0. GLOBAL SETTINGS (edit here)
# ===========================================================

# --- Run / IO ---
STRICT_PAGE_RANGES = True          # If True, require every PDF to be present in page_ranges
RUN_LABEL = "ALL"                  # written into extraction_summary
MIN_DOC_TOKENS = 5                 # drop documents with fewer tokens than this after filtering

# --- Text preprocessing ---
TOKEN_MIN_LEN = 3                  # drop tokens shorter than this
MIN_DF = 5                         # token must appear in at least MIN_DF documents
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

# --- OpenAI embeddings ---
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_BATCH_SIZE = 200

# --- Clustering (LSH / neighbors) ---
N_BITS = 256
N_TABLES = 32
NEIGHBOR_ALG = "lsh"               # "lsh" or "brute"
TARGET_CLUSTER_SIZE = 50           # target / soft cap for words per cluster

# --- Textual Factors / SVD ---
N_TOPICS_PER_CLUSTER = 2           # 1 or 2
# Drop clusters whose first singular value is below this threshold (0 = keep all)
MIN_SINGULAR_VALUE = 50

# --- Optional outputs ---
MERGE_ALL_TF_LOADINGS_IN_EXTRACTION_SUMMARY = True  # makes extraction_summary very wide

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
    clean_and_normalize_text,    # cleans and normalizes the 'content' text.
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
reports_folder = os.path.join(TEXT_ANALYTICS_DIR, "Bank reports")
print("Reports folder:", reports_folder)

# Your own page_ranges (copied from your notebook)


page_ranges = {
    # --- BDS ---
    "bds_2015.pdf": list(chain(range(111, 141), range(195, 238))),
    "bds_2016.pdf": list(chain(range(125, 155), range(215, 264))),
    "bds_2017.pdf": list(chain(range(129, 159), range(232, 282))),
    "bds_2018.pdf": list(chain(range(136, 170), range(245, 287))),
    "bds_2019.pdf": list(chain(range(100, 104), range(308, 360))),
    "bds_2020.pdf": list(chain(range(113, 117), range(245, 249), range(360, 417))),
    "bds_2021.pdf": list(chain(range(93, 98), range(265, 280), range(462, 510))),
    "bds_2022.pdf": list(chain(range(89, 94), range(115, 130), range(318, 327), range(500, 553))),
    "bds_2023.pdf": list(chain(range(127, 133), range(165, 190), range(746, 798))),
    "bds_2024.pdf": list(chain(range(130, 136), range(205, 273), range(600, 662), range(811, 867))),

    # --- Bankinter ---
    "bankinter_2015.pdf": list(chain(range(45, 60))),
    "bankinter_2016.pdf": list(chain(range(46, 61))),
    "bankinter_2017.pdf": list(chain(range(44, 61))),
    "bankinter_2018.pdf": list(chain(range(40, 57))),
    "bankinter_2019.pdf": list(chain(range(59, 75))),
    "bankinter_2020.pdf": list(chain(range(65, 83))),
    "bankinter_2021.pdf": list(chain(range(60, 74))),
    "bankinter_2022.pdf": list(chain(range(73, 89))),
    "bankinter_2023.pdf": list(chain(range(76, 86))),
    "bankinter_2024.pdf": list(chain(range(67, 85))),

    # --- Banco Bilbao (BBVA) ---
    # Some files use a NYSE_ prefix; include both exact-name variants
    "NYSE_BBVA_2015.pdf": list(chain(range(75, 109))),
    "NYSE_BBVA_2016.pdf": list(chain(range(34, 72), range(310, 319))),
    "NYSE_BBVA_2017.pdf": list(chain(range(37, 40), range(102, 109))),
    "NYSE_BBVA_2018.pdf": list(chain(range(39, 43), range(126, 141))),
    "NYSE_BBVA_2019.pdf": list(chain(range(130, 148), range(208, 213), range(302, 341))),
    "NYSE_BBVA_2020.pdf": list(chain(range(185, 206), range(302, 341))),
    "NYSE_BBVA_2021.pdf": list(chain(range(173, 191), range(291, 341))),
    "NYSE_BBVA_2022.pdf": list(chain(range(183, 204), range(335, 385))),
    "NYSE_BBVA_2023.pdf": list(chain(range(284, 306), range(412, 460))),
    "NYSE_BBVA_2024.pdf": list(chain(range(418, 441), range(514, 561))),


    # --- UniCredit ---
    # Some files are misspelled as 'unicredt_YYYY.pdf' in the folder; include both variants
    "unicredit_2015.pdf": list(chain(range(314, 460))),
    "unicredit_2016.pdf": list(chain(range(282, 335))),
    "unicredit_2017.pdf": list(chain(range(177, 260))),
    "unicredit_2018.pdf": list(chain(range(279, 416))),
    "unicredit_2019.pdf": list(chain(range(250, 380))),
    "unicredit_2020.pdf": list(chain(range(263, 399))),
    "unicredit_2021.pdf": list(chain(range(318, 445))),
    "unicredit_2022.pdf": list(chain(range(362, 516))),
    "unicredit_2023.pdf": list(chain(range(397, 568))),
    "unicredit_2024.pdf": list(chain(range(524, 687))),

    # --- Santander ---
    "santander_2015.pdf": list(chain(range(51, 55), range(173, 289))),
    "santander_2016.pdf": list(chain(range(157, 274))),
    "santander_2017.pdf": list(chain(range(195, 287))),
    "santander_2018.pdf": list(chain(range(396, 472), range(687, 716))),
    "santander_2019.pdf": list(chain(range(387, 463), range(696, 732))),
    "santander_2020.pdf": list(chain(range(419, 493), range(782, 794))),
    "santander_2021.pdf": list(chain(range(429, 513), range(730, 765))),
    "santander_2022.pdf": list(chain(range(418, 502), range(727, 763))),
    "santander_2023.pdf": list(chain(range(450, 512), range(742, 778))),
    "santander_2024.pdf": list(chain(range(501, 561), range(787, 823))),

    # --- ING ---
    # Folder uses NYSE_IDG_YYYY.pdf; some runs may also use IDG_YYYY.pdf
    "NYSE_IDG_2015.pdf": list(chain(range(54, 60), range(264, 333), range(358, 379))),
    "NYSE_IDG_2016.pdf": list(chain(range(56, 64), range(249, 319), range(349, 369))),
    "NYSE_IDG_2017.pdf": list(chain(range(42, 51), range(223, 289), range(316, 337))),
    "NYSE_IDG_2018.pdf": list(chain(range(60, 74), range(391, 418))),
    "NYSE_IDG_2019.pdf": list(chain(range(72, 80), range(161, 251))),
    "NYSE_IDG_2020.pdf": list(chain(range(81, 183), range(396, 419))),
    "NYSE_IDG_2021.pdf": list(chain(range(44, 150))),
    "NYSE_IDG_2022.pdf": list(chain(range(102, 187), range(322, 342))),
    "NYSE_IDG_2023.pdf": list(chain(range(130, 206), range(380, 400))),
    "NYSE_IDG_2024.pdf": list(chain(range(42, 105), range(242, 258))),


    # --- Barclays ---
    # Folder uses LSE_BARC_YYYY.pdf; some runs may also use barclays_YYYY.pdf
    "LSE_BARC_2015.pdf": list(chain(range(120, 216))),
    "LSE_BARC_2016.pdf": list(chain(range(12, 14), range(133, 236))),
    "LSE_BARC_2017.pdf": list(chain(range(12, 14), range(116, 196))),
    "LSE_BARC_2018.pdf": list(chain(range(128, 217))),
    "LSE_BARC_2019.pdf": list(chain(range(126, 212))),
    "LSE_BARC_2020.pdf": list(chain(range(142, 238))),
    "LSE_BARC_2021.pdf": list(chain(range(43, 46), range(199, 294))),
    "LSE_BARC_2022.pdf": list(chain(range(57, 59), range(74, 78), range(265, 379))),
    "LSE_BARC_2023.pdf": list(chain(range(52, 55), range(66, 73), range(255, 374))),
    "LSE_BARC_2024.pdf": list(chain(range(51, 54), range(64, 69), range(263, 397))),


    # --- Caixa ---
    "caixa_2015.pdf": list(chain(range(120, 141), range(400, 412))),
    "caixa_2016.pdf": list(chain(range(34, 39), range(107, 128), range(426, 434))),
    "caixa_2017.pdf": list(chain(range(36, 39), range(93, 112), range(416, 424))),
    "caixa_2018.pdf": list(chain(range(41, 44), range(90, 110), range(565, 566))),
    "caixa_2019.pdf": list(chain(range(41, 44), range(81, 98), range(523, 526))),
    "caixa_2020.pdf": list(chain(range(45, 47), range(82, 100))),
    "caixa_2021.pdf": list(chain(range(123, 134), range(270, 274))),
    "caixa_2022.pdf": list(chain(range(153, 183), range(453, 458), range(478, 479))),
    "caixa_2023.pdf": list(chain(range(159, 188))),
    "caixa_2024.pdf": list(chain(range(115, 126), range(819, 827))),

    # --- UBS---
    "UBS_2015.pdf": list(chain(range(60, 77), range(154, 289))),
    "UBS_2016.pdf": list(chain(range(45, 59), range(106, 213))),
    "UBS_2017.pdf": list(chain(range(46, 59), range(100, 211))),
    "UBS_2018.pdf": list(chain(range(52, 66), range(107, 223))),
    "UBS_2019.pdf": list(chain(range(65, 76), range(107, 206))),
    "UBS_2020.pdf": list(chain(range(61, 73), range(95, 187))),
    "UBS_2021.pdf": list(chain(range(68, 81), range(102, 195))),
    "UBS_2022.pdf": list(chain(range(57, 68), range(83, 165))),
    "UBS_2023.pdf": list(chain(range(60, 73), range(96, 183))),
    "UBS_2024.pdf": list(chain(range(49, 63), range(87, 160))),

    # --- La Banque Postale SA ---
    "Banque_postale_2015.pdf": list(chain(range(76, 112))),
    "Banque_postale_2016.pdf": list(chain(range(80, 150))),
    "Banque_postale_2017.pdf": list(chain(range(84, 156))),
    "Banque_postale_2018.pdf": list(chain(range(94, 176))),
    "Banque_postale_2019.pdf": list(chain(range(91, 176))),
    "Banque_postale_2020.pdf": list(chain(range(100, 203))),
    "Banque_postale_2021.pdf": list(chain(range(109, 224))),
    "Banque_postale_2022.pdf": list(chain(range(422, 565))),
    "Banque_postale_2023.pdf": list(chain(range(422, 565))),
    "Banque_postale_2024.pdf": list(chain(range(610, 762))),

    # --- DZ BANK AG ---
    "dz-bank_2015.pdf": list(chain(range(74, 171))),
    "dz-bank_2016.pdf": list(chain(range(73, 186))),
    "dz-bank_2017.pdf": list(chain(range(65, 175))),
    "dz-bank_2018.pdf": list(chain(range(67, 171))),
    "dz-bank_2019.pdf": list(chain(range(61, 161))),
    "dz-bank_2020.pdf": list(chain(range(72, 192))),
    "dz-bank_2021.pdf": list(chain(range(68, 188))),
    "dz-bank_2022.pdf": list(chain(range(71, 197))),
    "dz-bank_2023.pdf": list(chain(range(66, 191))),
    "dz-bank_2024.pdf": list(chain(range(66, 197))),

    # --- Deutsche ---
    "deutsche_2015.pdf": list(chain(range(71, 191))),
    "deutsche_2016.pdf": list(chain(range(81, 202))),
    "deutsche_2017.pdf": list(chain(range(57, 163))),
    "deutsche_2018.pdf": list(chain(range(57, 172))),
    "deutsche_2019.pdf": list(chain(range(61, 188))),
    "deutsche_2020.pdf": list(chain(range(57, 187))),
    "deutsche_2021.pdf": list(chain(range(59, 196))),
    "deutsche_2022.pdf": list(chain(range(59, 204))),
    "deutsche_2023.pdf": list(chain(range(68, 220))),
    "deutsche_2024.pdf": list(chain(range(69, 221))),

    # --- Credit Agricole ---
    "Agricole_2015.pdf": list(chain(range(43, 126))),
    "Agricole_2016.pdf": list(chain(range(48, 139))),
    "Agricole_2017.pdf": list(chain(range(44, 163))),
    "Agricole_2018.pdf": list(chain(range(47, 195))),
    "Agricole_2019.pdf": list(chain(range(47, 196))),
    "Agricole_2020.pdf": list(chain(range(39, 202))),
    "Agricole_2021.pdf": list(chain(range(35, 213))),
    "Agricole_2022.pdf": list(chain(range(37, 259))),
    "Agricole_2023.pdf": list(chain(range(39, 272))),
    "Agricole_2024.pdf": list(chain(range(222, 280))),

    # --- SEB ---
    "SEB_2015.pdf": list(chain(range(7, 63))),
    "SEB_2016.pdf": list(chain(range(7, 61))),
    "SEB_2017.pdf": list(chain(range(8, 74))),
    "SEB_2018.pdf": list(chain(range(8, 93))),
    "SEB_2019.pdf": list(chain(range(2, 59))),
    "SEB_2020.pdf": list(chain(range(2, 57))),
    "SEB_2021.pdf": list(chain(range(4, 79))),
    "SEB_2022.pdf": list(chain(range(2, 78))),
    "SEB_2023.pdf": list(chain(range(2, 80))),
    "SEB_2024.pdf": list(chain(range(2, 78))),

    # --- Nykredit ---
    "nykredit_2015.pdf": list(chain(range(1, 57))),
    "nykredit_2016.pdf": list(chain(range(2, 54))),
    "nykredit_2017.pdf": list(chain(range(1, 56))),
    "nykredit_2018.pdf": list(chain(range(1, 55))),
    "nykredit_2019.pdf": list(chain(range(1, 58))),
    "nykredit_2020.pdf": list(chain(range(0, 69))),
    "nykredit_2021.pdf": list(chain(range(0, 67))),
    "nykredit_2022.pdf": list(chain(range(0, 69))),
    "nykredit_2023.pdf": list(chain(range(0, 71))),
    "nykredit_2024.pdf": list(chain(range(0, 56))),


    # --- Banco BPM ---
    "BPM_2015.pdf": list(chain(range(105, 107), range(283, 358))),
    "BPM_2016.pdf": list(chain(range(103, 106), range(283, 356))),
    "BPM_2017.pdf": list(chain(range(105, 109), range(321, 419))),
    "BPM_2018.pdf": list(chain(range(109, 112), range(329, 434))),
    "BPM_2019.pdf": list(chain(range(104, 109), range(323, 431))),
    "BPM_2020.pdf": list(chain(range(115, 121), range(346, 460))),
    "BPM_2021.pdf": list(chain(range(348, 473))),
    "BPM_2022.pdf": list(chain(range(115, 121), range(375, 519))),
    "BPM_2023.pdf": list(chain(range(129, 135), range(413, 560))),
    "BPM_2024.pdf": list(chain(range(125, 128), range(611, 757))),

    # --- OP Financial Group ---
    "OP_2015.pdf": list(chain(range(277, 330))),
    "OP_2016.pdf": list(chain(range(135, 162))),
    "OP_2017.pdf": list(chain(range(162, 189))),
    "OP_2018.pdf": list(chain(range(180, 213))),
    "OP_2019.pdf": list(chain(range(199, 233))),
    "OP_2020.pdf": list(chain(range(1, 69))),
    "OP_2021.pdf": list(chain(range(1, 31))),
    "OP_2022.pdf": list(chain(range(1, 35))),
    "OP_2023.pdf": list(chain(range(1, 34))),
    "OP_2024.pdf": list(chain(range(530, 543))),

    # --- BPER Banca SPA ---
    "bper-banca-spa_2015.pdf": list(chain(range(106, 118), range(658, 733))),
    "bper-banca-spa_2016.pdf": list(chain(range(118, 130), range(638, 715))),
    "bper-banca-spa_2017.pdf": list(chain(range(94, 106), range(311, 401))),
    "bper-banca-spa_2018.pdf": list(chain(range(75, 102), range(273, 390))),
    "bper-banca-spa_2019.pdf": list(chain(range(73, 89), range(286, 407))),
    "bper-banca-spa_2020.pdf": list(chain(range(24, 43), range(76, 88), range(284, 418))),
    "bper-banca-spa_2021.pdf": list(chain(range(22, 36), range(69, 82), range(278, 412))),
    "bper-banca-spa_2022.pdf": list(chain(range(24, 33), range(66, 76), range(248, 361))),
    "bper-banca-spa_2023.pdf": list(chain(range(54, 63), range(422, 534))),
    "bper-banca-spa_2024.pdf": list(chain(range(56, 66), range(508, 618))),

    # --- Nordea ---
    "nordea_2015.pdf": list(chain(range(0, 61))),
    "nordea_2016.pdf": list(chain(range(0, 115))),
    "nordea_2017.pdf": list(chain(range(0, 173))),
    "nordea_2018.pdf": list(chain(range(0, 75))),
    "nordea_2019.pdf": list(chain(range(0, 45))),
    "nordea_2020.pdf": list(chain(range(0, 48))),
    "nordea_2021.pdf": list(chain(range(0, 52))),
    "nordea_2022.pdf": list(chain(range(0, 61))),
    "nordea_2023.pdf": list(chain(range(1, 59))),
    "nordea_2024.pdf": list(chain(range(0, 70))),

    # --- Danske ---
    "danske_2015.pdf": list(chain(range(0, 90))),
    "danske_2016.pdf": list(chain(range(0, 77))),
    "danske_2017.pdf": list(chain(range(0, 87))),
    "danske_2018.pdf": list(chain(range(0, 83))),
    "danske_2019.pdf": list(chain(range(0, 81))),
    "danske_2020.pdf": list(chain(range(0, 89))),
    "danske_2021.pdf": list(chain(range(0, 97))),
    "danske_2022.pdf": list(chain(range(0, 87))),
    "danske_2023.pdf": list(chain(range(0, 129))),
    "danske_2024.pdf": list(chain(range(0, 125))),

    # --- Intesa Sanpaolo ---
    "intesa-sanpaolo-group_2015.pdf": list(chain(range(292, 386))),
    "intesa-sanpaolo-group_2016.pdf": list(chain(range(312, 408))),
    "intesa-sanpaolo-group_2017.pdf": list(chain(range(324, 444))),
    "intesa-sanpaolo-group_2018.pdf": list(chain(range(382, 516))),
    "intesa-sanpaolo-group_2019.pdf": list(chain(range(371, 507))),
    "intesa-sanpaolo-group_2020.pdf": list(chain(range(403, 546))),
    "intesa-sanpaolo-group_2021.pdf": list(chain(range(411, 575))),
    "intesa-sanpaolo-group_2022.pdf": list(chain(range(414, 582))),
    "intesa-sanpaolo-group_2023.pdf": list(chain(range(428, 593))),
    "intesa-sanpaolo-group_2024.pdf": list(chain(range(632, 797))),

    # --- Erste ---
    "erste_2015.pdf": list(chain(range(195, 238))),
    "erste_2016.pdf": list(chain(range(185, 229))),
    "erste_2017.pdf": list(chain(range(185, 235))),
    "erste_2018.pdf": list(chain(range(199, 243))),
    "erste_2019.pdf": list(chain(range(200, 249))),
    "erste_2020.pdf": list(chain(range(196, 250))),
    "erste_2021.pdf": list(chain(range(200, 251))),
    "erste_2022.pdf": list(chain(range(214, 261))),
    "erste_2023.pdf": list(chain(range(334, 386))),
    "erste_2024.pdf": list(chain(range(295, 349))),


    # --- Handelsbanken ---
    "handelsbanken_2015.pdf": list(chain(range(0, 126))),
    "handelsbanken_2016.pdf": list(chain(range(0, 51))),
    "handelsbanken_2017.pdf": list(chain(range(0, 49))),
    "handelsbanken_2018.pdf": list(chain(range(0, 48))),
    "handelsbanken_2019.pdf": list(chain(range(0, 52))),
    "handelsbanken_2020.pdf": list(chain(range(0, 58))),
    "handelsbanken_2021.pdf": list(chain(range(0, 57))),
    "handelsbanken_2022.pdf": list(chain(range(0, 56))),
    "handelsbanken_2023.pdf": list(chain(range(0, 74))),
    "handelsbanken_2024.pdf": list(chain(range(0, 100))),

    # --- DNB ---
    "DNB_2015.pdf": list(chain(range(0, 104))),
    "DNB_2016.pdf": list(chain(range(0, 75))),
    "DNB_2017.pdf": list(chain(range(0, 67))),
    "DNB_2018.pdf": list(chain(range(0, 59))),
    "DNB_2019.pdf": list(chain(range(0, 65))),
    "DNB_2020.pdf": list(chain(range(0, 74))),
    "DNB_2021.pdf": list(chain(range(0, 69))),
    "DNB_2022.pdf": list(chain(range(0, 64))),
    "DNB_2023.pdf": list(chain(range(0, 91))),
    "DNB_2024.pdf": list(chain(range(0, 97))),

    # --- Banca Monte dei Paschi ---
    "banca_monte_2015.pdf": list(chain(range(232, 308))),
    "banca_monte_2016.pdf": list(chain(range(296, 420))),
    "banca_monte_2017.pdf": list(chain(range(323, 451))),
    "banca_monte_2018.pdf": list(chain(range(344, 487))),
    "banca_monte_2019.pdf": list(chain(range(83, 87), range(324, 457))),
    "banca_monte_2020.pdf": list(chain(range(113, 119), range(372, 510))),
    "banca_monte_2021.pdf": list(chain(range(105, 112), range(385, 534))),
    "banca_monte_2022.pdf": list(chain(range(107, 114), range(381, 536))),
    "banca_monte_2023.pdf": list(chain(range(75, 85), range(340, 483))),
    "banca_monte_2024.pdf": list(chain(range(77, 85), range(502, 648))),
}



# ============================================================
# 1. LOAD TEXT FROM PDF´s
# ============================================================

# Flat-lookup version for loading report paragraphs
def load_report_paragraphs(reports_folder, page_ranges, strict=True):
    report_paragraphs = []
    report_paragraphs_source = []

    print(f"Looking for PDFs in: {reports_folder}")

    for path, dirs, files in os.walk(reports_folder):
        pdfs = [file for file in files if file.lower().endswith(".pdf")]
        if not pdfs:
            continue
        print("Found PDFs:", pdfs)

        for _file in pdfs:
            print(f"Processing {_file}...")
            full_path = os.path.join(path, _file)

            if _file not in page_ranges:
                if strict:
                    raise ValueError(
                        f"File '{_file}' not found in page_ranges. "
                        "Add it explicitly or rename the file."
                    )
                else:
                    continue

            pages_to_process = page_ranges[_file]

            with fitz.open(full_path) as doc:
                total_pages = len(doc)

                if pages_to_process is None:
                    pages_to_process = range(total_pages)

                actual_pages = []
                for page_num in pages_to_process:
                    if isinstance(page_num, int):
                        if page_num < 0:
                            actual_page = total_pages + page_num
                        else:
                            actual_page = page_num

                        if 0 <= actual_page < total_pages:
                            actual_pages.append(actual_page)

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

def train_openai_embeddings(df, model_name: str, batch_size: int):
    """
    Build word embeddings using OpenAI's embedding API.
    Trains on paragraph-level tokens.
    """
    vocab = sorted(set(chain.from_iterable(df["tokens"].tolist())))
    print(f"Vocabulary size: {len(vocab)} words")

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

def cluster_words(
        embedding_matrix: np.ndarray,
        target_cluster_size: int = 50,
        neighbor_alg: str = NEIGHBOR_ALG,
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
# QC HELPERS (extraction summary)
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
            "company": bank,
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

def main():
    print("\n=== STEP 1: Load paragraphs from PDFs ===")
    page_ranges_year = page_ranges
    print("Running analysis on FULL corpus (all years)")
    print(f"Total files included: {len(page_ranges_year)}")

    report_paragraphs, report_sources = load_report_paragraphs(
        reports_folder,
        page_ranges_year,
        strict=STRICT_PAGE_RANGES
    )
    print(f"Loaded {len(report_paragraphs)} paragraphs")
    if len(report_paragraphs) == 0:
        raise RuntimeError(
            f"Loaded 0 paragraphs. Check PDFs exist in: {reports_folder} and filenames match page_ranges keys."
        )
    df_paragraphs = pd.DataFrame({
        "content": report_paragraphs,
        "file": report_sources
    })
    # Clean + tokenize paragraphs for Word2Vec training
    df_paragraphs = preprocess_text_and_tokens(df_paragraphs, text_col="content", tokens_col="tokens")

    print("\n=== STEP 2: Build document-level DataFrame ===")
    df_docs = build_document_dataframe(report_paragraphs, report_sources)

    print("\n=== STEP 3: Clean text + tokenize + count words ===")
    df_docs = preprocess_text_and_tokens(df_docs, text_col="content", tokens_col="tokens")
    df_docs_before_doclen_filter = df_docs.copy()

    df_docs = df_docs[df_docs["tokens"].apply(len) >= MIN_DOC_TOKENS].copy()
    out_folder = os.path.join(TEXT_ANALYTICS_DIR, "outputs_textual_factors")
    os.makedirs(out_folder, exist_ok=True)

    extraction_summary = build_extraction_summary(
        df_docs_before_filter=df_docs_before_doclen_filter,
        df_docs_after_filter=df_docs,
        page_ranges_year=page_ranges_year,
        year_label=RUN_LABEL,
    )
    summary_path = os.path.join(out_folder, "extraction_summary_ALL_V1.csv")
    extraction_summary.to_csv(summary_path, index=False)
    print(f"\nSaved extraction summary to: {summary_path}")
    print(extraction_summary.head(10))


    print("\n=== STEP 4: Create OpenAI Embeddings ===")
    vocab, embedding_matrix = train_openai_embeddings(
        df_paragraphs,
        model_name=EMBEDDING_MODEL,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    print(f"Vocabulary size: {len(vocab)}")

    print(f"\n=== STEP 5: Cluster word embeddings (sequential clustering; target_cluster_size={TARGET_CLUSTER_SIZE}) ===")
    ec, clusters, cluster_words_map, word_cluster_map = cluster_words(
        embedding_matrix,
        target_cluster_size=TARGET_CLUSTER_SIZE,
        neighbor_alg=NEIGHBOR_ALG
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

    # ------------------------------------------------------------
    # Drop topics with low singular values - only if applied
    # ------------------------------------------------------------
    if MIN_SINGULAR_VALUE > 0:
        sv_df = tf_results["singular_values_df"].copy()

        # Column names differ across helper versions; support both.
        # Current expected columns (from your output CSV): cluster, leading_singular, second_singular
        sv_cluster_col = "cluster" if "cluster" in sv_df.columns else ("cluster_id" if "cluster_id" in sv_df.columns else None)
        sv_value_col = (
            "leading_singular" if "leading_singular" in sv_df.columns
            else ("singular_value" if "singular_value" in sv_df.columns
                  else ("singular_values" if "singular_values" in sv_df.columns else None))
        )

        if sv_cluster_col is None or sv_value_col is None:
            raise KeyError(
                f"singular_values_df missing expected columns. "
                f"Found columns: {list(sv_df.columns)}"
            )

        # Keep only clusters meeting threshold (use leading singular value)
        keep_clusters = sv_df.loc[
            sv_df[sv_value_col] >= MIN_SINGULAR_VALUE,
            sv_cluster_col
        ].astype(int).tolist()

        total_clusters = int(sv_df[sv_cluster_col].nunique())
        print(f"Dropping clusters with {sv_value_col} < {MIN_SINGULAR_VALUE}")
        print(f"Keeping {len(keep_clusters)} out of {total_clusters} clusters")

        # Filter document-level loadings (these frames use 'cluster_id')
        if "cluster_id" in tf_results["first_doc_topics_df"].columns:
            tf_results["first_doc_topics_df"] = tf_results["first_doc_topics_df"][
                tf_results["first_doc_topics_df"]["cluster_id"].isin(keep_clusters)
            ].copy()

        if not tf_results["second_doc_topics_df"].empty and "cluster_id" in tf_results["second_doc_topics_df"].columns:
            tf_results["second_doc_topics_df"] = tf_results["second_doc_topics_df"][
                tf_results["second_doc_topics_df"]["cluster_id"].isin(keep_clusters)
            ].copy()

        # Filter word-level loadings
        if "cluster_id" in tf_results["topics_words_df"].columns:
            tf_results["topics_words_df"] = tf_results["topics_words_df"][
                tf_results["topics_words_df"]["cluster_id"].isin(keep_clusters)
            ].copy()

        if "topics_words2_df" in tf_results and not tf_results["topics_words2_df"].empty and "cluster_id" in tf_results["topics_words2_df"].columns:
            tf_results["topics_words2_df"] = tf_results["topics_words2_df"][
                tf_results["topics_words2_df"]["cluster_id"].isin(keep_clusters)
            ].copy()

        # Filter singular values and importances
        tf_results["singular_values_df"] = sv_df[
            sv_df[sv_cluster_col].isin(keep_clusters)
        ].copy()

        if "cluster_id" in tf_results["topic_importances_df"].columns:
            tf_results["topic_importances_df"] = tf_results["topic_importances_df"][
                tf_results["topic_importances_df"]["cluster_id"].isin(keep_clusters)
            ].copy()
        elif "cluster" in tf_results["topic_importances_df"].columns:
            tf_results["topic_importances_df"] = tf_results["topic_importances_df"][
                tf_results["topic_importances_df"]["cluster"].isin(keep_clusters)
            ].copy()

    if N_TOPICS_PER_CLUSTER < 2:
        print("\nNote: N_TOPICS_PER_CLUSTER=1, so TF2 outputs are skipped.")

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