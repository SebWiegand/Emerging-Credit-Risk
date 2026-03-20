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
CONG_REP_DIR = os.path.join(TEXT_ANALYTICS_DIR, "Cong et al. rep")   -   # .../Text analytics/Cong et al. rep
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
# Reports live in:      <repo>/Text analytics/Virksomheder/
reports_folder = os.path.join(TEXT_ANALYTICS_DIR, "Virksomheder")
print("Reports folder:", reports_folder)

# Your own page_ranges (copied from your notebook)
# ============================================================
# PAGE RANGES — ORGANIZED BY YEAR → BANK
# NOTE: ranges are PDF page indices (0-based, end exclusive)
# ============================================================
from itertools import chain

page_ranges = {
    # National Grid
    "ngg_2024.pdf": list(chain(range(21, 30), range(193, 212))),
    "ngg_2023.pdf": list(chain(range(17, 29), range(188, 216))),
    "ngg_2022.pdf": list(chain([5], range(40, 43), range(59, 60))),
    "ngg_2020.pdf": list(chain(range(21, 25), range(81, 85), range(183, 203))),
    "ngg_2019.pdf": list(chain(range(19, 22), range(163, 173))),
    "ngg_2018.pdf": list(chain(range(17, 21), range(159, 186))),
    "ngg_2017.pdf": list(chain(range(14, 18), range(146, 171))),
    "ngg_2016.pdf": list(chain(range(25, 29), range(150, 170))),
    "ngg_2015.pdf": list(chain(range(37, 41), range(142, 177))),
    "ngg_2014.pdf": list(chain(range(23, 24), range(138, 150))),
    "ngg_2013.pdf": list(chain(range(29, 33), range(148, 158))),
    "ngg_2012.pdf": list(chain(range(39, 46), range(155, 163))),
    "ngg_2011.pdf": list(chain(range(31, 34), range(151, 158))),
    "ngg_2010.pdf": list(chain(range(89, 95), range(158, 165))),
    "ngg_2009.pdf": list(chain(range(39, 40), range(167, 173))),
    "ngg_2007.pdf": list(chain([28], [132], range(136, 139))),
    "ngg_2006.pdf": list(chain([28], range(57, 58), range(74, 75))),
    "ngg_2005.pdf": list(chain(range(61, 68))),

    # Red Eléctrica
    "red_2024.pdf": list(chain(range(55, 58), range(114, 118))),
    "red_2023.pdf": list(chain(range(22, 24), [60])),
    "red_2022.pdf": list(chain(range(61, 64), range(122, 126), range(133, 136), [179])),
    "red_2021.pdf": list(chain(range(47, 48), range(82, 86), range(153, 157))),
    "red_2020.pdf": list(chain(range(63, 66), range(125, 127))),
    "red_2019.pdf": list(chain(range(53, 55), range(110, 123))),
    "red_2018.pdf": list(chain(range(76, 80), [138])),
    "red_2017.pdf": list(chain([15])),
    "red_2016.pdf": list(chain([18])),
    "red_2015.pdf": list(chain([16])),
    "red_2014.pdf": list(chain([14])),
    "red_2013.pdf": list(chain([14])),
    "red_2012.pdf": list(chain([28])),
    "red_2011.pdf": list(chain(range(39, 42))),
    "red_2010.pdf": list(chain(range(40, 44))),
    "red_2009.pdf": list(chain(range(51, 54))),
    "red_2008.pdf": list(chain([27])),
    "red_2007.pdf": list(chain(range(91, 93))),
    "red_2006.pdf": list(chain([85])),
    "red_2005.pdf": list(chain([90])),

    # REC Silicon
    "rec_2024.pdf": list(chain(range(20, 24), range(32, 40), range(91, 94))),
    "rec_2023.pdf": list(chain(range(21, 23), [34], range(59, 61), range(81, 83))),
    "rec_2022.pdf": list(chain(range(11, 13), [32], range(56, 68))),
    "rec_2021.pdf": list(chain(range(11, 14), range(20, 28), [42], [79])),
    "rec_2020.pdf": list(chain(range(12, 19), range(39, 42))),
    "rec_2019.pdf": list(chain(range(12, 17), range(40, 42))),
    "rec_2018.pdf": list(chain(range(12, 17), range(42, 44))),
    "rec_2017.pdf": list(chain(range(9, 11), range(36, 38))),
    "rec_2016.pdf": list(chain([10], range(36, 38))),
    "rec_2015.pdf": list(chain(range(9, 11), [34])),
    "rec_2014.pdf": list(chain(range(9, 11), range(34, 37))),
    "rec_2013.pdf": list(chain(range(9, 11), range(34, 36))),
    "rec_2012.pdf": list(chain(range(18, 21))),
    "rec_2011.pdf": list(chain(range(18, 21), range(66, 69))),

    # Prysmian
    "prysmian_2024.pdf": list(chain(range(77, 88), range(365, 371))),
    "prysmian_2023.pdf": list(chain(range(77, 88), range(365, 371))),
    "prysmian_2022.pdf": list(chain(range(76, 89), range(347, 358))),
    "prysmian_2021.pdf": list(chain(range(88, 107), range(234, 251))),
    "prysmian_2020.pdf": list(chain(range(94, 110), range(146, 163))),
    "prysmian_2019.pdf": list(chain(range(97, 112), range(157, 170))),
    "prysmian_2018.pdf": list(chain(range(94, 109), range(174, 184))),
    "prysmian_2017.pdf": list(chain(range(85, 99), range(147, 158))),
    "prysmian_2016.pdf": list(chain(range(84, 104), range(189, 199))),
    "prysmian_2015.pdf": list(chain(range(84, 102), range(175, 187))),
    "prysmian_2014.pdf": list(chain(range(92, 110), range(180, 190))),
    "prysmian_2013.pdf": list(chain(range(74, 92), range(263, 267))),
    "prysmian_2012.pdf": list(chain(range(94, 109), range(182, 191))),

    # Neste Oyj
    "neste_2024.pdf": list(chain(range(78, 84), range(126, 129), range(161, 169))),
    "neste_2023.pdf": list(chain(range(134, 140), [163], range(183, 191))),
    "neste_2022.pdf": list(chain(range(135, 140), range(183, 191))),
    "neste_2021.pdf": list(chain(range(131, 136), range(173, 183))),
    "neste_2020.pdf": list(chain([57], [62], range(101, 106), range(140, 148))),
    "neste_2019.pdf": list(chain(range(88, 93), range(135, 143))),
    "neste_2018.pdf": list(chain(range(87, 93), range(133, 141))),
    "neste_2017.pdf": list(chain(range(80, 86), range(125, 134))),
    "neste_2016.pdf": list(chain(range(79, 85), range(130, 143))),
    "neste_2015.pdf": list(chain(range(75, 80), range(118, 133))),
    "neste_2014.pdf": list(chain(range(41, 44), range(147, 152), range(192, 201))),
    "neste_2013.pdf": list(chain(range(16, 18), range(201, 203), range(226, 239))),
    "neste_2012.pdf": list(chain(range(61, 64), range(234, 244))),
    "neste_2011.pdf": list(chain(range(261, 273), range(349, 360))),
    "neste_2010.pdf": list(chain(range(72, 77), range(106, 117))),
    "neste_2009.pdf": list(chain(range(55, 59), range(92, 99))),
    "neste_2008.pdf": list(chain(range(69, 73), range(90, 98))),
    "neste_2007.pdf": list(chain(range(43, 47), range(60, 63), range(94, 102))),
    "neste_2006.pdf": list(chain(range(35, 38), [86, 87])),
    "neste_2005.pdf": list(chain(range(20, 24), range(46, 48))),

    # ERG
    "ERG_2024.pdf": list(chain(range(40, 51), range(331, 335))),
    "ERG_2023.pdf": list(chain(range(55, 72), range(310, 315))),
    "ERG_2022.pdf": list(chain(range(64, 81), range(334, 339))),
    "ERG_2021.pdf": list(chain(range(60, 75), range(312, 328), [294], [169])),
    "ERG_2020.pdf": list(chain(range(52, 68), range(339, 391))),
    "ERG_2019.pdf": list(chain(range(63, 81), range(328, 337), [139, 140])),
    "ERG_2018.pdf": list(chain(range(61, 79), range(369, 378))),
    "ERG_2017.pdf": list(chain(range(65, 81), range(323, 330))),
    "ERG_2016.pdf": list(chain(range(57, 72), range(378, 382))),
    "ERG_2015.pdf": list(chain(range(58, 69), range(362, 366))),
    "ERG_2014.pdf": list(chain(range(59, 66), range(230, 236))),
    "ERG_2013.pdf": list(chain(range(62, 66), range(227, 233))),
    "ERG_2012.pdf": list(chain(range(60, 64), range(285, 290))),
    "ERG_2011.pdf": list(chain(range(52, 57), range(263, 269))),
    "ERG_2010.pdf": list(chain(range(57, 62), range(295, 300))),
    "ERG_2009.pdf": list(chain(range(54, 58), range(253, 256))),
    "ERG_2008.pdf": list(chain(range(17, 21), range(75, 85))),

    # Iberdrola
    "iberdrola_2024.pdf": list(chain(range(124, 127))),
    "iberdrola_2023.pdf": list(chain(range(106, 110))),
    "iberdrola_2022.pdf": list(chain(range(103, 107))),
    "iberdrola_2021.pdf": list(chain(range(83, 87))),
    "iberdrola_2020.pdf": list(chain(range(89, 93))),
    "iberdrola_2019.pdf": list(chain(range(81, 86))),
    "iberdrola_2018.pdf": list(chain(range(87, 101))),
    "iberdrola_2017.pdf": list(chain(range(89, 94))),
    "iberdrola_2016.pdf": list(chain(range(87, 92))),
    "iberdrola_2015.pdf": list(chain(range(79, 82))),
    "iberdrola_2014.pdf": list(chain(range(70, 73))),
    "iberdrola_2010.pdf": list(chain(range(164, 168))),

    # E.ON
    "EON_2024.pdf": list(chain(range(98, 104), range(197, 200))),
    "EON_2023.pdf": list(chain(range(119, 128), range(213, 217))),
    "EON_2022.pdf": list(chain(range(124, 134), range(271, 275), [212])),
    "EON_2021.pdf": list(chain(range(82, 92), range(247, 250), [37])),
    "EON_2020.pdf": list(chain(range(58, 67), [8], range(205, 209))),
    "EON_2019.pdf": list(chain(range(196, 201), range(43, 50))),
    "EON_2018.pdf": list(chain(range(204, 209), range(43, 50))),
    "EON_2017.pdf": list(chain(range(198, 203), range(55, 62))),
    "EON_2016.pdf": list(chain(range(63, 72), range(198, 203))),
    "EON_2015.pdf": list(chain(range(61, 71), range(190, 195))),
    "EON_2014.pdf": list(chain(range(61, 69), range(190, 195))),
    "EON_2013.pdf": list(chain(range(61, 69), range(190, 195))),
    "EON_2012.pdf": list(chain(range(65, 75), range(179, 183))),
    "EON_2011.pdf": list(chain(range(51, 67), range(146, 151))),
    "EON_2010.pdf": list(chain(range(41, 57), range(136, 141), [9])),
    "EON_2008.pdf": list(chain(range(45, 54), range(135, 138), [11])),
    "EON_2007.pdf": list(chain([35], range(65, 78), range(196, 201))),
    "EON_2006.pdf": list(chain([166], range(61, 69), range(176, 179))),
    "EON_2005.pdf": list(chain(range(45, 48), range(59, 61))),

    # Neoen SAS
    "neoen_2023.pdf": list(chain(range(100, 104))),
    "neoen_2022.pdf": list(chain(range(106, 110), [61])),
    "neoen_2021.pdf": list(chain(range(97, 100), [8])),
    "neoen_2020.pdf": list(chain(range(92, 95), [9])),
    "neoen_2019.pdf": list(chain(range(36, 58))),

    # Ørsted
    "orsted_2005.pdf": list(chain(range(18, 23), range(101, 106))),
    "orsted_2006.pdf": list(chain(range(34, 42), range(118, 124))),
    "orsted_2007.pdf": list(chain(range(41, 45), [175, 176])),
    "orsted_2008.pdf": list(chain(range(43, 50), [174, 175, 176])),
    "orsted_2009.pdf": list(chain(range(31, 38), [194, 195])),
    "orsted_2010.pdf": list(chain(range(57, 66))),
    "orsted_2011.pdf": list(chain(range(47, 59))),
    "orsted_2012.pdf": list(chain(range(30, 41))),
    "orsted_2013.pdf": list(chain(range(34, 47))),
    "orsted_2014.pdf": list(chain(range(34, 46))),
    "orsted_2015.pdf": list(chain(range(33, 46))),
    "orsted_2016.pdf": list(chain(range(44, 63))),
    "orsted_2017.pdf": list(chain(range(44, 60))),
    "orsted_2018.pdf": list(chain(range(145, 158), range(58, 71))),
    "orsted_2019.pdf": list(chain(range(58, 64), range(134, 147))),
    "orsted_2020.pdf": list(chain(range(69, 75), range(145, 157))),
    "orsted_2021.pdf": list(chain(range(29, 37), range(131, 149))),
    "orsted_2022.pdf": list(chain(range(37, 43), range(132, 153))),
    "orsted_2023.pdf": list(chain(range(31, 40), range(208, 224))),
    "orsted_2024.pdf": list(chain(range(25, 33), range(216, 232))),

    # Nordex SE
    "Nordex_2024.pdf": list(chain(range(47, 68), range(261, 269))),
    "Nordex_2023.pdf": list(chain(range(53, 71), range(117, 126))),
    "Nordex_2022.pdf": list(chain(range(55, 72), range(123, 133))),
    "Nordex_2021.pdf": list(chain(range(57, 72), range(115, 123))),
    "Nordex_2020.pdf": list(chain(range(52, 65), range(105, 113))),
    "Nordex_2019.pdf": list(chain(range(50, 64), range(102, 111))),
    "Nordex_2018.pdf": list(chain(range(52, 66), range(107, 113))),
    "Nordex_2017.pdf": list(chain(range(55, 68), range(111, 115))),
    "Nordex_2016.pdf": list(chain(range(80, 98), range(129, 134))),

    # NEL ASA
    "nel_2024.pdf": list(chain(range(24, 27), range(129, 133))),
    "nel_2023.pdf": list(chain([25], range(128, 136))),
    "nel_2022.pdf": list(chain([25], range(108, 114))),
    "nel_2021.pdf": list(chain([46], range(106, 110))),
    "nel_2020.pdf": list(chain(range(16, 18), range(77, 83))),
    "nel_2019.pdf": list(chain(range(16, 18), range(83, 90))),
    "nel_2017.pdf": list(chain([16], range(68, 74))),
    "nel_2016.pdf": list(chain(range(16, 18), range(73, 80))),
    "nel_2015.pdf": list(chain(range(29, 34))),
    "nel_2014.pdf": list(chain(range(14, 22), range(56, 60))),

    # Meyer Burger Technology AG
    "meyer_2023.pdf": list(chain([129, 130])),
    "meyer_2022.pdf": list(chain(range(118, 121), [5])),
    "meyer_2021.pdf": list(chain([20])),
    "meyer_2020.pdf": list(chain([18], range(91, 94))),
    "meyer_2019.pdf": list(chain([90, 91])),
    "meyer_2018.pdf": list(chain(range(98, 101))),
    "meyer_2017.pdf": list(chain([96, 97])),
    "meyer_2016.pdf": list(chain(range(110, 113))),
    "meyer_2015.pdf": list(chain(range(104, 107))),

    # ITM Power
    "ITM_2024.pdf": list(chain(range(34, 39), [95])),
    "ITM_2023.pdf": list(chain(range(48, 54), [116])),
    "ITM_2022.pdf": list(chain(range(30, 35), [104])),
    "ITM_2021.pdf": list(chain(range(21, 25), range(94, 98))),
    "ITM_2020.pdf": list(chain(range(63, 71), range(174, 177))),
    "ITM_2019.pdf": list(chain(range(51, 54), range(133, 139))),
    "ITM_2018.pdf": list(chain(range(61, 65), range(125, 130))),
    "ITM_2017.pdf": list(chain(range(39, 43), range(116, 189))),
    "ITM_2016.pdf": list(chain(range(27, 29), range(96, 99))),
    "ITM_2015.pdf": list(chain([23], range(90, 93))),

    # Enel
    "Enel_2024.pdf": list(chain(range(531, 541), range(51, 54), range(62, 77), range(103, 122))),
    "Enel_2023.pdf": list(chain(range(391, 401), range(92, 131))),
    "Enel_2022.pdf": list(chain(range(113, 150), range(411, 417))),
    "Enel_2021.pdf": list(chain(range(78, 96), range(97, 130))),
    "Enel_2020.pdf": list(chain(range(66, 108))),
    "Enel_2019.pdf": list(chain(range(56, 76))),
    "Enel_2018.pdf": list(chain(range(145, 151))),
    "Enel_2017.pdf": list(chain(range(145, 152))),
    "Enel_2016.pdf": list(chain(range(138, 142), range(274, 283))),
    "Enel_2015.pdf": list(chain(range(125, 131), range(265, 274))),
    "Enel_2014.pdf": list(chain(range(101, 107), range(249, 257))),
    "Enel_2013.pdf": list(chain(range(98, 104), range(170, 181))),
    "Enel_2012.pdf": list(chain(range(91, 98), range(166, 174))),
    "Enel_2011.pdf": list(chain(range(92, 98), range(163, 172))),
    "Enel_2010.pdf": list(chain(range(109, 114), range(170, 179))),
    "Enel_2009.pdf": list(chain(range(139, 145), range(193, 204))),
    "Enel_2008.pdf": list(chain(range(123, 129))),
    "Enel_2007.pdf": list(chain(range(155, 171))),
    "Enel_2006.pdf": list(chain(range(144, 154))),
    "Enel_2005.pdf": list(chain(range(134, 137))),

    # EDP
    "EDP_2024.pdf": list(chain(range(40, 53))),
    "EDP_2023.pdf": list(chain(range(38, 54))),
    "EDP_2022.pdf": list(chain(range(33, 45))),
    "EDP_2021.pdf": list(chain(range(147, 167))),
    "EDP_2020.pdf": list(chain(range(337, 397))),
    "EDP_2019.pdf": list(chain(range(301, 326))),
    "EDP_2018.pdf": list(chain(range(302, 309))),
    "EDP_2017.pdf": list(chain(range(156, 162))),
    "EDP_2016.pdf": list(chain(range(320, 326))),
    "EDP_2015.pdf": list(chain(range(299, 305))),
    "EDP_2014.pdf": list(chain(range(267, 271))),
    "EDP_2013.pdf": list(chain(range(211, 215))),
    "EDP_2012.pdf": list(chain(range(190, 194))),
    "EDP_2011.pdf": list(chain(range(181, 185))),
    "EDP_2010.pdf": list(chain(range(132, 139))),
    "EDP_2009.pdf": list(chain(range(166, 169), range(117, 124))),
    "EDP_2008.pdf": list(chain(range(50, 54))),
    "EDP_2007.pdf": list(chain([10])),
    "EDP_2006.pdf": list(chain(range(4, 12))),
    "EDP_2005.pdf": list(chain(range(5, 11))),

    # Encavis AG
    "Encavis_2024.pdf": list(chain(range(33, 46))),
    "Encavis_2023.pdf": list(chain(range(45, 59))),
    "Encavis_2022.pdf": list(chain(range(43, 55))),
    "Encavis_2021.pdf": list(chain(range(43, 61))),
    "Encavis_2020.pdf": list(chain(range(49, 68))),
    "Encavis_2019.pdf": list(chain(range(45, 63))),
    "Encavis_2018.pdf": list(chain(range(50, 62))),
    "Encavis_2017.pdf": list(chain(range(49, 62))),
    "Encavis_2016.pdf": list(chain(range(57, 66))),
    "Encavis_2015.pdf": list(chain(range(27, 32))),
    "Encavis_2014.pdf": list(chain(range(53, 60))),
    "Encavis_2013.pdf": list(chain(range(48, 53))),
    "Encavis_2012.pdf": list(chain(range(51, 57))),

    # Ceres
    "Ceres_2024.pdf": list(chain(range(40, 44))),
    "Ceres_2023.pdf": list(chain(range(61, 62), [4])),
    "Ceres_2022.pdf": list(chain(range(98, 99), [40])),
    "Ceres_2021.pdf": list(chain(range(91, 92), [50])),
    "Ceres_2020.pdf": list(chain(range(89, 91))),
    "Ceres_2019.pdf": list(chain(range(74, 76))),
    "Ceres_2018.pdf": list(chain(range(64, 65))),
    "Ceres_2017.pdf": list(chain(range(58, 59))),
    "Ceres_2016.pdf": list(chain(range(52, 54))),
    "Ceres_2015.pdf": list(chain(range(44, 45))),
    "Ceres_2014.pdf": list(chain(range(34, 35))),
    "Ceres_2013.pdf": list(chain([32])),

    # Vestas
    "vestas_2005.pdf": list(chain(range(50, 54))),
    "vestas_2006.pdf": list(chain(range(16, 21), range(80, 87))),
    "vestas_2007.pdf": list(chain(range(18, 23), range(80, 90))),
    "vestas_2008.pdf": list(chain(range(18, 27), range(84, 94))),
    "vestas_2009.pdf": list(chain(range(17, 28), range(102, 114))),
    "vestas_2010.pdf": list(chain(range(19, 26), range(118, 128))),
    "vestas_2011.pdf": list(chain(range(24, 34), range(95, 105))),
    "vestas_2012.pdf": list(chain(range(29, 35), range(99, 109))),
    "vestas_2013.pdf": list(chain(range(30, 35), range(84, 97))),
    "vestas_2014.pdf": list(chain(range(34, 37), range(93, 103))),
    "vestas_2015.pdf": list(chain(range(48, 58), range(99, 106))),
    "vestas_2016.pdf": list(chain(range(38, 45), range(85, 96))),
    "vestas_2017.pdf": list(chain(range(40, 45), range(94, 104))),
    "vestas_2018.pdf": list(chain(range(45, 48), range(77, 87))),
    "vestas_2019.pdf": list(chain(range(51, 56), range(86, 96))),
    "vestas_2020.pdf": list(chain(range(29, 32), range(45, 513))),
    "vestas_2021.pdf": list(chain(range(43, 47), range(103, 113))),
    "vestas_2022.pdf": list(chain(range(49, 54), range(104, 112))),
    "vestas_2023.pdf": list(chain(range(37, 48), range(80, 88))),
    "vestas_2024.pdf": list(chain(range(49, 89), range(167, 175))),

    # RWE AG
    "RWE_2005.pdf": list(chain(range(70, 80), range(153, 160))),
    "RWE_2006.pdf": list(chain(range(81, 90))),
    "RWE_2007.pdf": list(chain(range(99, 116))),
    "RWE_2008.pdf": list(chain(range(99, 121))),
    "RWE_2009.pdf": list(chain(range(97, 118))),
    "RWE_2010.pdf": list(chain(range(118, 143))),
    "RWE_2011.pdf": list(chain(range(61, 65))),
    "RWE_2012.pdf": list(chain(range(88, 107))),
    "RWE_2013.pdf": list(chain(range(85, 111))),
    "RWE_2014.pdf": list(chain(range(74, 97))),
    "RWE_2015.pdf": list(chain(range(79, 93))),
    "RWE_2016.pdf": list(chain(range(78, 92))),
    "RWE_2017.pdf": list(chain(range(74, 88))),
    "RWE_2018.pdf": list(chain(range(74, 87))),
    "RWE_2019.pdf": list(chain(range(84, 97))),
    "RWE_2020.pdf": list(chain(range(64, 80), range(165, 182))),
    "RWE_2021.pdf": list(chain(range(66, 80), range(155, 175))),
    "RWE_2022.pdf": list(chain(range(62, 75), range(183, 200))),
    "RWE_2023.pdf": list(chain(range(59, 72), range(189, 207))),
    "RWE_2024.pdf": list(chain(range(61, 103), range(259, 276))),

    #SMA
    "smasolar_2013.pdf": list(chain(range(89, 117))),
    "smasolar_2014.pdf": list(chain(range(82, 104))),
    "smasolar_2015.pdf": list(chain(range(56, 70))),
    "smasolar_2016.pdf": list(chain(range(82, 101))),
    "smasolar_2017.pdf": list(chain(range(81, 100))),
    "smasolar_2018.pdf": list(chain(range(57, 75))),
    "smasolar_2019.pdf": list(chain(range(56, 74))),
    "smasolar_2020.pdf": list(chain(range(59, 79))),
    "smasolar_2021.pdf": list(chain(range(62, 87))),
    "smasolar_2022.pdf": list(chain(range(82, 104))),
    "smasolar_2023.pdf": list(chain(range(83, 116))),
    "smasolar_2024.pdf": list(chain(range(43, 87), range(236, 241))),

    #Scated
    "scatec_2013.pdf": list(chain(range(9, 14), range(33, 36))),
    "scatec_2014.pdf": list(chain(range(23, 30), range(54, 58))),
    "scatec_2015.pdf": list(chain(range(23, 30), range(61, 65))),
    "scatec_2016.pdf": list(chain(range(29, 36), range(85, 100))),
    "scatec_2017.pdf": list(chain(range(38, 45), range(114, 121))),
    "scatec_2018.pdf": list(chain(range(23, 32), range(53, 61))),
    "scatec_2019.pdf": list(chain(range(24, 32), range(54, 67))),
    "scatec_2020.pdf": list(chain(range(23, 33), range(56, 66))),
    "scatec_2021.pdf": list(chain(range(37, 43), range(122, 130))),
    "scatec_2022.pdf": list(chain(range(46, 54), range(85, 95))),
    "scatec_2023.pdf": list(chain(range(35, 44), range(79, 85))),
    "scatec_2024.pdf": list(chain(range(28, 34), range(158, 169))),

    #Simens
    "siemens_2005.pdf": list(chain(range(35, 42))),
    "siemens_2006.pdf": list(chain(range(38, 45))),
    "siemens_2007.pdf": list(chain(range(45, 52))),
    "siemens_2008.pdf": list(chain(range(32, 38))),
    "siemens_2009.pdf": list(chain(range(28, 33))),
    "siemens_2010.pdf": list(chain(range(206, 213), range(279, 299))),
    "siemens_2011.pdf": list(chain(range(218, 256), range(325, 341))),
    "siemens_2012.pdf": list(chain(range(200, 225), range(288, 303))),
    "siemens_2013.pdf": list(chain(range(225, 250), range(313, 330))),
    "siemens_2014.pdf": list(chain(range(222, 246), range(298, 310))),
    "siemens_2015.pdf": list(chain(range(24, 37), range(94, 102))),
    "siemens_2016.pdf": list(chain(range(22, 38), range(88, 98))),
    "siemens_2017.pdf": list(chain(range(24, 40), range(90, 100))),
    "siemens_2018.pdf": list(chain(range(29, 44), range(95, 105))),
    "siemens_2019.pdf": list(chain(range(27, 41), range(113, 121))),
    "siemens_2020.pdf": list(chain(range(30, 49), range(133, 143))),
    "siemens_2021.pdf": list(chain(range(25, 38), range(73, 79))),
    "siemens_2022.pdf": list(chain(range(22, 35), range(68, 74))),
    "siemens_2023.pdf": list(chain(range(21, 36), range(76, 83))),
    "siemens_2024.pdf": list(chain(range(21, 33), range(76, 84))),

    # Solaria
    "solaria_2020.pdf": list(chain(range(1, 16))),
    "solaria_2021.pdf": list(chain(range(23, 30), range(58, 63))),
    "solaria_2022.pdf": list(chain(range(28, 34), range(67, 73))),
    "solaria_2023.pdf": list(chain(range(29, 37), range(85, 91))),
    "solaria_2024.pdf": list(chain(range(25, 36), range(62, 70), range(81, 86))),

    #SSE PLC
    "SSE_2005.pdf": list(chain(range(26, 34))),
    "SSE_2006.pdf": list(chain(range(29, 38), range(85, 90))),
    "SSE_2007.pdf": list(chain(range(37, 45), range(94, 98))),
    "SSE_2008.pdf": list(chain(range(41, 51), range(103, 111))),
    "SSE_2009.pdf": list(chain(range(52, 70), range(131, 145))),
    "SSE_2010.pdf": list(chain(range(52, 68), range(136, 149))),
    "SSE_2011.pdf": list(chain(range(46, 70), range(136, 152))),
    "SSE_2012.pdf": list(chain(range(61, 78), range(151, 167))),
    "SSE_2013.pdf": list(chain(range(70, 90), range(156, 170))),
    "SSE_2014.pdf": list(chain(range(22, 32), range(150, 166))),
    "SSE_2015.pdf": list(chain(range(12, 18), range(66, 90), range(172, 188))),
    "SSE_2016.pdf": list(chain(range(16, 24), range(68, 95), range(175, 192))),
    "SSE_2017.pdf": list(chain(range(24, 30), range(174, 190))),
    "SSE_2018.pdf": list(chain(range(23, 36), range(217, 231))),
    "SSE_2019.pdf": list(chain(range(64, 91), range(238, 254))),
    "SSE_2020.pdf": list(chain(range(21, 40), range(261, 306))),
    "SSE_2021.pdf": list(chain(range(40, 66), range(119, 146), range(272, 286))),
    "SSE_2022.pdf": list(chain(range(42, 83), range(312, 330))),
    "SSE_2023.pdf": list(chain(range(31, 48), range(150, 158))),
    "SSE_2024.pdf": list(chain(range(59, 160), range(297, 308))),

    #Terna
    "Terna_2005.pdf": list(chain(range(261, 334))),
    "Terna_2006.pdf": list(chain(range(167, 195), range(317, 335))),
    "terna_2007.pdf": list(chain(range(151, 177), range(285, 309))),
    "terna_2008.pdf": list(chain(range(65, 73), range(136, 151), range(215, 232))),
    "terna_2009.pdf": list(chain(range(64, 74), range(113, 132), range(215, 222))),
    "terna_2010.pdf": list(chain(range(76, 82), range(180, 190))),
    "terna_2011.pdf": list(chain(range(81, 97), range(173, 180))),
    "terna_2012.pdf": list(chain(range(83, 97), range(218, 229))),
    "terna_2013.pdf": list(chain(range(63, 83), range(291, 300))),
    "terna_2014.pdf": list(chain(range(61, 78), range(300, 311))),
    "terna_2015.pdf": list(chain(range(70, 88), range(243, 252))),
    "terna_2016.pdf": list(chain(range(62, 90), range(213, 223))),
    "terna_2017.pdf": list(chain(range(124, 137))),
    "terna_2018.pdf": list(chain(range(20, 25), range(203, 213))),
    "terna_2019.pdf": list(chain(range(70, 83), range(226, 241))),
    "terna_2020.pdf": list(chain(range(86, 101), range(265, 275))),
    "terna_2021.pdf": list(chain(range(55, 83), range(163, 170))),
    "terna_2022.pdf": list(chain(range(62, 96))),
    "terna_2023.pdf": list(chain(range(58, 113), range(370, 376))),
    "terna_2024.pdf": list(chain(range(154, 224))),

    #Verbio
    "verbio_2006.pdf": list(chain(range(39, 50), range(108, 112))),
    "verbio_2007.pdf": list(chain(range(30, 34), range(90, 98))),
    "verbio_2008.pdf": list(chain(range(49, 56), range(125, 136))),
    "verbio_2009.pdf": list(chain(range(46, 52), range(117, 129))),
    "verbio_2010.pdf": list(chain(range(69, 77), range(145, 155))),
    "verbio_2011.pdf": list(chain(range(51, 57), range(123, 135))),
    "verbio_2012.pdf": list(chain(range(33, 43), range(99, 111))),
    "verbio_2013.pdf": list(chain(range(41, 50), range(108, 119))),
    "verbio_2014.pdf": list(chain(range(36, 45), range(93, 103))),
    "verbio_2015.pdf": list(chain(range(33, 47), range(93, 102))),
    "verbio_2016.pdf": list(chain(range(33, 48), range(95, 104))),
    "verbio_2017.pdf": list(chain(range(34, 48), range(93, 102))),
    "verbio_2018.pdf": list(chain(range(34, 49), range(97, 105))),
    "verbio_2019.pdf": list(chain(range(33, 46), range(93, 102))),
    "verbio_2020.pdf": list(chain(range(42, 56), range(101, 111))),
    "verbio_2021.pdf": list(chain(range(38, 52), range(106, 115))),
    "verbio_2022.pdf": list(chain(range(42, 58), range(110, 121))),
    "verbio_2023.pdf": list(chain(range(45, 70), range(155, 180))),
    "verbio_2024.pdf": list(chain(range(45, 71), range(161, 172))),

    #Verbund
    "verbund_2007.pdf": list(chain(range(60, 69), range(123, 133))),
    "verbund_2008.pdf": list(chain(range(79, 89), range(149, 161))),
    "verbund_2009.pdf": list(chain(range(82, 96), range(179, 193))),
    "verbund_2010.pdf": list(chain(range(93, 105), range(190, 197))),
    "verbund_2011.pdf": list(chain(range(65, 76), range(201, 214))),
    "verbund_2012.pdf": list(chain(range(46, 65), range(191, 199))),
    "verbund_2013.pdf": list(chain(range(46, 56), range(191, 199))),
    "verbund_2014.pdf": list(chain(range(44, 56), range(180, 190))),
    "verbund_2015.pdf": list(chain(range(97, 105), range(222, 234))),
    "verbund_2016.pdf": list(chain(range(105, 111), range(228, 240))),
    "verbund_2017.pdf": list(chain(range(101, 115), range(250, 258))),
    "verbund_2018.pdf": list(chain(range(100, 120), range(254, 265))),
    "verbund_2019.pdf": list(chain(range(101, 122), range(254, 263))),
    "verbund_2020.pdf": list(chain(range(110, 133), range(275, 286))),
    "verbund_2021.pdf": list(chain(range(110, 131), range(292, 300))),
    "verbund_2022.pdf": list(chain(range(120, 142), range(320, 330))),
    "verbund_2023.pdf": list(chain(range(120, 146), range(347, 355))),
    "verbund_2024.pdf": list(chain(range(128, 156), range(456, 466))),

    #Wacker
    "wacker_2005.pdf": list(chain(range(71, 75))),
    "wacker_2006.pdf": list(chain(range(62, 70))),
    "wacker_2007.pdf": list(chain(range(83, 90))),
    "wacker_2008.pdf": list(chain(range(147, 151))),
    "wacker_2009.pdf": list(chain(range(116, 129), range(203, 212))),
    "wacker_2010.pdf": list(chain(range(114, 134), range(213, 219))),
    "wacker_2011.pdf": list(chain(range(114, 134), range(213, 219))),
    "wacker_2012.pdf": list(chain(range(122, 141), range(228, 235))),
    "wacker_2013.pdf": list(chain(range(142, 168), range(256, 264))),
    "wacker_2014.pdf": list(chain(range(144, 170), range(254, 268))),
    "wacker_2015.pdf": list(chain(range(136, 170), range(253, 260))),
    "wacker_2016.pdf": list(chain(range(90, 116), range(166, 173))),
    "wacker_2017.pdf": list(chain(range(82, 107), range(156, 164))),
    "wacker_2018.pdf": list(chain(range(81, 105), range(163, 170))),
    "wacker_2019.pdf": list(chain(range(74, 97), range(151, 163))),
    "wacker_2020.pdf": list(chain(range(85, 108), range(164, 170))),
    "wacker_2021.pdf": list(chain(range(87, 109), range(163, 170))),
    "wacker_2022.pdf": list(chain(range(96, 119), range(173, 181))),
    "wacker_2023.pdf": list(chain(range(94, 125), range(208, 217))),
    "wacker_2024.pdf": list(chain(range(91, 128), range(299, 306))),

    #Acciona
    "Acciona_2015.pdf": list(chain(range(40, 57), range(159, 173))),
    "Acciona_2016.pdf": list(chain(range(28, 57), range(140, 165))),
    "Acciona_2017.pdf": list(chain(range(27, 43), range(117, 125), range(157, 174))),
    "Acciona_2018.pdf": list(chain(range(56, 65), range(135, 155))),
    "Acciona_2019.pdf": list(chain(range(27, 78), range(132, 151))),
    "Acciona_2020.pdf": list(chain(range(16, 20), range(60, 65))),
    "Acciona_2021.pdf": list(chain(range(21, 25), range(92, 103))),
    "Acciona_2022.pdf": list(chain(range(20, 26), range(146, 161))),
    "Acciona_2023.pdf": list(chain(range(41, 65), range(248, 263))),
    "Acciona_2024.pdf": list(chain(range(31, 56), range(345, 375))),

    #Endesa
    "endesa_2024.pdf": list(chain(range(70, 97), range(647, 657))),
    "endesa_2023.pdf": list(chain(range(55, 118), range(423, 430))),
    "endesa_2022.pdf": list(chain(range(55, 113), range(405, 414))),
    "endesa_2020.pdf": list(chain(range(60, 63), range(95, 105), range(118, 119))),
    "endesa_2019.pdf": list(chain(range(120, 141), range(467, 468))),
    "endesa_2018.pdf": list(chain(range(78, 82), range(574, 577))),
    "endesa_2017.pdf": list(chain(range(80, 84), range(602, 606))),
    "endesa_2016.pdf": list(chain(range(75, 79), range(429, 431))),
    "endesa_2015.pdf": list(chain(range(78, 82), range(427, 429), range(491, 495))),
    "endesa_2014.pdf": list(chain(range(77, 80), range(112, 125), range(417, 419))),
    "endesa_2013.pdf": list(chain(range(219, 228))),
    "endesa_2012.pdf": list(chain(range(88, 94), range(149, 160))),
    "endesa_2011.pdf": list(chain(range(89, 97), range(155, 167), range(240, 245))),
    "endesa_2010.pdf": list(chain(range(206, 225), range(322, 325))),
    "endesa_2009.pdf": list(chain(range(77, 82), range(257, 264))),
    "endesa_2008.pdf": list(chain(range(69, 74), range(148, 157), range(239, 249))),
    "endesa_2007.pdf": list(chain(range(53, 59))),
    "endesa_2006.pdf": list(chain(range(131, 142))),
    "endesa_2005.pdf": list(chain(range(171, 176))),

    #Grenergy
    "grenergy_2024.pdf": list(chain(range(114, 117), range(176, 180))),
    "grenergy_2023.pdf": list(chain(range(99, 104), range(155, 158))),
    "grenergy_2022.pdf": list(chain(range(85, 88), range(139, 145))),
    "grenergy_2021.pdf": list(chain(range(85, 90), range(132, 137))),
    "grenergy_2020.pdf": list(chain(range(78, 81), range(109, 112))),
    "grenergy_2019.pdf": list(chain(range(78, 102))),
    "grenergy_2018.pdf": list(chain(range(67, 84))),

    #drax
    "drax_2008.pdf": list(chain(range(22, 31), range(76, 82))),
    "drax_2009.pdf": list(chain(range(32, 38), range(95, 102))),
    "drax_2010.pdf": list(chain(range(27, 34), range(93, 99))),
    "drax_2011.pdf": list(chain(range(33, 48), range(103, 109))),
    "drax_2012.pdf": list(chain(range(30, 44), range(107, 112))),
    "drax_2013.pdf": list(chain(range(44, 55), range(124, 135))),
    "drax_2014.pdf": list(chain(range(45, 54), range(125, 132))),
    "drax_2015.pdf": list(chain(range(51, 60), range(137, 145))),
    "drax_2016.pdf": list(chain(range(54, 64), range(158, 166))),
    "drax_2017.pdf": list(chain(range(48, 63), range(162, 171))),
    "drax_2018.pdf": list(chain(range(45, 56), range(164, 175))),
    "drax_2019.pdf": list(chain(range(48, 64), range(186, 211))),
    "drax_2020.pdf": list(chain(range(60, 80), range(209, 226))),
    "drax_2021.pdf": list(chain(range(47, 96), range(246, 277))),
    "drax_2022.pdf": list(chain(range(52, 96), range(257, 276))),
    "drax_2023.pdf": list(chain(range(58, 110), range(256, 276))),
    "drax_2024.pdf": list(chain(range(60, 90), range(240, 266))),

    #Greenvold
    "greenvolt_2023.pdf": list(chain(range(58, 76), range(144, 186))),
    "greenvolt_2024.pdf": list(chain(range(33, 91), range(166, 192))),

    #Tyssen
    "Thyssenkrupp_2023.pdf": list(chain(range(63, 80), range(125, 134))),
    "Thyssenkrupp_2024.pdf": list(chain(range(78, 98), range(143, 150))),

    #Umicore
    "umicore_2005.pdf": list(chain(range(76, 85), range(118, 124))),
    "umicore_2006.pdf": list(chain(range(77, 84), range(119, 125))),
    "umicore_2007.pdf": list(chain(range(89, 97), range(142, 147))),
    "umicore_2008.pdf": list(chain(range(49, 53), range(75, 80))),
    "umicore_2009.pdf": list(chain(range(84, 90), range(133, 138))),
    "umicore_2010.pdf": list(chain(range(85, 93), range(137, 142))),
    "umicore_2011.pdf": list(chain(range(60, 69), range(155, 163))),
    "umicore_2012.pdf": list(chain(range(85, 94), range(187, 194))),
    "umicore_2013.pdf": list(chain(range(59, 69), range(166, 173))),
    "umicore_2014.pdf": list(chain(range(65, 73), range(171, 178))),
    "umicore_2015.pdf": list(chain(range(56, 66), range(168, 175))),
    "umicore_2016.pdf": list(chain(range(34, 48), range(103, 114))),
    "umicore_2017.pdf": list(chain(range(31, 47), range(109, 116))),
    "umicore_2018.pdf": list(chain(range(41, 54), range(110, 120))),
    "umicore_2019.pdf": list(chain(range(58, 69), range(112, 125))),
    "umicore_2020.pdf": list(chain(range(73, 87), range(122, 129))),
    "umicore_2021.pdf": list(chain(range(91, 108), range(119, 130))),
    "umicore_2022.pdf": list(chain(range(129, 150), range(165, 173))),
    "umicore_2023.pdf": list(chain(range(181, 195), range(210, 219))),
    "umicore_2024.pdf": list(chain(range(52, 58), range(72, 81))),

    #Cadelar
    "cadeler_2020.pdf": list(chain(range(12, 19), range(55, 59))),
    "cadeler_2021.pdf": list(chain(range(10, 14), range(61, 66))),
    "cadeler_2022.pdf": list(chain(range(13, 18), range(108, 118))),
    "cadeler_2023.pdf": list(chain(range(16, 24), range(149, 162))),
    "cadeler_2024.pdf": list(chain(range(20, 75), range(199, 210))),

    #fortum
    "fortum_2005.pdf": list(chain(range(42, 49), range(55, 57))),
    "fortum_2006.pdf": list(chain(range(51, 58))),
    "fortum_2007.pdf": list(chain(range(16, 25), range(30, 46))),
    "fortum_2008.pdf": list(chain(range(98, 107), range(119, 134))),
    "fortum_2009.pdf": list(chain(range(67, 80), range(97, 107))),
    "fortum_2010.pdf": list(chain(range(81, 92))),
    "fortum_2011.pdf": list(chain(range(22, 30), range(53, 64))),
    "fortum_2012.pdf": list(chain(range(44, 56), range(223, 230))),
    "fortum_2013.pdf": list(chain(range(54, 66), range(99, 112))),
    "fortum_2014.pdf": list(chain(range(101, 111))),
    "fortum_2015.pdf": list(chain(range(18, 26), range(35, 43))),
    "fortum_2016.pdf": list(chain(range(31, 40))),
    "fortum_2017.pdf": list(chain(range(35, 45), range(55, 70))),
    "fortum_2018.pdf": list(chain(range(30, 47), range(65, 77))),
    "fortum_2019.pdf": list(chain(range(41, 53), range(66, 78))),
    "fortum_2020.pdf": list(chain(range(19, 39), range(56, 64))),
    "fortum_2021.pdf": list(chain(range(19, 38), range(54, 64))),
    "fortum_2022.pdf": list(chain(range(30, 44))),
    "fortum_2023.pdf": list(chain(range(37, 50), range(65, 75))),
    "fortum_2024.pdf": list(chain(range(22, 33), range(124, 136))),

    # MOL
    "mol_2024.pdf": list(chain([5], range(8, 10), [39], range(238, 240), range(228, 229))),
    "mol_2023.pdf": list(chain([7], range(10, 14), range(112, 120))),
    "mol_2022.pdf": list(chain([7], range(9, 12), [43], [165])),
    "mol_2021.pdf": list(chain([7], range(9, 12), [39], range(94, 95))),
    "mol_2020.pdf": list(chain(range(9, 11), [37], range(97, 98), [121])),
    "mol_2019.pdf": list(chain(range(8, 9), range(95, 101))),
    "mol_2018.pdf": list(chain(range(9, 10), range(96, 103))),
    "mol_2017.pdf": list(chain([17], range(49, 50), range(87, 88))),
    "mol_2016.pdf": list(chain([17], range(45, 46), [104])),
    "mol_2015.pdf": list(chain([16], [43], range(89, 92))),
    "mol_2014.pdf": list(chain([40], [51], range(77, 78), [120])),
    "mol_2013.pdf": list(chain([6], [32], [122])),
    "mol_2012.pdf": list(chain([5], range(32, 33), range(66, 67))),
    "mol_2011.pdf": list(chain([5], range(34, 35), range(69, 71))),
    "mol_2010.pdf": list(chain([5], range(38, 49), range(73, 76))),
    "mol_2009.pdf": list(chain([37], [51], [119])),
    "mol_2008.pdf": list(chain([73], [99], range(152, 158))),
    "mol_2007.pdf": list(chain(range(31, 33), range(72, 75))),
    "mol_2006.pdf": list(chain([4], range(22, 24), range(59, 61))),
    "mol_2005.pdf": list(chain(range(4, 7))),


    # TotalEnergies
    "totalenergies_2024.pdf": list(chain(range(130, 188))),
    "totalenergies_2023.pdf": list(chain(range(128, 188))),
    "totalenergies_2022.pdf": list(chain(range(118, 176))),
    "totalenergies_2021.pdf": list(chain(range(120, 178))),
    "totalenergies_2020.pdf": list(chain(range(90, 136))),
    "totalenergies_2019.pdf": list(chain(range(82, 130))),
    "totalenergies_2018.pdf": list(chain(range(74, 112))),
    "totalenergies_2017.pdf": list(chain(range(74, 104))),
    "totalenergies_2016.pdf": list(chain(range(64, 89))),
    "totalenergies_2015.pdf": list(chain(range(64, 83))),
    "totalenergies_2014.pdf": list(chain(range(74, 100))),
    "totalenergies_2013.pdf": list(chain(range(77, 103))),
    "totalenergies_2012.pdf": list(chain(range(74, 95))),
    "totalenergies_2011.pdf": list(chain(range(72, 92))),
    "totalenergies_2010.pdf": list(chain(range(64, 85))),
    "totalenergies_2009.pdf": list(chain(range(72, 90))),
    "totalenergies_2008.pdf": list(chain(range(70, 86))),
    "totalenergies_2007.pdf": list(chain(range(64, 82))),
    "totalenergies_2006.pdf": list(chain(range(76, 92))),
    "totalenergies_2005.pdf": list(chain(range(76, 92))),

    # Naturgy Energy Group
    "naturgy_2024.pdf": list(chain(range(69, 77), range(103, 124))),
    "naturgy_2023.pdf": list(chain(range(69, 75), range(100, 125))),
    "naturgy_2022.pdf": list(chain(range(226, 245), range(370, 405))),
    "naturgy_2021.pdf": list(chain(range(205, 226), range(445, 476))),
    "naturgy_2020.pdf": list(chain(range(250, 278), range(340, 354))),
    "naturgy_2019.pdf": list(chain(range(187, 210), range(280, 306))),
    "naturgy_2018.pdf": list(chain(range(169, 185), range(215, 230))),
    "naturgy_2017.pdf": list(chain(range(190, 207), range(250, 265))),
    "naturgy_2016.pdf": list(chain(range(70, 81), range(326, 336))),
    "naturgy_2015.pdf": list(chain(range(60, 66), range(302, 311))),
    # 2014 intentionally skipped (you wrote: den er fucked)
    "naturgy_2013.pdf": list(chain(range(50, 60), range(170, 177))),
    "naturgy_2012.pdf": list(chain(range(40, 50), range(226, 236))),
    "naturgy_2011.pdf": list(chain(range(40, 50), range(273, 285))),
    "naturgy_2010.pdf": list(chain(range(50, 60), range(250, 260))),
    "naturgy_2009.pdf": list(chain(range(150, 164), range(228, 240))),
    "naturgy_2008.pdf": list(chain(range(125, 132), range(224, 236))),
    "naturgy_2007.pdf": list(chain(range(135, 142))),
    "naturgy_2006.pdf": list(chain(range(106, 114))),
    "naturgy_2005.pdf": list(chain(range(86, 94))),

    # MVV
    "mvv_2005.pdf": list(chain(range(48, 57))),
    "mvv_2006.pdf": list(chain(range(48, 55), range(120, 127))),
    "mvv_2007.pdf": list(chain(range(53, 63))),
    "mvv_2008.pdf": list(chain(range(12, 24))),
    "mvv_2009.pdf": list(chain(range(81, 90))),
    "mvv_2010.pdf": list(chain(range(93, 106))),
    "mvv_2011.pdf": list(chain(range(94, 106))),
    "mvv_2012.pdf": list(chain(range(1, 36))),
    "mvv_2013.pdf": list(chain(range(16, 23))),
    "mvv_2014.pdf": list(chain(range(88, 116), range(158, 168))),
    "mvv_2015.pdf": list(chain(range(1, 10))),
    "mvv_2016.pdf": list(chain(range(44, 64))),
    "mvv_2017.pdf": list(chain(range(100, 117))),
    "mvv_2018.pdf": list(chain(range(107, 125))),
    "mvv_2019.pdf": list(chain(range(74, 90))),
    "mvv_2020.pdf": list(chain(range(78, 91))),
    "mvv_2021.pdf": list(chain(range(95, 110), range(156, 169))),
    "mvv_2022.pdf": list(chain(range(110, 127), range(179, 190))),
    "mvv_2023.pdf": list(chain(range(110, 139), range(200, 213))),
    "mvv_2024.pdf": list(chain(range(117, 139), range(206, 224))),

    # FUGRO NV
    "fugro_2024.pdf": list(chain(range(107, 115))),
    "fugro_2023.pdf": list(chain(range(68, 75), range(150, 159))),
    "fugro_2022.pdf": list(chain(range(71, 80), range(165, 176))),
    "fugro_2021.pdf": list(chain(range(66, 77), range(169, 180))),
    "fugro_2020.pdf": list(chain(range(70, 81), range(185, 200))),
    "fugro_2019.pdf": list(chain(range(67, 78), range(176, 193))),
    "fugro_2018.pdf": list(chain(range(58, 66), range(162, 174))),
    "fugro_2017.pdf": list(chain(range(58, 66), range(161, 177))),
    "fugro_2016.pdf": list(chain(range(53, 62), range(157, 172))),
    "fugro_2015.pdf": list(chain(range(67, 75), range(163, 179))),
    "fugro_2014.pdf": list(chain(range(65, 72), range(162, 178))),
    "fugro_2013.pdf": list(chain(range(73, 80), range(173, 193))),
    "fugro_2012.pdf": list(chain(range(85, 92), range(169, 188))),
    "fugro_2011.pdf": list(chain(range(74, 80), range(152, 170))),
    "fugro_2010.pdf": list(chain(range(72, 78), range(148, 166))),
    "fugro_2009.pdf": list(chain(range(66, 71), range(130, 145))),
    "fugro_2008.pdf": list(chain(range(62, 68), range(125, 138))),
    "fugro_2007.pdf": list(chain(range(54, 60), range(117, 131))),
    "fugro_2006.pdf": list(chain(range(55, 59), range(116, 117))),
    "fugro_2005.pdf": list(chain(range(56, 60), range(109, 111))),

    # CropEnergies
    "cropenergies_2022.pdf": list(chain(range(74, 91))),
    "cropenergies_2021.pdf": list(chain(range(66, 83))),
    "cropenergies_2020.pdf": list(chain(range(86, 102), range(154, 156))),
    "cropenergies_2019.pdf": list(chain(range(59, 72), range(125, 127))),
    "cropenergies_2018.pdf": list(chain(range(56, 80), range(123, 125))),
    "cropenergies_2017.pdf": list(chain(range(56, 80), range(123, 126))),
    "cropenergies_2016.pdf": list(chain(range(68, 82), range(123, 125))),
    "cropenergies_2015.pdf": list(chain(range(69, 80), range(125, 127))),
    "cropenergies_2014.pdf": list(chain(range(61, 75), range(130, 132))),
    "cropenergies_2013.pdf": list(chain(range(63, 72), range(122, 124))),
    "cropenergies_2012.pdf": list(chain(range(62, 69), range(110, 112))),
    "cropenergies_2011.pdf": list(chain(range(56, 61), range(99, 103))),
    "cropenergies_2010.pdf": list(chain(range(52, 58))),
    "cropenergies_2009.pdf": list(chain(range(52, 56))),
    "cropenergies_2008.pdf": list(chain(range(42, 44), range(77, 79), range(49, 50))),
    "cropenergies_2007.pdf": list(chain(range(36, 38), range(40, 41))),
    "cropenergies_2006.pdf": list(chain(range(30, 33), range(59, 60), [37])),

    # SFC
    "sfc_2024.pdf": list(chain(range(59, 87))),
    "sfc_2023.pdf": list(chain(range(58, 82))),
    "sfc_2022.pdf": list(chain(range(58, 80))),
    "sfc_2021.pdf": list(chain(range(57, 78))),
    "sfc_2020.pdf": list(chain(range(88, 105))),
    "sfc_2019.pdf": list(chain(range(79, 98))),
    "sfc_2018.pdf": list(chain(range(79, 95))),
    "sfc_2017.pdf": list(chain(range(77, 94))),
    "sfc_2016.pdf": list(chain(range(78, 85))),
    "sfc_2015.pdf": list(chain(range(73, 86))),
    "sfc_2014.pdf": list(chain(range(72, 78))),
    "sfc_2013.pdf": list(chain(range(67, 73))),
    "sfc_2012.pdf": list(chain(range(53, 60))),
    "sfc_2011.pdf": list(chain(range(54, 60))),
    "sfc_2010.pdf": list(chain(range(56, 60))),
    "sfc_2009.pdf": list(chain(range(44, 49))),
    "sfc_2008.pdf": list(chain(range(36, 39))),
    "sfc_2007.pdf": list(chain(range(45, 47))),
    "sfc_2006.pdf": list(chain([27])),

    # AFC (LSE_AFC)
    "LSE_AFC_2024.pdf": list(chain(range(18, 23), range(50, 51))),
    "LSE_AFC_2023.pdf": list(chain(range(27, 29), range(97, 98))),
    # 2022 incomplete (45-47, 114-?) → second range needs end page
    "LSE_AFC_2022.pdf": list(chain(range(45, 47))),
    "LSE_AFC_2021.pdf": list(chain(range(31, 32), range(94, 95))),
    "LSE_AFC_2020.pdf": list(chain(range(47, 49), range(95, 97))),
    "LSE_AFC_2019.pdf": list(chain(range(42, 46))),
    "LSE_AFC_2018.pdf": list(chain(range(12, 13), range(38, 39))),
    "LSE_AFC_2017.pdf": list(chain(range(25, 26), [56])),
    "LSE_AFC_2016.pdf": list(chain([19], [21])),
    "LSE_AFC_2015.pdf": list(chain([12], [18], range(47, 48))),
    "LSE_AFC_2014.pdf": list(chain([8], range(37, 39))),
    "LSE_AFC_2013.pdf": list(chain([13], range(39, 40))),
    "LSE_AFC_2012.pdf": list(chain([29], range(50, 51))),
    "LSE_AFC_2011.pdf": list(chain([13], range(40, 41))),
    "LSE_AFC_2010.pdf": list(chain([11], range(34, 35))),
    "LSE_AFC_2009.pdf": list(chain([9], [32])),
    "LSE_AFC_2008.pdf": list(chain([28])),
    "LSE_AFC_2007.pdf": list(chain([28])),

    # Invinity (LSE_IES)
    "LSE_IES_2024.pdf": list(chain(range(17, 18), range(84, 87))),
    "LSE_IES_2023.pdf": list(chain(range(15, 18), range(83, 86))),
    "LSE_IES_2022.pdf": list(chain(range(18, 20), range(87, 93))),
    "LSE_IES_2021.pdf": list(chain(range(19, 21), range(87, 90))),
    "LSE_IES_2020.pdf": list(chain(range(9, 12), range(88, 92))),
    "LSE_IES_2019.pdf": list(chain(range(7, 10), range(60, 63))),
    "LSE_IES_2018.pdf": list(chain(range(10, 13), range(60, 62))),
    "LSE_IES_2017.pdf": list(chain([8], [12], range(51, 54))),
    "LSE_IES_2016.pdf": list(chain([13], range(52, 54))),
    "LSE_IES_2015.pdf": list(chain(range(54, 57))),
    "LSE_IES_2014.pdf": list(chain([15], range(52, 54))),
    "LSE_IES_2013.pdf": list(chain(range(58, 60))),
    "LSE_IES_2012.pdf": list(chain(range(55, 58))),
    "LSE_IES_2011.pdf": list(chain([13], range(63, 68))),
    "LSE_IES_2010.pdf": list(chain([19], range(77, 78))),
    "LSE_IES_2008.pdf": list(chain([27])),
    "LSE_IES_2007.pdf": list(chain([13], range(57, 58))),
    "LSE_IES_2006.pdf": list(chain(range(16, 17), [44])),

    # Kemira Oyj
    "kemira_2005.pdf": list(chain(range(1, 54))),
    "kemira_2006.pdf": list(chain(range(37, 47), range(96, 103))),
    # 2007 unreadable → intentionally skipped
    "kemira_2008.pdf": list(chain(range(38, 45), range(107, 115))),
    "kemira_2009.pdf": list(chain(range(53, 61), range(117, 127))),
    "kemira_2010.pdf": list(chain(range(25, 35))),
    "kemira_2011.pdf": list(chain(range(65, 104), range(305, 315))),
    "kemira_2012.pdf": list(chain(range(137, 160), range(209, 215))),
    "kemira_2013.pdf": list(chain(range(120, 131), range(195, 205))),
    "kemira_2014.pdf": list(chain(range(89, 115), range(130, 155))),
    "kemira_2015.pdf": list(chain(range(117, 139), range(182, 192))),
    "kemira_2016.pdf": list(chain(range(120, 126), range(167, 177))),
    "kemira_2017.pdf": list(chain(range(26, 58), range(184, 194))),
    "kemira_2018.pdf": list(chain(range(14, 24), range(185, 195))),
    "kemira_2019.pdf": list(chain(range(13, 23), range(185, 195))),
    "kemira_2020.pdf": list(chain(range(38, 45), range(117, 126))),
    "kemira_2021.pdf": list(chain(range(41, 59), range(194, 214))),
    "kemira_2022.pdf": list(chain(range(1, 34))),
    "kemira_2023.pdf": list(chain(range(22, 62), range(150, 160))),
    "kemira_2024.pdf": list(chain(range(72, 157), range(213, 221))),

    # A2A
    "A2A_2007.pdf": list(chain(range(35, 42))),
    "A2A_2008.pdf": list(chain(range(96, 108))),
    "A2A_2009.pdf": list(chain(range(116, 128))),
    "A2A_2010.pdf": list(chain(range(105, 120))),
    "A2A_2011.pdf": list(chain(range(110, 132))),
    "A2A_2012.pdf": list(chain(range(118, 137))),
    "A2A_2013.pdf": list(chain(range(100, 113))),
    "A2A_2014.pdf": list(chain(range(112, 128))),
    "A2A_2015.pdf": list(chain(range(124, 144))),
    "A2A_2016.pdf": list(chain(range(10, 18), range(80, 90))),
    "A2A_2017.pdf": list(chain(range(115, 136))),
    "A2A_2018.pdf": list(chain(range(118, 138))),
    "A2A_2019.pdf": list(chain(range(127, 150))),
    "A2A_2020.pdf": list(chain(range(132, 148))),
    "A2A_2021.pdf": list(chain(range(125, 141))),
    "A2A_2022.pdf": list(chain(range(12, 34))),
    "A2A_2023.pdf": list(chain(range(120, 134))),
    "A2A_2024.pdf": list(chain(range(47, 52), range(390, 410))),

    # Edison (NYSE_EIX)
    "NYSE_EIX_2005.pdf": list(chain(range(38, 59))),
    "NYSE_EIX_2006.pdf": list(chain(range(45, 66))),
    "NYSE_EIX_2007.pdf": list(chain(range(60, 86))),
    "NYSE_EIX_2008.pdf": list(chain(range(60, 80))),
    "NYSE_EIX_2009.pdf": list(chain(range(63, 80), range(142, 152))),
    "NYSE_EIX_2010.pdf": list(chain(range(44, 54), range(95, 103))),
    "NYSE_EIX_2011.pdf": list(chain(range(41, 51), range(82, 97))),
    "NYSE_EIX_2012.pdf": list(chain(range(33, 43), range(65, 70))),
    "NYSE_EIX_2013.pdf": list(chain(range(29, 35), range(62, 74))),
    "NYSE_EIX_2014.pdf": list(chain(range(43, 59))),
    "NYSE_EIX_2015.pdf": list(chain(range(40, 56))),
    # 2016 incomplete (40–54 & ?) → second range missing
    "NYSE_EIX_2016.pdf": list(chain(range(40, 54))),
    "NYSE_EIX_2017.pdf": list(chain(range(45, 61))),
    "NYSE_EIX_2018.pdf": list(chain(range(43, 61))),
    "NYSE_EIX_2019.pdf": list(chain(range(57, 70))),
    "NYSE_EIX_2020.pdf": list(chain(range(52, 69))),
    "NYSE_EIX_2021.pdf": list(chain(range(45, 72))),
    "NYSE_EIX_2022.pdf": list(chain(range(44, 72))),
    "NYSE_EIX_2023.pdf": list(chain(range(41, 70))),
    "NYSE_EIX_2024.pdf": list(chain(range(38, 60))),

    # OMV (OTC_OMVKY)
    "OTC_OMVKY_2005.pdf": list(chain(range(25, 32))),
    "OTC_OMVKY_2006.pdf": list(chain(range(62, 67), range(115, 130))),
    "OTC_OMVKY_2007.pdf": list(chain(range(64, 71), range(118, 134))),
    "OTC_OMVKY_2008.pdf": list(chain(range(62, 73), range(115, 132))),
    "OTC_OMVKY_2009.pdf": list(chain(range(64, 71), range(119, 130))),
    "OTC_OMVKY_2010.pdf": list(chain(range(59, 66), range(116, 132))),
    "OTC_OMVKY_2011.pdf": list(chain(range(61, 73), range(123, 136))),
    "OTC_OMVKY_2012.pdf": list(chain(range(61, 68), range(125, 145))),
    "OTC_OMVKY_2013.pdf": list(chain(range(61, 70), range(127, 141))),
    "OTC_OMVKY_2014.pdf": list(chain(range(65, 80), range(140, 151))),
    "OTC_OMVKY_2015.pdf": list(chain(range(31, 41), range(119, 130))),
    "OTC_OMVKY_2016.pdf": list(chain(range(86, 95), range(189, 200))),
    "OTC_OMVKY_2017.pdf": list(chain(range(83, 114), range(190, 207))),
    "OTC_OMVKY_2018.pdf": list(chain(range(78, 88), range(188, 200))),
    "OTC_OMVKY_2019.pdf": list(chain(range(82, 98), range(192, 210))),
    "OTC_OMVKY_2020.pdf": list(chain(range(67, 83), range(170, 185))),
    "OTC_OMVKY_2021.pdf": list(chain(range(69, 86), range(181, 200))),
    "OTC_OMVKY_2022.pdf": list(chain(range(80, 92), range(193, 210))),
    "OTC_OMVKY_2023.pdf": list(chain(range(83, 99), range(193, 206))),
    "OTC_OMVKY_2024.pdf": list(chain(range(80, 156), range(344, 403))),

    # CEZ (cez-group)
    "cez-group_2005.pdf": list(chain(range(32, 41), range(100, 118))),
    "cez-group_2006.pdf": list(chain(range(69, 75), range(159, 169))),
    "cez-group_2007.pdf": list(chain(range(82, 87), range(144, 152), range(180, 192))),
    "cez-group_2008.pdf": list(chain(range(56, 63), range(162, 172))),
    "cez-group_2009.pdf": list(chain(range(78, 85), range(200, 218), range(246, 257))),
    "cez-group_2010.pdf": list(chain(range(91, 101), range(251, 261))),
    "cez-group_2011.pdf": list(chain(range(72, 80), range(240, 253))),
    "cez-group_2012.pdf": list(chain(range(89, 100), range(259, 270))),
    "cez-group_2013.pdf": list(chain(range(84, 92), range(236, 246))),
    "cez-group_2014.pdf": list(chain(range(85, 92), range(260, 271))),
    "cez-group_2015.pdf": list(chain(range(84, 90), range(258, 265))),
    "cez-group_2016.pdf": list(chain(range(77, 96), range(253, 269))),
    "cez-group_2017.pdf": list(chain(range(84, 90), range(273, 283))),
    "cez-group_2018.pdf": list(chain(range(29, 51), range(280, 291))),
    "cez-group_2019.pdf": list(chain(range(21, 40), range(280, 292))),
    "cez-group_2020.pdf": list(chain(range(27, 40), range(341, 350))),
    "cez-group_2021.pdf": list(chain(range(29, 66), range(276, 286))),
    "cez-group_2022.pdf": list(chain(range(60, 72), range(280, 292))),
    "cez-group_2023.pdf": list(chain(range(60, 86), range(280, 296))),
    "cez-group_2024.pdf": list(chain(range(297, 310), range(420, 455))),

}




# ============================================================
# 1. LOAD TEXT FROM PDF´s
# ============================================================

def load_report_paragraphs(reports_folder, page_ranges, strict=True):
    report_paragraphs = []
    report_paragraphs_source = []

    print(f"Looking for PDFs in: {reports_folder}")
    # Exact lookup: filenames in Reports/ must match keys in page_ranges
    page_ranges_exact = page_ranges
    target_files = set(page_ranges_exact.keys())
    found_targets = set()

    for path, dirs, files in os.walk(reports_folder):
        pdfs = [file for file in files if file.lower().endswith(".pdf")]
        if not pdfs:
            continue
        print("Found PDFs:", pdfs)

        for _file in pdfs:
            print(f"Processing {_file}...")
            full_path = os.path.join(path, _file)

            # Only process PDFs that are explicitly listed in page_ranges.
            # This allows the folder to contain many other PDFs without raising errors.
            file_key = _file
            if file_key not in page_ranges_exact:
                continue

            found_targets.add(file_key)
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

    if strict:
        missing = sorted(target_files - found_targets)
        if missing:
            raise ValueError(
                "The following PDFs were listed in page_ranges but were not found under reports_folder: "
                f"{missing}. Check the folder path and filenames."
            )

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
MIN_DF = 10            # token must appear in at least MIN_DF documents


# --- Bigram (phrase) settings ---
# If True, we augment the unigram token stream with frequent bigrams as extra tokens
# e.g. ["interest", "rate"] -> adds "interest_rate" (while keeping "interest" and "rate").
USE_BIGRAMS = True
BIGRAM_MIN_COUNT = 150   # bigram must appear at least this many times in the corpus

# Cache learned bigrams so we only learn + log them once per run
_CACHED_BIGRAM_SET: set[tuple[str, str]] | None = None

EXTRA_DROP_WORDS = {
    # Generic report boilerplate
    "annual", "report", "reports", "group", "plc", "page", "pages", "section", "chapter",
    "table", "tables", "figure", "figures", "statement", "statements",
    "introduction", "overview", "note", "notes", "euro",

    # Bank names / identifiers (extend as needed)
    "barclays", "seb", "ubs", "ing", "danske", "deutschebank", "deutsche", "bank",
    "bnp", "paribas", "fortis", "oppohjola", "op", "pohjola", "sweden", "poland", "norway", "dispute", "colleague",

    # Common legal entities
    "limited", "ltd", "ab", "asa", "as", "vesta", "portugal", "polska", "estonia", "lithuania", "latvia", "region", "turkey", "Hungary"
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
        # keep alphabetic tokens OR underscore-joined alphabetic bigrams (e.g. "interest_rate")
        if not all(part.isalpha() for part in t.split("_")):
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

# --- Bigram helpers ---
def _learn_frequent_bigrams(docs_tokens: list[list[str]], min_count: int) -> set[tuple[str, str]]:
    """Learn frequent bigrams from the corpus (simple frequency threshold)."""
    bigram_counts = Counter()
    for toks in docs_tokens:
        if not toks or len(toks) < 2:
            continue
        for a, b in zip(toks, toks[1:]):
            # safety: ignore any token that is already a phrase or weird
            if ("_" in a) or ("_" in b):
                continue
            bigram_counts[(a, b)] += 1
    return {bg for bg, c in bigram_counts.items() if c >= int(min_count)}


def _augment_with_bigrams(toks: list[str], bigram_set: set[tuple[str, str]]) -> list[str]:
    """Insert bigram tokens while keeping unigrams.

    Example:
      ["interest","rate","risk"] -> ["interest_rate","interest","rate","risk"]
    """
    if not toks or len(toks) < 2:
        return toks

    out: list[str] = []
    for a, b in zip(toks, toks[1:]):
        if (a, b) in bigram_set:
            out.append(f"{a}_{b}")
        out.append(a)
    out.append(toks[-1])
    return out


def _rebuild_word_freq_from_tokens(df: pd.DataFrame, tokens_col: str = "tokens") -> pd.DataFrame:
    """Ensure df['word_freq'] matches df[tokens_col] after any token transformation."""
    df = df.copy()
    df["word_freq"] = df[tokens_col].apply(lambda toks: Counter(toks) if isinstance(toks, list) else Counter())
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
    3b) Optionally learn + insert frequent bigrams (while keeping unigrams)
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

    # 3) Per-document token cleanup (unigrams)
    df[tokens_col] = df[tokens_col].apply(_basic_token_filter)

    # 3b) OPTIONAL: learn + insert frequent bigrams (keep unigrams too)
    # NOTE: We cache the learned bigrams so the message only prints once even if
    # preprocess_text_and_tokens() is called multiple times in a single run.
    if USE_BIGRAMS:
        global _CACHED_BIGRAM_SET
        if _CACHED_BIGRAM_SET is None:
            _CACHED_BIGRAM_SET = _learn_frequent_bigrams(df[tokens_col].tolist(), min_count=BIGRAM_MIN_COUNT)
            print(f"Learned {len(_CACHED_BIGRAM_SET)} bigrams with count >= {BIGRAM_MIN_COUNT}")
        df[tokens_col] = df[tokens_col].apply(lambda toks: _augment_with_bigrams(toks, _CACHED_BIGRAM_SET))

    # 4) DF-based filtering across corpus (applies to both unigrams and bigrams)
    df = _df_filter_tokens(df, tokens_col=tokens_col, min_df=min_df)

    # IMPORTANT: rebuild word_freq so later tables use counts from the final token list
    df = _rebuild_word_freq_from_tokens(df, tokens_col=tokens_col)

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
N_TABLES = 64

# Neighbor search algorithm:
# - "lsh"   : fast approximate nearest neighbors (FAISS LSH)
# - "brute" : exact brute-force neighbors (slow but deterministic)
DEFAULT_NEIGHBOR_ALG = "lsh"

def cluster_words(
    embedding_matrix: np.ndarray,
    target_cluster_size: int = 30,
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
N_TOPICS_PER_CLUSTER = 1

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
        bank, year_parsed = _parse_bank_year_from_filename(fname)

        # Compute total number of pages extracted (handles multiple ranges)
        if rng is None:
            n_pages = None
        else:
            try:
                n_pages = int(len(rng))
            except Exception:
                # rng may be an iterable of ranges or page indices
                n_pages = int(sum(len(r) for r in rng))

        n_tok_after = int(tok_after.get(fname, 0))
        status = "ok" if n_tok_after > 0 else "empty_or_missing"

        rows.append({
            "year": year_parsed,
            "run_label": year_label,
            "file": fname,
            "bank": bank,
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
    summary_path = os.path.join(out_folder, "extraction_summary_ALL_V1.csv")
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

    print("\n=== STEP 5: Cluster word embeddings (LSH sequential clustering; target_cluster_size=50) ===")
    ec, clusters, cluster_words_map, word_cluster_map = cluster_words(
        embedding_matrix,
        target_cluster_size=30,
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