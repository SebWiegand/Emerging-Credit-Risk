# =============================================================================
# EMERGING RISK PANEL – STEP-BY-STEP EXPLANATION
# =============================================================================
# WHAT THIS FILE DOES (IN PLAIN ENGLISH)
#
# This script builds a quarterly measure of "Emerging Risk" for banks.
#
# Big Picture:
# 1. We take DAILY stock returns for each bank.
# 2. For each quarter, we compute how strongly each pair of banks moves together
#    (their return covariance).
# 3. We explain that covariance using:
#       (a) simple financial controls (size, volatility, turnover, etc.)
#       (b) text-based topic exposures from annual reports
# 4. We compare:
#       - A regression with only controls
#       - A regression with controls + text exposures
# 5. The improvement in adjusted R² (ΔAdj. R²) is our Emerging Risk Index.
#
# Interpretation:
# If adding text exposures significantly improves explanatory power,
# it suggests that bank disclosures contain risk information not captured
# by traditional financial variables.
#
# Output:
# - panel_bank_quarter.csv     → quarterly bank-level controls + exposures
# - panel_pairs_quarter.csv    → bank-pair covariance dataset
# - regression_results.csv     → per-quarter regression results
# - regression_results_with_z.csv → standardized Emerging Risk Index
#
# This file is organized in clearly numbered sections.
# Each section explains what it does before the code.
# =============================================================================

# -----------------------------------------------------------------------------
# Libraries
#   pandas/numpy: data wrangling and numerical work (returns, covariances)
#   itertools: generate all bank pairs (i, j)
#   statsmodels: per-quarter OLS regressions and adjusted R²
#   matplotlib: plots
#   os/glob: filesystem paths and optional CSV auto-detection
#   json/ast/re: parsing topic-word files and building API payloads
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from itertools import combinations
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os, glob
import math
import json

# ---------------------------
# CONFIGURATION
# ---------------------------
# Input daily market data (Compustat security daily collapsed to firm-level later)
CSV_PATH = "New_Data.csv"  # daily market data (filtered / prepared for the 16 matched banks)


# Minimum number of daily returns within a quarter required to compute reliable
# volatility/covariance. Quarters with fewer observations are dropped.
MIN_DAYS_PER_QUARTER = 20  # minimum trading days per quarter to include bank in analysis

# ---------------------------
# BASELINE PERIOD (for z-score standardization)
# ---------------------------
BASELINE_START_Q = "2016Q1"
BASELINE_END_Q   = "2018Q4"
MIN_BASELINE_QUARTERS = 8

# ---------------------------
# TOPIC SELECTION (fixed across all quarters)
# ---------------------------
# We want the *same* topic set each quarter so ΔAdj.R² is comparable over time.
# We therefore select a fixed set of topics once (global), then run all quarters with that set.
#
# This also prevents the regression from trying to include hundreds of topics with only ~120 pair
# observations per quarter (16 banks => max 120 pairs), which causes df_resid=0 warnings.
USE_FIXED_TOPIC_SET = True
TOP_K_TOPICS = 60  # hard cut: use the top-K topics by importance (adjust if needed)
TOPIC_IMPORTANCES_PATH = "topic_importances.csv"  # expects columns: cluster, leading_importance

# --------------------------------------------------
# BANK UNIVERSE (manual mapping from extraction_summary_ALL bank tokens -> gvkey)
# We restrict the entire pipeline (market + text) to these banks only.
# --------------------------------------------------
TARGET_GVKEYS = [
    "15181",   # BBVA
    "14140",   # SANTANDER
    "12673",   # BARCLAYS
    "15575",   # COMMERZBANK
    "24563",   # CREDIT AGRICOLE
    "15552",   # DANSKE BANK
    "15576",   # DEUTSCHE BANK
    "15538",   # DNB
    "214659",  # ERSTE
    "15703",   # KBC
    "272817",  # RAIFFEISEN
    "15671",   # SEB (SKANDINAVISKA ENSKILDA BANK)
    "15654",   # HANDELSBANKEN
    "24578",   # SWEDBANK
    "144496",  # UBS
    "15549",   # UNICREDIT
]

# Map the extraction_summary_ALL 'bank' tokens to the target gvkeys (manual, no fuzzy matching).
BANK_TOKEN_TO_GVKEY = {
    "AMROBANK": "320764",          # ABN AMRO (NOTE: not present in New_Data.csv universe if you keep only 16)
    "BARCLAYS": "12673",
    "BBVA": "15181",
    "COMMERZBANK": "15575",
    "CREDITAGRICOLE": "24563",
    "DANSKEBANK": "15552",
    "DEUTSCHE": "15576",
    "DNB": "15538",
    "ERSTE": "214659",
    "HANDELSBANKEN": "15654",
    "IDG": "15617",                # ING (NOTE: not present in New_Data.csv universe if you keep only 16)
    "KBC": "15703",
    "NORDEA": None,                # not in New_Data.csv
    "RAIFFEISEN": "272817",
    "SANTANDER": "14140",
    "SEB": "15671",
    "SWEDBANK": "24578",
    "UBS": "144496",
    "UNICREDIT": "15549",
}

#
# --- Text-topic inputs (already extracted) ---
# This file is expected to contain firm-year topic loadings (and optionally other
# diagnostics). We will:
#   1) read the firm-year topic loadings from this file
#   2) anchor each year at Q1
#   3) carry forward exposures to each quarter using merge_asof

# NOTE: Hard-coded relative location of the extraction summary within this project.
# We DO NOT auto-search other folders: if this path is wrong, we fail loudly.
EXTRACTION_SUMMARY_PATH = os.path.join(
    "Text analytics",
    "Scripts",
    "outputs_textual_factors",
    "extraction_summary_ALL.csv",
)



#
# TEXT: BUILD BANK-YEAR TOPIC EXPOSURES FROM extraction_summary_ALL.csv
#

def build_quarterly_topic_panel_from_summary(summary_path: str) -> tuple[pd.DataFrame, list[str]]:
    """Build quarterly topic exposures from a pre-built extraction summary.

    The extraction_summary_ALL.csv is assumed to contain *firm-year* topic loadings.
    We do NOT parse PDFs here.

    Expected (robust) schema
    ------------------------
    Required:
      - year column (case-insensitive): one of ['year', 'YEAR', 'fyear', 'FYEAR', 'report_year', 'REPORT_YEAR']
      - firm identifier:
          * preferred: gvkey column (case-insensitive): ['gvkey']
          * fallback: bank/name column (case-insensitive): ['bank'] that can be matched to `conm` in the market data (CSV_PATH)

    Topic columns:
      - Any columns starting with one of these prefixes (case-insensitive):
          'topic_loading_', 'topic_', 't_'  (excluding the id columns above)
      - If none of the prefixes match, we fall back to: all numeric columns except ids.

    Returns
    -------
    (topics_fy, topic_cols)
      topics_fy: DataFrame with columns [gvkey, quarter, topic_*]
        Each row is a firm-year anchored at Q1.
      topic_cols: list of topic-loading column names.
    """
    # Resolve path strictly relative to this script (no auto-search).
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path_resolved = summary_path
    if not os.path.isabs(summary_path_resolved):
        summary_path_resolved = os.path.join(base_dir, summary_path_resolved)

    if not os.path.isfile(summary_path_resolved):
        raise FileNotFoundError(
            "[TEXT] extraction_summary_ALL.csv not found at the configured location. "
            f"Expected: {summary_path_resolved}"
        )

    print(f"[TEXT] Using extraction summary at: {summary_path_resolved}")
    s = pd.read_csv(summary_path_resolved)
    # Normalize column names for detection
    cols = list(s.columns)
    lower = {c.lower(): c for c in cols}

    # Identify year column
    year_col = None
    for cand in ["year", "fyear", "report_year"]:
        if cand in lower:
            year_col = lower[cand]
            break

    if year_col is None:
        print("[TEXT] extraction_summary_ALL.csv is missing the required year column.")
        print(f"[TEXT] Need one of: year/fyear/report_year. Found columns: {cols[:60]}")
        return pd.DataFrame(), []

    # Preferred: gvkey column
    gvkey_col = lower.get("gvkey")

    # Fallback: derive gvkey from the extraction 'bank' token using the manual map above.
    # This avoids fuzzy matching and guarantees consistent identifiers.
    if gvkey_col is None:
        bank_col = lower.get("bank")
        if bank_col is None:
            print("[TEXT] extraction_summary_ALL.csv is missing gvkey and also has no 'bank' column for name matching.")
            print(f"[TEXT] Found columns: {cols[:60]}")
            return pd.DataFrame(), []

        s["bank_clean"] = s[bank_col].astype(str).str.upper().str.strip()
        s["gvkey"] = s["bank_clean"].map(BANK_TOKEN_TO_GVKEY)

        miss = s["gvkey"].isna().mean()
        print(f"[TEXT] Derived gvkey via manual BANK_TOKEN_TO_GVKEY map. Missing gvkey share: {miss:.2%}")
        if miss > 0:
            examples = (
                s.loc[s["gvkey"].isna(), bank_col]
                .astype(str)
                .value_counts()
                .head(30)
            )
            if len(examples) > 0:
                print("[TEXT] Top unmatched bank tokens (first 30):")
                print(examples)

        gvkey_col = "gvkey"

    # Clean types
    s[gvkey_col] = s[gvkey_col].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})
    s[year_col] = pd.to_numeric(s[year_col], errors="coerce").astype("Int64")
    s = s.dropna(subset=[year_col, gvkey_col]).copy()
    s[year_col] = s[year_col].astype(int)

    # Detect topic columns
    id_cols = {gvkey_col, year_col}
    # Exclude common non-topic metadata columns if present
    for extra in [
        "run_label", "file", "bank", "pages_from", "pages_to", "n_pages", "n_paragraphs",
        "n_tokens_before", "n_tokens_after", "status", "bank_clean", "conm_clean"
    ]:
        if extra in lower:
            id_cols.add(lower[extra])
    prefixes = ("topic_loading_", "topic_", "t_")

    topic_cols = [
        c for c in cols
        if c not in id_cols and isinstance(c, str) and c.lower().startswith(prefixes)
    ]

    # Fallback: all numeric columns except ids
    if not topic_cols:
        numeric_cols = [c for c in cols if c not in id_cols and pd.api.types.is_numeric_dtype(s[c])]
        topic_cols = numeric_cols

    if not topic_cols:
        print("[TEXT] No topic loading columns detected in extraction summary.")
        return pd.DataFrame(), []

    # Keep only the firm-year topic matrix and average duplicates if any
    fy = (
        s[[gvkey_col, year_col] + topic_cols]
        .groupby([gvkey_col, year_col], as_index=False)[topic_cols]
        .mean()
    )

    # Standardize column names used downstream
    fy = fy.rename(columns={gvkey_col: "gvkey", year_col: "year"})

    # Anchor report year at Q1 and carry forward later via merge_asof
    fy["quarter"] = pd.PeriodIndex(fy["year"].astype(str) + "Q1", freq="Q")
    fy = fy.drop(columns=["year"]).sort_values(["gvkey", "quarter"]).reset_index(drop=True)

    # Optionally apply a fixed topic set (top-K by importance) so the same topics are used every quarter.
    if USE_FIXED_TOPIC_SET:
        imp_path = TOPIC_IMPORTANCES_PATH
        # Resolve path relative to this script if not absolute
        if not os.path.isabs(imp_path):
            imp_path = os.path.join(base_dir, imp_path)

        if not os.path.isfile(imp_path):
            raise FileNotFoundError(
                f"[TEXT] USE_FIXED_TOPIC_SET=True but topic_importances file not found: {imp_path}"
            )

        imp = pd.read_csv(imp_path)
        imp_cols = {c.lower(): c for c in imp.columns}
        if "cluster" not in imp_cols or "leading_importance" not in imp_cols:
            raise ValueError(
                "[TEXT] topic_importances.csv must contain columns 'cluster' and 'leading_importance'. "
                f"Found: {list(imp.columns)}"
            )

        cluster_col = imp_cols["cluster"]
        li_col = imp_cols["leading_importance"]

        # Rank clusters by importance (descending)
        imp = imp.copy()
        imp[li_col] = pd.to_numeric(imp[li_col], errors="coerce")
        imp = imp.dropna(subset=[cluster_col, li_col])
        imp = imp.sort_values(li_col, ascending=False)

        # Your exposure columns are named like "topic_loading_9".
        ranked_topic_names = [f"topic_loading_{int(x)}" for x in imp[cluster_col].tolist()]

        # Keep only topics that exist in the extraction summary
        ranked_topic_names = [t for t in ranked_topic_names if t in topic_cols]

        if not ranked_topic_names:
            raise ValueError(
                "[TEXT] Fixed-topic selection produced an empty topic list. "
                "Check that topic_importances.csv clusters match exposure column names like 'topic_loading_<id>'."
            )

        # Hard cut to TOP_K_TOPICS
        selected = ranked_topic_names[: int(TOP_K_TOPICS)]

        # Restrict both the output matrix and the returned topic column list
        fy = fy[["gvkey", "quarter"] + selected]
        topic_cols = selected

        print(
            f"[TEXT] Using fixed topic set: top {len(topic_cols)} topics from {os.path.basename(imp_path)} "
            f"(TOP_K_TOPICS={TOP_K_TOPICS})."
        )

    return fy, topic_cols

# Output of this section: raw daily issue-level data (df) with correct dtypes.
# [gvkey, quarter(=YYYYQ1), topic_1, topic_2, ...]

#
# ------------------------------------------------------------------
# SECTION 1: LOAD DAILY MARKET DATA
# ------------------------------------------------------------------
# We load daily stock market data for a fixed set of banks.
# Each row represents one bank on one trading day.
# We restrict the dataset to our chosen bank universe.
# ------------------------------------------------------------------
# ---------------------------
# 1) LOAD DATA
# ---------------------------
df = pd.read_csv(CSV_PATH) #load data
# Restrict the market data to the target gvkeys (16 banks)
df["gvkey"] = df["gvkey"].astype(str)
df = df[df["gvkey"].isin(TARGET_GVKEYS)].copy()
print(f"[MKT] Using {df['gvkey'].nunique()} gvkeys from New_Data.csv (target={len(TARGET_GVKEYS)}).")
df['datadate'] = pd.to_datetime(df['datadate']) #ensure date is datetime
for col in ['cshoc','cshtrd','prccd','prchd','prcld']: #ensure numeric types
    if col in df.columns: #check column exists
        df[col] = pd.to_numeric(df[col], errors='coerce') #convert to numeric, coerce errors to NaN

#
#
# ------------------------------------------------------------------
# SECTION 2: CLEAN AND COLLAPSE TO ONE PRICE PER BANK PER DAY
# ------------------------------------------------------------------
# Some banks have multiple listed share classes.
# We collapse them into one firm-level daily price series.
# Output: one row per (bank, date).
# ------------------------------------------------------------------
# ---------------------------
# 2) COLLAPSE TO FIRM-LEVEL DAILY PRICE  (prirow is the IID, e.g., "04W")
# ---------------------------
def collapse_firm_daily(group):
    # Try to match prirow (a string like "04W") to iid
    primary_iid = None
    if 'prirow' in group.columns and group['prirow'].notna().any():
        # most common non-null prirow for this gvkey-date (usually one)
        s = group['prirow'].dropna().astype(str).str.strip() #For the gvkey–datadate group, it reads the primary issue code (e.g., "04W") from prirow. If multiple rows disagree (rare), it takes the mode (most frequent).
        if not s.empty:
            primary_iid = s.mode().iloc[0]

    # If we have a primary iid, select the matching issue by iid
    if primary_iid is not None and 'iid' in group.columns:
        g = group[group['iid'].astype(str).str.strip() == str(primary_iid).strip()] #It then filters the group to only include rows where the iid matches this primary issue code.
        if not g.empty:
            # if multiple rows, prefer the largest cshoc (largest number of shares outstanding)
            g = g.sort_values('cshoc', ascending=False).head(1)
            return pd.Series({
                'prccd_firm': g['prccd'].iloc[0],
                'cshoc_firm': g['cshoc'].iloc[0],
                'cshtrd_firm': g['cshtrd'].iloc[0],
                'gsector': g['gsector'].iloc[0],
                'loc': g['loc'].iloc[0],
                'curcdd': g['curcdd'].iloc[0]
            })

    # Fallback if we cannot confidently identify the primary issue:
    # We collapse multiple listed issues into a single firm-level series by taking
    # a market-cap-weighted average price using shares outstanding (cshoc) as weights.
    g = group.copy()
    w = g['cshoc'].replace(0, np.nan)
    if w.notna().any() and w.sum() > 0:
        price = (g['prccd'] * w).sum() / w.sum()
    else:
        price = g['prccd'].mean()

    # mode helpers for categorical columns
    def mode_or_first(s): #returns the mode (most common value) of a Series, or first value if no mode
        s = s.dropna() #drop NaNs
        return s.mode().iloc[0] if not s.mode().empty else (s.iloc[0] if not s.empty else np.nan) #return mode or first value

    return pd.Series({ #returns a Series with firm-level daily data
        'prccd_firm': price, #weighted average price
        'cshoc_firm': g['cshoc'].mean(), #average shares outstanding
        'cshtrd_firm': g['cshtrd'].mean(), #average shares traded
        'gsector': mode_or_first(g['gsector']), #most common sector
        'loc': mode_or_first(g['loc']), #most common location
        'curcdd': mode_or_first(g['curcdd']) #most common currency
    })

#Yields one row per firm per date
collapsed = (
    df.groupby(['gvkey','datadate'], group_keys=False, sort=False) #group by firm and date
      .apply(collapse_firm_daily, include_groups=False)    #apply collapse_firm_daily function
      .reset_index()                 #keep gvkey/datadate as normal columns
)
#Sanity check

m = df['prirow'].notna() & (df['iid'].astype(str).str.strip() == df['prirow'].astype(str).str.strip()) #boolean Series where prirow matches iid
mismatch_rate = 1 - m.mean() #compute share of rows where prirow matches iid
print("Share of rows where prirow != iid (expected; prirow is sparse / issue-level):", mismatch_rate)
# Note: a high mismatch rate is expected because prirow is often missing and is an issue-level attribute.

#
# ------------------------------------------------------------------
# SECTION 3: COMPUTE DAILY RETURNS AND ASSIGN QUARTERS
# ------------------------------------------------------------------
# We compute log returns from daily prices.
# We also assign each day to a calendar quarter.
# These returns are later used to compute covariance.
# ------------------------------------------------------------------
# ---------------------------
# 3) COMPUTE RETURNS & QUARTERS
# ---------------------------
collapsed = collapsed.sort_values(['gvkey','datadate']).reset_index(drop=True) #sort by gvkey and date

# Make sure ids are simple types
collapsed['gvkey'] = collapsed['gvkey'].astype(str) #ensure gvkey is string

# Sort first
collapsed = collapsed.sort_values(['gvkey','datadate']).reset_index(drop=True) #sort by gvkey and date

# Compute returns in an index-preserving way
collapsed['log_price'] = np.log(collapsed['prccd_firm']) #compute log price
collapsed['ret'] = collapsed.groupby('gvkey', sort=False)['log_price'].diff() #compute log returns per gvkey

# (optional) drop helper
collapsed = collapsed.drop(columns='log_price') #drop log_price column

collapsed['quarter'] = collapsed['datadate'].dt.to_period('Q') #compute quarter identifier
collapsed['mktcap'] = collapsed['prccd_firm'] * collapsed['cshoc_firm'] #compute market cap (used later for size calculation)

#
# ------------------------------------------------------------------
# SECTION 4: BUILD QUARTERLY BANK CHARACTERISTICS
# ------------------------------------------------------------------
# We aggregate daily data into quarterly variables:
# - Size (log market cap)
# - Turnover
# - Volatility
# These act as standard financial controls in the regression.
# ------------------------------------------------------------------
# ---------------------------
# 4) QUARTERLY BANK CONTROLS
# ---------------------------
bank_quarter = (
    collapsed.groupby(['gvkey','quarter']).agg( #aggregate to quarterly level
        avg_price=('prccd_firm','mean'), #average daily price in the quarter
        avg_mktcap=('mktcap','mean'), #average daily market cap in the quarter
        turnover=('cshtrd_firm','mean'), #average shares traded
        shares=('cshoc_firm','mean'), #average shares outstanding
        vol=('ret', lambda x: np.nan if x.count() < 2 else np.nanstd(x, ddof=1)), #volatility of daily returns
        gsector=('gsector', lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]), #most common sector
        loc=('loc', lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]), #most common HQ country
        curcdd=('curcdd', lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]), #most common currency
        n_days=('ret','count') #number of trading days with returns
    )
    .reset_index()
)
bank_quarter = bank_quarter[bank_quarter['n_days'] >= MIN_DAYS_PER_QUARTER].copy() #keep only quarters with enough trading days
bank_quarter['size'] = np.log(bank_quarter['avg_mktcap'].replace({0: np.nan})) #log size (market cap).
bank_quarter['turnover'] = (bank_quarter['turnover'] / bank_quarter['shares']).replace([np.inf,-np.inf], np.nan) #turnover ratio

#
#
# ------------------------------------------------------------------
# SECTION 5: ADD TEXT-BASED TOPIC EXPOSURES
# ------------------------------------------------------------------
# We merge annual report topic loadings to each bank.
# Each topic measures how strongly a bank discusses a specific risk theme.
# We lag exposures by one quarter to avoid look-ahead bias.
# ------------------------------------------------------------------
# ---------------------------
# 5) EXPOSURES: DISCLOSURE-BASED TOPIC LOADINGS (REQUIRED)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load firm-year topic loadings from the prebuilt extraction summary
topics_fy, topic_cols = build_quarterly_topic_panel_from_summary(EXTRACTION_SUMMARY_PATH)

# Restrict text exposures to the same gvkey universe as the market data
topics_fy["gvkey"] = topics_fy["gvkey"].astype(str)
topics_fy = topics_fy[topics_fy["gvkey"].isin(TARGET_GVKEYS)].copy()
print(f"[TEXT] Topic exposures available for {topics_fy['gvkey'].nunique()} gvkeys after filtering to target universe.")

# Ensure types for merge
bank_quarter["gvkey"] = bank_quarter["gvkey"].astype(str)

if topics_fy.empty:
    raise ValueError(
        "[TEXT] Topic panel is empty. Ensure extraction_summary_ALL.csv contains a year column and topic_loading_* columns, "
        "and either a gvkey column OR a bank column that can be matched to `conm` in the market data (CSV_PATH)."
    )

# (topics_fy["gvkey"] = topics_fy["gvkey"].astype(str))  # removed, handled above

# Ensure topic columns are numeric and fill missing as 0 (e.g., early quarters before first disclosure)
for c in topic_cols:
    if c in topics_fy.columns:
        topics_fy[c] = pd.to_numeric(topics_fy[c], errors="coerce")

# merge_asof requires the 'on' key to be globally sorted.
# Sort primarily by the timestamp key, then by gvkey.
bank_quarter = bank_quarter.copy()
topics_fy = topics_fy.copy()

bank_quarter["_qts"] = bank_quarter["quarter"].dt.to_timestamp()
topics_fy["_qts"] = topics_fy["quarter"].dt.to_timestamp()

bank_quarter = bank_quarter.dropna(subset=["_qts"]).sort_values(["_qts", "gvkey"]).reset_index(drop=True)
topics_fy = topics_fy.dropna(subset=["_qts"]).sort_values(["_qts", "gvkey"]).reset_index(drop=True)

if not bank_quarter["_qts"].is_monotonic_increasing:
    raise ValueError("bank_quarter _qts is not globally sorted ascending; merge_asof requires this")
if not topics_fy["_qts"].is_monotonic_increasing:
    raise ValueError("topics_fy _qts is not globally sorted ascending; merge_asof requires this")

# Economic interpretation: each quarter uses the most recent available annual
# disclosure topics for that bank (carried forward until the next report).
bank_quarter = pd.merge_asof(
    bank_quarter,
    topics_fy.drop(columns=["quarter"]),
    by="gvkey",
    on="_qts",
    direction="backward",
)

# Restore original quarter column and drop helper
bank_quarter = bank_quarter.rename(columns={"_qts": "quarter_ts"})

# If any topic columns are still missing (e.g., early quarters before first disclosure), fill with 0
for c in topic_cols:
    if c in bank_quarter.columns:
        bank_quarter[c] = pd.to_numeric(bank_quarter[c], errors="coerce").fillna(0.0)
    else:
        # If a topic column wasn't merged (unexpected), create it as 0 to keep the regression design consistent.
        bank_quarter[c] = 0.0

print(f"Merged {len(topic_cols)} topic exposure columns from disclosures.")
exposure_cols = topic_cols  # already fixed (if USE_FIXED_TOPIC_SET=True)

# Lag by one quarter
def lag_by_quarter(df_bq):
    # Vectorized lag creation to avoid pandas DataFrame fragmentation warnings
    df_bq = df_bq.sort_values(['gvkey', 'quarter']).copy()
    lag_cols = ['size', 'turnover', 'vol'] + exposure_cols

    # Compute all lags at once
    lagged = df_bq.groupby('gvkey', sort=False)[lag_cols].shift(1)
    lagged = lagged.add_suffix('_lag1')

    # Attach in one concat (fast, non-fragmenting)
    df_bq = pd.concat([df_bq, lagged], axis=1)

    # Optional: ensure a compact memory layout
    return df_bq.copy()

bank_quarter = lag_by_quarter(bank_quarter)
# Drop helper timestamp column if present
if "quarter_ts" in bank_quarter.columns:
    bank_quarter = bank_quarter.drop(columns=["quarter_ts"])
bank_quarter.to_csv("panel_bank_quarter.csv", index=False)

#
# ------------------------------------------------------------------
# SECTION 6: COMPUTE BANK-PAIR COVARIANCE
# ------------------------------------------------------------------
# For each quarter:
# - We compute covariance of daily returns between every pair of banks.
# Output: one row per (bank i, bank j, quarter).
# This is the dependent variable in the regression.
# ------------------------------------------------------------------
# ---------------------------
# 6) PAIRWISE COVARIANCES
# ---------------------------
daily_q = collapsed[['gvkey','datadate','quarter','ret']].dropna().copy() #keep only relevant columns
pair_rows = [] #list to hold pairwise covariance rows
for q, qdf in daily_q.groupby('quarter'): #for each quarter
    counts = qdf.groupby('gvkey')['ret'].count() #count trading days per bank
    valid_g = counts[counts >= MIN_DAYS_PER_QUARTER].index.tolist() #banks with enough trading days
    qdf = qdf[qdf['gvkey'].isin(valid_g)] #filter to valid banks
    pivot = qdf.pivot_table(index='datadate', columns='gvkey', values='ret', aggfunc='first') #pivot to wide format
    pivot = pivot.dropna(axis=1, thresh=MIN_DAYS_PER_QUARTER) #drop banks with insufficient data
    gvkeys = list(pivot.columns) #list of valid gvkeys
    if len(gvkeys) < 2:
        continue #skip quarters with less than 2 banks
    for i,j in combinations(gvkeys,2): #for each bank pair
        rij = pivot[[i,j]].dropna() #drop rows with NaNs for either bank
        if len(rij) < MIN_DAYS_PER_QUARTER: continue #skip pairs with insufficient overlapping data
        cov_ij = np.cov(rij[i], rij[j], ddof=1)[0,1] #compute covariance
        pair_rows.append({'quarter':q, 'gvkey_i':i, 'gvkey_j':j, 'cov_ij':cov_ij}) #store result
pairs = pd.DataFrame(pair_rows) #create DataFrame from pairwise covariance rows

#
# ------------------------------------------------------------------
# SECTION 7: BUILD REGRESSION DATASET
# ------------------------------------------------------------------
# We merge lagged controls and exposures into the bank-pair dataset.
# We create:
# - Products of topic exposures (shared risk intensity)
# - Products of controls
# - Same-sector / same-country indicators
# Output: regression-ready pair-quarter panel.
# ------------------------------------------------------------------
# ---------------------------
# 7) MERGE CONTROLS, BUILD REGR DATA
# ---------------------------
# NOTE ON TRANSFORMATIONS:
# - For topic exposures we use PRODUCTS (exposure_i × exposure_j)
#   because we want to measure shared intensity of the same risk theme.
# - For controls we also use products for consistency in this implementation.
#   (Alternative specifications could use absolute differences instead.)

lag_cols = [c for c in bank_quarter.columns if c.endswith('_lag1')] #columns with lagged controls
id_cols = ['gvkey','quarter','gsector','loc','curcdd'] #identifier columns
bq_lag = bank_quarter[id_cols + lag_cols].copy() #bank-quarter lagged controls
bq_i = bq_lag.rename(columns={'gvkey':'gvkey_i'}) #rename for merging
bq_j = bq_lag.rename(columns={'gvkey':'gvkey_j'}) #rename for merging
pairs = pairs.merge(bq_i, on=['gvkey_i','quarter'], how='left') #merge bank i controls
pairs = pairs.merge(bq_j, on=['gvkey_j','quarter'], how='left', suffixes=('_i','_j')) #merge bank j controls

# Build new columns in one go to avoid pandas fragmentation warnings
new_cols = {}

# Products of lagged controls
for base in ['size', 'turnover', 'vol']:
    new_cols[f'{base}_prod_lag1'] = pairs[f'{base}_lag1_i'] * pairs[f'{base}_lag1_j']

# Products of lagged topic exposures
for base in exposure_cols:
    new_cols[f'{base}_prod_lag1'] = pairs[f'{base}_lag1_i'] * pairs[f'{base}_lag1_j']

# Same-* indicators
new_cols['same_sector'] = (pairs['gsector_i'] == pairs['gsector_j']).astype(int)
new_cols['same_country'] = (pairs['loc_i'] == pairs['loc_j']).astype(int)
new_cols['same_currency'] = (pairs['curcdd_i'] == pairs['curcdd_j']).astype(int)

pairs = pd.concat([pairs, pd.DataFrame(new_cols, index=pairs.index)], axis=1)
pairs = pairs.dropna(subset=['cov_ij']+[c for c in pairs.columns if c.endswith('_lag1')]) #drop rows with missing data
pairs = pairs.copy()  # defragment for downstream performance
pairs.to_csv("panel_pairs_quarter.csv", index=False) #save pairs-quarter panel


#
# ------------------------------------------------------------------
# SECTION 8: RUN QUARTERLY REGRESSIONS
# ------------------------------------------------------------------
# What happens here?
#
# For EACH quarter separately:
#
#   Model 1 (Baseline):
#       Covariance ~ Financial Controls
#
#   Model 2 (Full Model):
#       Covariance ~ Financial Controls + Topic Exposures
#
# The Emerging Risk Index for that quarter is:
#
#       ΔAdj.R² = Adj.R²(Full) − Adj.R²(Controls)
#
# Interpretation:
# If ΔAdj.R² is large and positive, topic exposures explain
# additional co-movement between banks beyond fundamentals.
# ------------------------------------------------------------------

results = []
topic_coef_rows = []
topic_contrib_rows = []

# --- Define regression variables clearly ---
control_vars = [
    'size_prod_lag1',
    'turnover_prod_lag1',
    'vol_prod_lag1',
    'same_sector',
    'same_country',
    'same_currency'
]

# Topic exposure variables (already built as products in Section 7)
exposure_vars = [f'{c}_prod_lag1' for c in exposure_cols]

# Sanity check: avoid too many regressors relative to pair observations
max_pairs_per_q = (len(TARGET_GVKEYS) * (len(TARGET_GVKEYS) - 1)) // 2
n_params_full = 1 + len(control_vars) + len(exposure_vars)

if USE_FIXED_TOPIC_SET and n_params_full >= max_pairs_per_q:
    print(
        f"[WARN] Full model has {n_params_full} parameters but max pairs per quarter is {max_pairs_per_q}. "
        f"Consider lowering TOP_K_TOPICS (currently {TOP_K_TOPICS})."
    )

# ------------------------------------------------------------------
# Loop over quarters
# ------------------------------------------------------------------
for q, qdf in pairs.groupby('quarter'):

    # =========================
    # Model 1: Controls Only
    # =========================
    X_controls = sm.add_constant(qdf[control_vars], has_constant='add')
    y = qdf['cov_ij']

    try:
        model_controls = sm.OLS(y, X_controls).fit()
        adjR2_controls = model_controls.rsquared_adj
    except Exception:
        adjR2_controls = np.nan

    # =========================
    # Model 2: Controls + Topics
    # =========================
    X_full = sm.add_constant(qdf[control_vars + exposure_vars], has_constant='add')

    try:
        model_full = sm.OLS(y, X_full).fit()
        adjR2_full = model_full.rsquared_adj

        # ---------------------------------------
        # Store topic-level coefficient results
        # ---------------------------------------
        for var in exposure_vars:
            topic_coef_rows.append({
                'quarter': str(q),
                'topic': var.replace('_prod_lag1', ''),
                'beta': float(model_full.params.get(var, np.nan)),
                't': float(model_full.tvalues.get(var, np.nan)),
                'p': float(model_full.pvalues.get(var, np.nan)),
            })

        # ---------------------------------------
        # Compute simple contribution importance
        # ---------------------------------------
        betas = (
            model_full.params
            .filter(like='_prod_lag1')
            .reindex(exposure_vars)
            .fillna(0.0)
        )

        topic_matrix = qdf[exposure_vars]
        contribution = topic_matrix.mul(betas, axis=1)

        importance = contribution.abs().mean(axis=0)

        for var, value in importance.items():
            topic_contrib_rows.append({
                'quarter': str(q),
                'topic': var.replace('_prod_lag1', ''),
                'mean_abs_contrib': float(value),
            })

    except Exception:
        adjR2_full = np.nan

    # =========================
    # Store Quarterly Results
    # =========================
    results.append({
        'quarter': str(q),
        'adjR2_controls': adjR2_controls,
        'adjR2_full': adjR2_full,
        'delta_adjR2': (
            adjR2_full - adjR2_controls
            if pd.notna(adjR2_full) and pd.notna(adjR2_controls)
            else np.nan
        )
    })

# Build final quarterly results table
res_df = pd.DataFrame(results).sort_values('quarter')

#
# ------------------------------------------------------------------
# SECTION 9: STANDARDIZE AND PLOT EMERGING RISK INDEX
# ------------------------------------------------------------------
# We convert ΔAdj. R² into a z-score relative to a baseline period.
# This makes spikes easier to interpret.
# ------------------------------------------------------------------
# ---------------------------
# 9) PLOT EMERGING RISK INDEX (z-score of ΔAdj. R²)
# ---------------------------
plt.figure(figsize=(9,4.5)) #create figure
plt.plot(pd.PeriodIndex(res_df['quarter'], freq='Q').to_timestamp(), res_df['adjR2_controls'], label='Controls only') #plot adjusted R² for controls only
plt.plot(pd.PeriodIndex(res_df['quarter'], freq='Q').to_timestamp(), res_df['adjR2_full'], label='Controls + exposures') #plot adjusted R² for full model
plt.title("Quarterly Adjusted R² (Controls vs +Exposures)") #set title
plt.xlabel("Quarter") #set x-axis label
plt.ylabel("Adjusted R²") #set y-axis label
plt.legend() #add legend
plt.tight_layout() #adjust layout
plt.savefig("emerging_risk_index.png") #save figure
print("All outputs saved: panel_bank_quarter.csv, panel_pairs_quarter.csv, regression_results.csv, emerging_risk_index.png") #notify user of saved outputs


# ---------------------------
# BASELINE STANDARDIZATION (z-score of ΔAdj.R²)
# ---------------------------
# We standardize ΔAdj.R² relative to a baseline period (default 1998Q1–2003Q4)
# to make spikes interpretable as deviations from a "normal" regime.
baseline_mask = (res_df["quarter"] >= BASELINE_START_Q) & (res_df["quarter"] <= BASELINE_END_Q)
baseline = res_df.loc[baseline_mask].copy()
base_vals = baseline["delta_adjR2"].dropna()

print(
    f"[BASELINE] Using baseline {BASELINE_START_Q}–{BASELINE_END_Q}: "
    f"{len(base_vals)} quarters with non-missing ΔAdj.R²."
)

# Guard against too few baseline points or zero std
if len(base_vals) >= MIN_BASELINE_QUARTERS and base_vals.std(ddof=1) > 0 and not np.isnan(base_vals.std(ddof=1)):
    mu = base_vals.mean()
    sigma = base_vals.std(ddof=1)
else:
    # If the baseline is missing/too short (e.g., market data starts later), fall back to full sample.
    # With your updated market data starting in 1998, you should normally NOT hit this branch.
    sample_vals = res_df["delta_adjR2"].dropna()
    mu = sample_vals.mean() if len(sample_vals) else np.nan
    sigma = sample_vals.std(ddof=1) if len(sample_vals) else np.nan
    print(
        "[BASELINE][WARN] Baseline period had too few usable quarters to compute mean/std. "
        "Falling back to full-sample standardization."
    )

# Compute z-score
res_df["z_score"] = (res_df["delta_adjR2"] - mu) / sigma

# Save an extended results file (keeps the original file unchanged)
res_df.to_csv("regression_results_with_z.csv", index=False) #save extended results with z-score
print(f"[BASELINE] z-score computed with mu={mu:.6g}, sigma={sigma:.6g}")

# --- Identify and save the top topic drivers for the peak quarter (max z-score) ---
# Restrict peak quarter search to 2023Q1–2024Q4 for interpretation, fallback to global max if empty.
try:
    # Restrict to quarters between 2023Q1 and 2024Q4 (inclusive)
    peak_window_mask = (res_df["quarter"] >= "2023Q1") & (res_df["quarter"] <= "2024Q4")
    res_df_peak_window = res_df.loc[peak_window_mask]
    if not res_df_peak_window.empty and res_df_peak_window["z_score"].notna().any():
        # Pick the max z-score quarter within the window
        peak_q = res_df_peak_window.loc[res_df_peak_window['z_score'].idxmax(), 'quarter']
        # Comment: Restricting the peak quarter to 2023Q1–2024Q4 focuses interpretation on recent risk episodes.
    else:
        # Fallback to global maximum if no quarters in window
        peak_q = res_df.loc[res_df['z_score'].idxmax(), 'quarter']
except Exception:
    peak_q = None

if peak_q is not None and topic_contrib_rows:
    contrib_df = pd.DataFrame(topic_contrib_rows)

    # Keep only the peak quarter
    peak_df = (
        contrib_df[contrib_df['quarter'] == str(peak_q)]
        .sort_values('mean_abs_contrib', ascending=False)
        .reset_index(drop=True)
    )

    # Compute contribution shares
    total_contrib = peak_df['mean_abs_contrib'].sum()
    if total_contrib > 0:
        peak_df['share'] = peak_df['mean_abs_contrib'] / total_contrib
    else:
        peak_df['share'] = 0.0

    peak_df['cum_share'] = peak_df['share'].cumsum()

    # Recommended cutoff: smallest set explaining 80% of total contribution
    peak_top80 = peak_df[peak_df['cum_share'] <= 0.80].copy()

    # Save outputs
    peak_df.to_csv('topic_drivers_peak_quarter_full.csv', index=False)
    peak_top80.to_csv('topic_drivers_peak_quarter_top80.csv', index=False)

    print(
        f"Saved topic drivers for peak quarter {peak_q}: "
        f"full ranking -> topic_drivers_peak_quarter_full.csv; "
        f"top 80% contributors -> topic_drivers_peak_quarter_top80.csv"
    )

# Output of this section: labeled topic-driver tables for the peak quarter.
# First we label with top words; optionally we call OpenAI to propose semantic
# labels (e.g., "Funding stress", "Climate transition risk").
# --------------------------------------------------
# 10) LABEL TOP-80 TOPICS USING TOPIC WORDS FILE
# --------------------------------------------------
# import ast
#
# # Path to topic-words file (adjust if needed)
# TOPIC_WORDS_PATH = os.path.join("outputs_textual_factors", "topics_words.csv")
#
# if os.path.isfile(TOPIC_WORDS_PATH) and os.path.isfile("topic_drivers_peak_quarter_top80.csv"):
#     # Load topic words
#     topics_words = pd.read_csv(TOPIC_WORDS_PATH)
#
#     # Detect topic-words format and build labels
#     topic_label_map = {}
#
#     # Case A: your file format (topic, topic_distribution)
#     if set(['topic', 'topic_distribution']).issubset(topics_words.columns):
#         import ast
#         import re
#
#         def parse_np_float_dict(s: str) -> dict:
#             """Parse strings like "{'word': np.float64(0.12), ...}" into {word: float}."""
#             if pd.isna(s):
#                 return {}
#             txt = str(s)
#             # Convert np.float64(x) -> x
#             txt = re.sub(r"np\.float64\(([^\)]+)\)", r"\1", txt)
#             # Also handle possible numpy.float64
#             txt = re.sub(r"numpy\.float64\(([^\)]+)\)", r"\1", txt)
#             try:
#                 d = ast.literal_eval(txt)
#             except Exception:
#                 return {}
#             # Coerce values to float
#             out = {}
#             for k, v in d.items():
#                 try:
#                     out[str(k)] = float(v)
#                 except Exception:
#                     continue
#             return out
#
#         def make_label_from_dist(dist: dict, k: int = 3) -> str:
#             if not dist:
#                 return None
#             top = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:k]
#             return ", ".join([w for w, _ in top])
#
#         for _, r in topics_words.iterrows():
#             tid = str(r['topic'])
#             dist = parse_np_float_dict(r['topic_distribution'])
#             topic_label_map[tid] = make_label_from_dist(dist, k=3)
#
#     # Case B: long format (topic, word[, weight])
#     elif set(['topic', 'word']).issubset(topics_words.columns):
#         weight_col = 'weight' if 'weight' in topics_words.columns else None
#         for tid, g in topics_words.groupby('topic'):
#             if weight_col:
#                 g = g.sort_values(weight_col, ascending=False)
#             top_words = g['word'].astype(str).head(3).tolist()
#             topic_label_map[str(tid)] = ', '.join(top_words)
#
#     # Case C: wide format with stringified dict (topic_id, words)
#     elif 'words' in topics_words.columns:
#         import ast
#
#         id_col = 'topic_id' if 'topic_id' in topics_words.columns else ('topic' if 'topic' in topics_words.columns else None)
#         if id_col is None:
#             raise ValueError("topics_words.csv has 'words' but no 'topic_id'/'topic' column")
#
#         def safe_eval_dict(x):
#             try:
#                 return ast.literal_eval(x)
#             except Exception:
#                 return {}
#
#         topics_words['words_dict'] = topics_words['words'].apply(safe_eval_dict)
#
#         def make_label(word_dict, k=3):
#             if not isinstance(word_dict, dict) or not word_dict:
#                 return None
#             top_words = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)[:k]
#             return ', '.join([w for w, _ in top_words])
#
#         for _, r in topics_words.iterrows():
#             topic_label_map[str(r[id_col])] = make_label(r['words_dict'], k=3)
#
#     else:
#         raise ValueError(
#             "Unrecognized topics_words.csv format. Expected either: "
#             "(topic, topic_distribution) OR (topic, word[, weight]) OR (topic_id/topic, words)."
#         )
#
#     # Load top-80 contributing topics
#     top80 = pd.read_csv("topic_drivers_peak_quarter_top80.csv")
#
#     # Our drivers file uses values like 'topic_loading_159' in column 'topic'.
#     # The topic-words file typically uses numeric ids like '159'.
#     top80['topic_id'] = top80['topic'].astype(str)
#     top80['topic_num'] = top80['topic_id'].apply(lambda s: s.split('_')[-1] if '_' in s else s)
#
#     # Map numeric topic ids to word-based labels
#     top80['topic_label'] = top80['topic_num'].map(topic_label_map)
#
#     # Save labeled output
#     top80.to_csv(
#         "topic_drivers_peak_quarter_top80_labeled.csv",
#         index=False
#     )
#
#     print("Saved labeled top-80 topics -> topic_drivers_peak_quarter_top80_labeled.csv")
#
#     # --------------------------------------------------
#     # Optional: Use OpenAI API to generate semantic topic names
#     # --------------------------------------------------
#     # Requires: `pip install openai` and OPENAI_API_KEY set in your environment.
#     # Produces: topic_drivers_peak_quarter_top80_semantic.csv
#
#     OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
#     SEMANTIC_OUT = "topic_drivers_peak_quarter_top80_semantic.csv"
#
#     def _top_words_for_topic_num(topic_num: str, k: int = 12):
#         # Prefer the full distribution (your topics_words.csv format)
#         if set(['topic', 'topic_distribution']).issubset(topics_words.columns):
#             row = topics_words[topics_words['topic'].astype(str) == str(topic_num)]
#             if len(row) == 1:
#                 dist = parse_np_float_dict(row.iloc[0]['topic_distribution'])
#                 if dist:
#                     top = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:k]
#                     return [w for w, _ in top]
#         # Fallback: use the 3-word label
#         lab = topic_label_map.get(str(topic_num))
#         if lab:
#             return [w.strip() for w in str(lab).split(',') if w.strip()]
#         return []
#
#     # Build payload: top words per topic
#     top80_sem = top80.copy()
#     top80_sem['top_words'] = top80_sem['topic_num'].apply(lambda x: _top_words_for_topic_num(x, k=12))
#
#     if os.environ.get("OPENAI_API_KEY"):
#         try:
#             from openai import OpenAI
#             client = OpenAI()
#
#             payload = [
#                 {"topic_id": str(r.topic_id), "top_words": r.top_words}
#                 for r in top80_sem.itertuples(index=False)
#             ]
#
#             prompt = (
#                 "You label latent topics extracted from bank risk disclosures. "
#                 "For each topic, you receive its highest-weighted words. "
#                 "Return ONLY valid JSON (no markdown) as an array of objects with keys: "
#                 "topic_id, semantic_label, one_sentence_interpretation. "
#                 "semantic_label must be 2-6 words, concrete, and finance/risk oriented.\n\n"
#                 "Topics: " + json.dumps(payload, ensure_ascii=False)
#             )
#
#             resp = client.responses.create(
#                 model=OPENAI_MODEL,
#                 input=[{"role": "user", "content": prompt}],
#             )
#
#             out_text = getattr(resp, "output_text", None)
#             if out_text is None:
#                 out_text = str(resp)
#
#             semantic = json.loads(out_text)
#             sem_df = pd.DataFrame(semantic)
#
#             top80_sem = top80_sem.merge(sem_df, on="topic_id", how="left")
#             top80_sem.to_csv(SEMANTIC_OUT, index=False)
#             print(f"Saved semantic labels via OpenAI -> {SEMANTIC_OUT}")
#
#         except Exception as e:
#             print(f"[LABEL] OpenAI semantic labeling failed: {e}")
#             print("[LABEL] Tip: ensure `pip install openai` and OPENAI_API_KEY is set.")
#     else:
#         print("[LABEL] OPENAI_API_KEY not set; skipping semantic labeling.")
# else:
#     print("[LABEL] Skipped topic labeling (topics_words.csv or top80 file not found)")

#
# This bar-style time series is the presentation-friendly version used to highlight "spikes".
# Plot the standardized Emerging Risk Index (histogram-style time series, like in the article)
quarters = pd.PeriodIndex(res_df['quarter'], freq='Q').to_timestamp()
z_vals = res_df['z_score']

plt.figure(figsize=(10, 5))
plt.bar(
    quarters,
    z_vals,
    width=70,  # width of bars in days
    color=np.where(z_vals >= 0, 'steelblue', 'indianred'),  # blue for positive, red for negative
    edgecolor='black',
    alpha=0.85
)
plt.axhline(0, color='black', linewidth=1)
plt.title("Emerging Risk Index (Standardized ΔAdj. R²)", fontsize=13)
plt.xlabel("Quarter")
plt.ylabel("Z-score of ΔAdj. R²")
plt.xlim(pd.Timestamp("2015-01-01"), quarters.max())
plt.tight_layout()
plt.savefig("emerging_risk_index_timeseries_histogram.png")
plt.close()

print("Saved: regression_results_with_z.csv, emerging_risk_index_timeseries_histogram.png, topic_coeffs_by_quarter.csv, topic_contrib_by_quarter.csv")

# =============================================================================
# END OF SCRIPT
# =============================================================================
# If you are new:
# 1. Start by running the file.
# 2. Inspect regression_results_with_z.csv.
# 3. Look at emerging_risk_index_timeseries_histogram.png.
#
# That is the final Emerging Risk Index.
# =============================================================================