# emerging_risk_panel_continued.py
# -----------------------------------------------------------------------------
# PURPOSE
#   Build a quarterly *bank-pair* panel from daily equity data and annual
#   disclosure-based topic exposures. For each quarter t we:
#     1) compute return covariance for every bank-pair (i, j)
#     2) build controls (size/turnover/volatility + same sector/country/currency)
#     3) add exposure-similarity terms based on text topics
#     4) run two regressions (controls-only vs controls+exposures)
#     5) define an "Emerging Risk Index" as ΔAdj.R² = AdjR²(full) − AdjR²(controls)
#        and standardize it relative to a baseline period.
#   Then we attribute the peak-quarter spike to the topics that contribute most,
#   and optionally generate human-readable topic labels via OpenAI.
# -----------------------------------------------------------------------------

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
CSV_PATH = "data2_Original.csv"  # daily market data
# Minimum number of daily returns within a quarter required to compute reliable
# volatility/covariance. Quarters with fewer observations are dropped.
MIN_DAYS_PER_QUARTER = 20  # minimum trading days per quarter to include bank in analysis

# Annual-report inputs used to construct quarterly topic exposures.
# Reports are expected to be named like: YEAR_FIRM_*.pdf (e.g. 2021_UBS_group.pdf)
# The topic model output (first_doc_topics.csv) must contain a 'document' column
# plus one column per topic loading.
REPORTS_DIR = "Reports"  # folder containing YEAR_FIRM_*.pdf (relative to this script)
# Document-topic loadings output (relative to this script)
# NOTE: folder name is 'putputs_textual_factors' in your project.
FIRST_DOC_TOPICS_PATH = os.path.join("outputs_textual_factors", "first_doc_topics.csv")

# Map report firm tokens (from filenames) to Compustat gvkeys (you already verified these)
FIRM_TO_GVKEY = {
    "Danske": "15552",
    "DeutscheBank": "15575",
    "ING": "15617",
    "UBS": "144496",
}


# below function auto-detects CSV file if none specified
def detect_csv():
    """If CSV_PATH is empty, pick the largest CSV in the current folder.

    This is just a convenience for interactive work. In production runs you
    should set CSV_PATH explicitly.
    """
    files = glob.glob("*.csv")
    if not files:
        raise FileNotFoundError("No CSV files found. Place your CSV in this folder.")
    files = sorted(files, key=lambda p: os.path.getsize(p), reverse=True)
    print(f"Auto-detected CSV: {files[0]}")
    return files[0]

if not CSV_PATH:
    CSV_PATH = detect_csv()

#
# TEXT: BUILD BANK-YEAR TOPIC EXPOSURES FROM first_doc_topics.csv + report filenames
#
def build_quarterly_topic_panel(base_dir: str) -> tuple[pd.DataFrame, list[str]]:
    """Build quarterly topic exposures from annual reports.

    Returns
    -------
    (topics_fy, topic_cols)
      topics_fy: DataFrame with columns [gvkey, quarter, topic_*]
        Each row is a firm (gvkey) and the *report year anchored at Q1*.
        Exposures are later carried forward to each quarter using merge_asof
        (i.e., each quarter inherits the most recent available annual disclosure).
      topic_cols: list of topic-loading column names.

    Key idea
    --------
    first_doc_topics.csv is indexed by an integer document id. We reconstruct
    the same document order by sorting filenames in REPORTS_DIR and assigning
    doc_id = 0..N-1. Each filename provides:
      - year (from YEAR_...)
      - firm token (from _FIRM_...)
    which we map to a Compustat gvkey via FIRM_TO_GVKEY.

    Requirements
    ------------
    - Reports live in REPORTS_DIR and follow YEAR_FIRM_*.pdf naming.
    - FIRST_DOC_TOPICS_PATH exists and contains 'document' + topic columns.
    """
    reports_path = os.path.join(base_dir, REPORTS_DIR)
    # FIRST_DOC_TOPICS_PATH may already be a relative path (e.g., subfolder/file)
    topics_path = os.path.join(base_dir, FIRST_DOC_TOPICS_PATH) if not os.path.isabs(FIRST_DOC_TOPICS_PATH) else FIRST_DOC_TOPICS_PATH

    reports_ok = os.path.isdir(reports_path)
    topics_ok = os.path.isfile(topics_path)

    if not reports_ok or not topics_ok:
        print("[TEXT] Topic inputs not found.")
        print(f"[TEXT] Expected Reports dir at: {reports_path}  (exists={reports_ok})")
        print(f"[TEXT] Expected first_doc_topics.csv at: {topics_path}  (exists={topics_ok})")
        return pd.DataFrame(), []

    # Build document map in the exact order used when creating the corpus (sorted filenames)
    pdfs = sorted([f for f in os.listdir(reports_path) if f.lower().endswith(".pdf")])
    if not pdfs:
        return pd.DataFrame(), []

    doc_rows = []
    for doc_id, fname in enumerate(pdfs):
        try:
            year_str, firm_token, _ = fname.split("_", 2)
            year = int(year_str)
        except Exception:
            raise ValueError(f"Report filename does not match YEAR_FIRM_*.pdf: {fname}")

        if firm_token not in FIRM_TO_GVKEY:
            raise KeyError(
                f"Firm token '{firm_token}' from filename '{fname}' not in FIRM_TO_GVKEY. "
                f"Known keys: {sorted(FIRM_TO_GVKEY.keys())}"
            )

        doc_rows.append({
            "document": doc_id,
            "gvkey": str(FIRM_TO_GVKEY[firm_token]),
            "year": year,
        })

    doc_map = pd.DataFrame(doc_rows)

    # Load doc-topic matrix
    t = pd.read_csv(topics_path)
    if "document" not in t.columns:
        raise ValueError(f"{FIRST_DOC_TOPICS_PATH} must contain a 'document' column")

    # Topic columns: anything except 'document'
    topic_cols = [c for c in t.columns if c != "document"]
    if not topic_cols:
        raise ValueError(f"No topic columns found in {FIRST_DOC_TOPICS_PATH}")

    # Attach gvkey/year to each document row
    t = t.merge(doc_map, on="document", how="left")
    if t[["gvkey", "year"]].isna().any().any():
        missing = t[t["gvkey"].isna() | t["year"].isna()][["document"]].head(10)
        raise ValueError(
            "Some documents could not be mapped to gvkey/year. "
            f"Check corpus order vs sorted filenames. Sample unmapped: {missing.to_dict(orient='records')}"
        )

    # Firm-year exposures (annual reports): if multiple docs per gvkey-year, average them
    fy = (
        t.groupby(["gvkey", "year"], as_index=False)[topic_cols]
         .mean()
    )

    # Convert report year to a quarter anchor (Q1) and carry forward via merge_asof later
    fy["quarter"] = pd.PeriodIndex(fy["year"].astype(str) + "Q1", freq="Q")
    fy = fy.drop(columns=["year"]).sort_values(["gvkey", "quarter"])

    return fy, topic_cols

# Output of this section: raw daily issue-level data (df) with correct dtypes.
# [gvkey, quarter(=YYYYQ1), topic_1, topic_2, ...]

# ---------------------------
# 1) LOAD DATA
# ---------------------------
df = pd.read_csv(CSV_PATH) #load data
df['datadate'] = pd.to_datetime(df['datadate']) #ensure date is datetime
for col in ['cshoc','cshtrd','prccd','prchd','prcld']: #ensure numeric types
    if col in df.columns: #check column exists
        df[col] = pd.to_numeric(df[col], errors='coerce') #convert to numeric, coerce errors to NaN

#
# Output of this section: one row per (gvkey, datadate) with a single firm-level
# price/volume series, preferring the primary issue when identifiable.
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

# Output of this section: daily log returns per firm and a quarter identifier.
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

# Output of this section: bank_quarter panel with size/turnover/volatility and
# identifiers (sector/country/currency) computed from daily data.
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
# Output of this section: bank_quarter augmented with disclosure-based topic
# exposures (required), and then lagged by one quarter so exposures at t-1 explain
# covariances at t.
# ---------------------------
# 5) EXPOSURES: DISCLOSURE-BASED TOPIC LOADINGS (REQUIRED)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Try to build quarterly topic exposures (carried forward from annual disclosures)
topics_fy, topic_cols = build_quarterly_topic_panel(BASE_DIR)

# Ensure types for merge
bank_quarter["gvkey"] = bank_quarter["gvkey"].astype(str)

if topics_fy.empty:
    raise FileNotFoundError(
        "[TEXT] Required topic inputs not found or empty. "
        "Ensure Reports/ exists and outputs_textual_factors/first_doc_topics.csv exists and matches the report order."
    )

# From here on we assume real disclosure-based topics exist.
topics_fy["gvkey"] = topics_fy["gvkey"].astype(str)

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

# If any topic columns are still missing (e.g., early years before first report), fill with 0
for c in topic_cols:
    if c in bank_quarter.columns:
        bank_quarter[c] = pd.to_numeric(bank_quarter[c], errors="coerce").fillna(0.0)

print(f"Merged {len(topic_cols)} topic exposure columns from disclosures.")
exposure_cols = topic_cols

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

# Output of this section: pairs DataFrame with one row per bank-pair (i, j) per
# quarter, containing the within-quarter covariance of daily returns.
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

# Output of this section: regression-ready pair panel including products of
# lagged controls/exposures and same-sector/country/currency indicators.
# ---------------------------
# 7) MERGE CONTROLS, BUILD REGR DATA
# ---------------------------
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

# Output of this section: per-quarter adjusted R² for controls vs full model,
# plus topic attribution (betas and contribution-based importance).
# ---------------------------
# 8) PER-QUARTER REGRESSIONS
# ---------------------------
results = [] #list to hold regression results
topic_coef_rows = []   # per-quarter topic coefficients / t-stats
topic_contrib_rows = []  # per-quarter topic contribution importance
control_vars = ['size_prod_lag1','turnover_prod_lag1','vol_prod_lag1',
                'same_sector','same_country','same_currency'] #control variables
exposure_vars = [f'{c}_prod_lag1' for c in exposure_cols]

for q, qdf in pairs.groupby('quarter'): #for each quarter
    Xc = sm.add_constant(qdf[control_vars], has_constant='add') #design matrix for controls only
    yc = qdf['cov_ij'] #response variable
    try:
        model_c = sm.OLS(yc, Xc).fit() #fit OLS model with controls only
        adjR2_c = model_c.rsquared_adj #get adjusted R²
    except Exception:
        adjR2_c = np.nan #if error, set adjusted R² to NaN
    Xf = sm.add_constant(qdf[control_vars+exposure_vars], has_constant='add') #design matrix for full model (controls + exposures)
    yf = qdf['cov_ij'] #response variable
    try:
        model_f = sm.OLS(yf, Xf).fit()  # fit OLS model with controls + exposures
        adjR2_f = model_f.rsquared_adj  # get adjusted R²

        # --- Record per-topic coefficients / t-stats for this quarter ---
        for v in exposure_vars:
            topic_coef_rows.append({
                'quarter': str(q),
                'topic': v.replace('_prod_lag1', ''),
                'beta': float(model_f.params.get(v, np.nan)),
                't': float(model_f.tvalues.get(v, np.nan)),
                'p': float(model_f.pvalues.get(v, np.nan)),
            })

        # --- Record per-topic contribution importance for this quarter ---
        # Keep only exposure betas that actually entered the model; fill dropped ones with 0
        betas = (
            model_f.params
            .filter(like='_prod_lag1')
            .reindex(exposure_vars)
            .fillna(0.0)
        )

        X_topics = qdf[exposure_vars]
        contrib = X_topics.mul(betas, axis=1)
        # mean_abs_contrib is a simple magnitude measure: average over pairs of |beta_k * x_k|.
        # It ranks which topics matter most *in this quarter* for explaining covariances.
        imp = contrib.abs().mean(axis=0)

        for v, val in imp.items():
            topic_contrib_rows.append({
                'quarter': str(q),
                'topic': v.replace('_prod_lag1', ''),
                'mean_abs_contrib': float(val),
            })

    except Exception:
        adjR2_f = np.nan  # if error, set adjusted R² to NaN
    results.append({'quarter':str(q),'adjR2_controls':adjR2_c,'adjR2_full':adjR2_f,
                    'delta_adjR2':(adjR2_f - adjR2_c) if (pd.notna(adjR2_f) and pd.notna(adjR2_c)) else np.nan}) #store results

res_df = pd.DataFrame(results).sort_values('quarter') #create results DataFrame
res_df.to_csv("regression_results.csv", index=False) #save regression results

# Save topic attribution outputs (coefficients and contribution-based importance)
if topic_coef_rows:
    pd.DataFrame(topic_coef_rows).to_csv("topic_coeffs_by_quarter.csv", index=False)
if topic_contrib_rows:
    pd.DataFrame(topic_contrib_rows).to_csv("topic_contrib_by_quarter.csv", index=False)

# Output of this section: plots and a standardized Emerging Risk Index (z-score)
# based on ΔAdj.R² relative to the chosen baseline period.
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

# Choose baseline period (paper uses 1998–2003)
baseline_mask = (res_df['quarter'] >= '1998Q1') & (res_df['quarter'] <= '2003Q4')
baseline = res_df.loc[baseline_mask].copy()

# If baseline has missing delta values, drop them for mean/std
base_vals = baseline['delta_adjR2'].dropna()

# Guard against too few baseline points or zero std
if len(base_vals) >= 3 and base_vals.std() not in (0, None) and not np.isnan(base_vals.std()): #ensure enough data for baseline
    mu = base_vals.mean() #baseline mean
    sigma = base_vals.std() #baseline std
else:
    # Fallback: use entire sample as baseline
    sample_vals = res_df['delta_adjR2'].dropna() #all available delta values
    mu = sample_vals.mean() if len(sample_vals) else np.nan #compute mean
    sigma = sample_vals.std() if len(sample_vals) else np.nan #compute std

# Compute z-score
res_df['z_score'] = (res_df['delta_adjR2'] - mu) / sigma #standardize delta adjusted R²

# Save an extended results file (keeps the original file unchanged)
res_df.to_csv("regression_results_with_z.csv", index=False) #save extended results with z-score

# --- Identify and save the top topic drivers for the peak quarter (max z-score) ---
try:
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
import ast

# Path to topic-words file (adjust if needed)
TOPIC_WORDS_PATH = os.path.join("outputs_textual_factors", "topics_words.csv")

if os.path.isfile(TOPIC_WORDS_PATH) and os.path.isfile("topic_drivers_peak_quarter_top80.csv"):
    # Load topic words
    topics_words = pd.read_csv(TOPIC_WORDS_PATH)

    # Detect topic-words format and build labels
    topic_label_map = {}

    # Case A: your file format (topic, topic_distribution)
    if set(['topic', 'topic_distribution']).issubset(topics_words.columns):
        import ast
        import re

        def parse_np_float_dict(s: str) -> dict:
            """Parse strings like "{'word': np.float64(0.12), ...}" into {word: float}."""
            if pd.isna(s):
                return {}
            txt = str(s)
            # Convert np.float64(x) -> x
            txt = re.sub(r"np\.float64\(([^\)]+)\)", r"\1", txt)
            # Also handle possible numpy.float64
            txt = re.sub(r"numpy\.float64\(([^\)]+)\)", r"\1", txt)
            try:
                d = ast.literal_eval(txt)
            except Exception:
                return {}
            # Coerce values to float
            out = {}
            for k, v in d.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue
            return out

        def make_label_from_dist(dist: dict, k: int = 3) -> str:
            if not dist:
                return None
            top = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:k]
            return ", ".join([w for w, _ in top])

        for _, r in topics_words.iterrows():
            tid = str(r['topic'])
            dist = parse_np_float_dict(r['topic_distribution'])
            topic_label_map[tid] = make_label_from_dist(dist, k=3)

    # Case B: long format (topic, word[, weight])
    elif set(['topic', 'word']).issubset(topics_words.columns):
        weight_col = 'weight' if 'weight' in topics_words.columns else None
        for tid, g in topics_words.groupby('topic'):
            if weight_col:
                g = g.sort_values(weight_col, ascending=False)
            top_words = g['word'].astype(str).head(3).tolist()
            topic_label_map[str(tid)] = ', '.join(top_words)

    # Case C: wide format with stringified dict (topic_id, words)
    elif 'words' in topics_words.columns:
        import ast

        id_col = 'topic_id' if 'topic_id' in topics_words.columns else ('topic' if 'topic' in topics_words.columns else None)
        if id_col is None:
            raise ValueError("topics_words.csv has 'words' but no 'topic_id'/'topic' column")

        def safe_eval_dict(x):
            try:
                return ast.literal_eval(x)
            except Exception:
                return {}

        topics_words['words_dict'] = topics_words['words'].apply(safe_eval_dict)

        def make_label(word_dict, k=3):
            if not isinstance(word_dict, dict) or not word_dict:
                return None
            top_words = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)[:k]
            return ', '.join([w for w, _ in top_words])

        for _, r in topics_words.iterrows():
            topic_label_map[str(r[id_col])] = make_label(r['words_dict'], k=3)

    else:
        raise ValueError(
            "Unrecognized topics_words.csv format. Expected either: "
            "(topic, topic_distribution) OR (topic, word[, weight]) OR (topic_id/topic, words)."
        )

    # Load top-80 contributing topics
    top80 = pd.read_csv("topic_drivers_peak_quarter_top80.csv")

    # Our drivers file uses values like 'topic_loading_159' in column 'topic'.
    # The topic-words file typically uses numeric ids like '159'.
    top80['topic_id'] = top80['topic'].astype(str)
    top80['topic_num'] = top80['topic_id'].apply(lambda s: s.split('_')[-1] if '_' in s else s)

    # Map numeric topic ids to word-based labels
    top80['topic_label'] = top80['topic_num'].map(topic_label_map)

    # Save labeled output
    top80.to_csv(
        "topic_drivers_peak_quarter_top80_labeled.csv",
        index=False
    )

    print("Saved labeled top-80 topics -> topic_drivers_peak_quarter_top80_labeled.csv")

    # --------------------------------------------------
    # Optional: Use OpenAI API to generate semantic topic names
    # --------------------------------------------------
    # Requires: `pip install openai` and OPENAI_API_KEY set in your environment.
    # Produces: topic_drivers_peak_quarter_top80_semantic.csv

    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    SEMANTIC_OUT = "topic_drivers_peak_quarter_top80_semantic.csv"

    def _top_words_for_topic_num(topic_num: str, k: int = 12):
        # Prefer the full distribution (your topics_words.csv format)
        if set(['topic', 'topic_distribution']).issubset(topics_words.columns):
            row = topics_words[topics_words['topic'].astype(str) == str(topic_num)]
            if len(row) == 1:
                dist = parse_np_float_dict(row.iloc[0]['topic_distribution'])
                if dist:
                    top = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:k]
                    return [w for w, _ in top]
        # Fallback: use the 3-word label
        lab = topic_label_map.get(str(topic_num))
        if lab:
            return [w.strip() for w in str(lab).split(',') if w.strip()]
        return []

    # Build payload: top words per topic
    top80_sem = top80.copy()
    top80_sem['top_words'] = top80_sem['topic_num'].apply(lambda x: _top_words_for_topic_num(x, k=12))

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()

            payload = [
                {"topic_id": str(r.topic_id), "top_words": r.top_words}
                for r in top80_sem.itertuples(index=False)
            ]

            prompt = (
                "You label latent topics extracted from bank risk disclosures. "
                "For each topic, you receive its highest-weighted words. "
                "Return ONLY valid JSON (no markdown) as an array of objects with keys: "
                "topic_id, semantic_label, one_sentence_interpretation. "
                "semantic_label must be 2-6 words, concrete, and finance/risk oriented.\n\n"
                "Topics: " + json.dumps(payload, ensure_ascii=False)
            )

            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=[{"role": "user", "content": prompt}],
            )

            out_text = getattr(resp, "output_text", None)
            if out_text is None:
                out_text = str(resp)

            semantic = json.loads(out_text)
            sem_df = pd.DataFrame(semantic)

            top80_sem = top80_sem.merge(sem_df, on="topic_id", how="left")
            top80_sem.to_csv(SEMANTIC_OUT, index=False)
            print(f"Saved semantic labels via OpenAI -> {SEMANTIC_OUT}")

        except Exception as e:
            print(f"[LABEL] OpenAI semantic labeling failed: {e}")
            print("[LABEL] Tip: ensure `pip install openai` and OPENAI_API_KEY is set.")
    else:
        print("[LABEL] OPENAI_API_KEY not set; skipping semantic labeling.")
else:
    print("[LABEL] Skipped topic labeling (topics_words.csv or top80 file not found)")

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
plt.tight_layout()
plt.savefig("emerging_risk_index_timeseries_histogram.png")
plt.close()

print("Saved: regression_results_with_z.csv, emerging_risk_index_timeseries_histogram.png, topic_coeffs_by_quarter.csv, topic_contrib_by_quarter.csv")