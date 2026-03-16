import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
import warnings
import matplotlib.pyplot as plt

# Resolve repo-root-relative paths robustly (works regardless of run cwd)
HERE = Path(__file__).resolve().parent              # .../Regression
REPO_ROOT = HERE.parent                             # .../Emerging-Credit-Risk_1

# ------------------------------------------------------------
# Helper: normalize firm identifiers (tickers / names)
# ------------------------------------------------------------
def _norm_id(x) -> str:
    """Normalize IDs for robust merges (tickers like 'ANA.MC', 'ORSTED.CO', etc.)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    # Keep case as-is for tickers (case can matter in some feeds), just trim spaces
    # but normalize multiple spaces and common weird whitespace.
    s = " ".join(s.split())
    return s



# Baseline window for z-scores in the dynamic setup:
# use the years inside the rolling text window BEFORE the target year.
BASELINE_START_YEAR = None
BASELINE_END_YEAR = None

# ============================================================
# DYNAMIC TEXT WINDOW
# ============================================================
# The regression year that should use the rolling 5-year text window ending in this year.
# Example: DYNAMIC_TARGET_YEAR = 2015 -> extraction_summary_2011_2015_V1.csv
DYNAMIC_TARGET_YEAR = 2020
DYNAMIC_WINDOW_LENGTH = 5
DYNAMIC_WINDOW_START = DYNAMIC_TARGET_YEAR - DYNAMIC_WINDOW_LENGTH + 1
DYNAMIC_WINDOW_END = DYNAMIC_TARGET_YEAR
DYNAMIC_TF_RUN_LABEL = f"{DYNAMIC_WINDOW_START}_{DYNAMIC_WINDOW_END}"

# --------------------------------------------------
# Baseline window for dynamic setup
# --------------------------------------------------
DYNAMIC_BASELINE_START_YEAR = DYNAMIC_WINDOW_START
DYNAMIC_BASELINE_END_YEAR = DYNAMIC_TARGET_YEAR - 1
TOP_Z_COUNT = 10

# ============================================================
# OUTPUT NAMING + SINGLE-TOPIC MODE
# ============================================================
# A short tag appended to output filenames (e.g., "gfc", "covid", "v2", "test1")
OUTPUT_TAG = "run1"

# If True, only run regressions + plots for ONE topic (TOPIC_SINGLE)
RUN_SINGLE_TOPIC = False

# The single topic to run when RUN_SINGLE_TOPIC=True
TOPIC_SINGLE = "topic_loading_204"
TOPIC_PLOT_TITLE_TEMPLATE = "Interest rate"


# If True, only generate ONE topic plot (TOPIC_SINGLE). If False, plot all topics in TOPIC_INCLUDE.
PLOT_SINGLE_TOPIC = True

# If True, run diagnostics/statistics block at the end of the script.
RUN_DIAGNOSTICS = False
RUN_VIF_TEST = False
PRUNE_TOPICS_BY_DR2 = False
TARGET_TOPIC_COUNT = 35
VIF_MAX_ROWS = 20000
VIF_PRINT_TOP_N = 30

# If set, overrides the default title for the static risk model plot.
STATIC_PLOT_TITLE = "Static risk model (4-quarter smoothed average topic z-score)"

# Minimum firm-year topic loading (absolute). Values below are treated as 0 (noise filter).
MIN_TOPIC_LOADING = 0

# Minimum baseline standard deviation used in z-score denominator (avoids NaN for near-constant topics)
MIN_BASELINE_SD = 0.00005  # small floor to avoid NaN z-scores when baseline dR2 is near-constant
# TRY AND WINDSORIZE the SD 1% or 5%.
# TRY AND WINDSORIZE the SD 1% or 5%.# TRY AND WINDSORIZE the SD 1% or 5%.
# Plot window (only affects graphs, not analysis)
PLOT_START_YEAR = 2005
PLOT_END_YEAR = 2024

 # Explicit topic list for manual testing
TOPIC_INCLUDE = [
"topic_loading_10",
"topic_loading_16",
"topic_loading_18",
"topic_loading_20",
"topic_loading_25",
"topic_loading_31",
"topic_loading_39",
"topic_loading_42",
"topic_loading_45",
"topic_loading_55",
"topic_loading_64",
"topic_loading_77",
"topic_loading_78",
"topic_loading_83",
"topic_loading_88",
"topic_loading_100",
"topic_loading_114",
"topic_loading_122",
"topic_loading_133",
"topic_loading_134",
"topic_loading_138",
"topic_loading_144",
"topic_loading_145",
"topic_loading_146",
"topic_loading_159",
"topic_loading_162",
"topic_loading_165",
"topic_loading_179",
"topic_loading_182",
"topic_loading_187",
"topic_loading_191",
"topic_loading_193",
"topic_loading_202",
"topic_loading_204",
"topic_loading_210",
"topic_loading_211",
"topic_loading_226",
"topic_loading_227",
"topic_loading_230",
"topic_loading_234",
"topic_loading_236",
"topic_loading_247",
"topic_loading_267",
"topic_loading_282",
"topic_loading_288",
"topic_loading_293",
"topic_loading_302",
"topic_loading_307",
"topic_loading_311",
"topic_loading_324",
"topic_loading_332",
"topic_loading_344",
"topic_loading_376",
"topic_loading_381",
"topic_loading_411",
"topic_loading_415",
"topic_loading_421",
"topic_loading_440",
"topic_loading_506",
"topic_loading_512",
"topic_loading_513",
"topic_loading_524",
"topic_loading_526",
"topic_loading_541",
"topic_loading_545",
"topic_loading_547",
"topic_loading_592",
"topic_loading_604",
]

# --------------------------------------------------
# Forward stepwise topic selection for the dynamic model
# Start from a broad candidate pool, rank topics by pooled forward stepwise fit,
# and keep up to STEPWISE_MAX_TOPICS topics for the final marginal-contribution stage.
# --------------------------------------------------
RUN_FORWARD_STEPWISE = True
STEPWISE_MAX_CANDIDATES = 150
STEPWISE_MAX_TOPICS = 30
STEPWISE_MIN_DR2 = None
TOPIC_EXCLUDE = []  # e.g. ["topic_loading_12"]

# Minimum number of valid pairs needed to estimate a topic regression in a quarter
MIN_VALID_PAIRS = 1

# ============================
# Step 0: Load data
# ============================
COV_CANDIDATES = [
    HERE / "Output" / "quarterly_pairwise_covariance.csv",
    REPO_ROOT / "Regression" / "Output" / "quarterly_pairwise_covariance.csv",
]
COV_PATH = next((p for p in COV_CANDIDATES if p.exists()), None)
if COV_PATH is None:
    raise FileNotFoundError(
        "Could not find quarterly_pairwise_covariance.csv. Tried:\n"
        + "\n".join(str(p) for p in COV_CANDIDATES)
    )
TF_CANDIDATES = [
    REPO_ROOT / "Text analytics" / "outputs_textual_factors" / f"extraction_summary_{DYNAMIC_TF_RUN_LABEL}_V1.csv",
    REPO_ROOT / "Text analytics" / "Scripts" / "outputs_textual_factors" / f"extraction_summary_{DYNAMIC_TF_RUN_LABEL}_V1.csv",
    REPO_ROOT / "Text analytics" / "outputs_textual_factors" / f"extraction_summary_{DYNAMIC_WINDOW_START}_{DYNAMIC_WINDOW_END}.csv",
    REPO_ROOT / "Text analytics" / "Scripts" / "outputs_textual_factors" / f"extraction_summary_{DYNAMIC_WINDOW_START}_{DYNAMIC_WINDOW_END}.csv",
]
TF_PATH = next((p for p in TF_CANDIDATES if p.exists()), None)
if TF_PATH is None:
    raise FileNotFoundError(
        f"Could not find rolling-window extraction summary for {DYNAMIC_TF_RUN_LABEL}. Tried:\n"
        + "\n".join(str(p) for p in TF_CANDIDATES)
    )

print("Loading covariance panel from:", COV_PATH)
print(f"Loading rolling-window extraction summary ({DYNAMIC_TF_RUN_LABEL}) from:", TF_PATH)

# Controls (firm-year financials) path
CTRL_CANDIDATES = [
    HERE / "Data" / "Control_variable_final.xlsx",
    REPO_ROOT / "Regression" / "Data" / "Control_variable_final.xlsx",
]
CTRL_PATH = next((p for p in CTRL_CANDIDATES if p.exists()), None)
if CTRL_PATH is None:
    raise FileNotFoundError(
        "Could not find Control_variable_final.xlsx. Tried:\n"
        + "\n".join(str(p) for p in CTRL_CANDIDATES)
    )
print("Loading controls from:", CTRL_PATH)

cov = pd.read_csv(COV_PATH)
tf = pd.read_csv(TF_PATH)
print(
    f"Dynamic text window in use: {DYNAMIC_WINDOW_START}-{DYNAMIC_WINDOW_END} "
    f"(target year = {DYNAMIC_TARGET_YEAR})"
)
# Compact TF diagnostics
try:
    tf["year"] = pd.to_numeric(tf["year"], errors="coerce")
    print("TF_PATH:", TF_PATH)
    print("TF year min/max:", int(tf["year"].min()), int(tf["year"].max()))
    print("TF rows:", int(len(tf)))
except Exception as _e:
    print("WARNING: TF diagnostics failed:", _e)
controls_raw = pd.read_excel(CTRL_PATH)

# Normalize covariance identifiers
for col in ["i", "j"]:
    if col in cov.columns:
        cov[col] = cov[col].astype(str).str.strip().map(_norm_id)

# -----------------------------
# Canonicalize columns
# -----------------------------
controls_raw.columns = [str(c).strip().lower() for c in controls_raw.columns]

col_map = {
    "company": "company",
    "company name": "company",
    "ticker": "ticker",
    "ticker name": "ticker",
    "stock code": "ticker",

    "year": "year",
    "year date": "year",
    "date": "date",

    "total assets": "total_assets",
    "total_assets": "total_assets",

    "company market capitalization": "market_cap",
    "market cap": "market_cap",
    "market capitalization": "market_cap",
    "market_cap": "market_cap",

    "capital expenditures - total, 5 yr cagr": "capex_5y_cagr",
    "capex_5y_cagr": "capex_5y_cagr",

    "net debt - mean": "net_debt",
    "net debt": "net_debt",
    "net_debt": "net_debt",

    "cash & cash equivalents - total": "cash",
    "cash & cash equivalents": "cash",
    "cash and cash equivalents": "cash",
    "cash": "cash",

    "ebit margin - %": "ebit_margin",
    "ebit margin": "ebit_margin",
    "ebit_margin": "ebit_margin",

    "net income after tax": "net_income",
    "net income": "net_income",
    "net_income": "net_income",

    "company green revenue percentage": "green_revenue_pct",
    "green_revenue_pct": "green_revenue_pct",

    "common equity - total": "common_equity",
    "common equity": "common_equity",
    "common_equity": "common_equity",
}

# Canonicalize columns
controls_raw = controls_raw.rename(columns={c: col_map.get(c, c) for c in controls_raw.columns})

#
# Build mappings between company name <-> ticker (most frequent per entity)
company_to_ticker = {}
ticker_to_company = {}
if "company" in controls_raw.columns and "ticker" in controls_raw.columns:
    tmp_map = (
        controls_raw[["company", "ticker"]]
        .dropna()
        .assign(
            company=lambda d: d["company"].astype(str).str.strip().map(_norm_id),
            ticker=lambda d: d["ticker"].astype(str).str.strip().map(_norm_id),
        )
    )
    if not tmp_map.empty:
        # most frequent ticker per company
        company_to_ticker = (
            tmp_map.groupby("company")["ticker"].agg(lambda s: s.value_counts().index[0]).to_dict()
        )
        # most frequent company per ticker
        ticker_to_company = (
            tmp_map.groupby("ticker")["company"].agg(lambda s: s.value_counts().index[0]).to_dict()
        )

# Diagnostics: show a few identifiers to confirm matching
print("Cov sample i identifiers:", cov["i"].dropna().astype(str).unique()[:10].tolist())
print("Controls sample companies:", controls_raw["company"].dropna().astype(str).unique()[:10].tolist())

# ------------------------------------------------------------
# Derive / parse year robustly (your Excel uses 'Year date')
# ------------------------------------------------------------
# If year is missing but date exists
if "year" not in controls_raw.columns and "date" in controls_raw.columns:
    controls_raw["year"] = pd.to_datetime(controls_raw["date"], errors="coerce").dt.year

# If year exists, parse robustly.
# Priority order:
#  (1) If values look like plain years (e.g., 2004, 2005, ...), keep as numeric years.
#  (2) Else, try datetime parsing of strings/datetimes.
#  (3) Else, try Excel serial day conversion.
if "year" in controls_raw.columns:
    y = controls_raw["year"]

    # (1) Plain numeric years?
    y_num = pd.to_numeric(y, errors="coerce")
    share_plain_year = float(((y_num >= 1900) & (y_num <= 2100)).mean()) if len(y_num) else 0.0
    if share_plain_year > 0.8:
        controls_raw["year"] = y_num
    else:
        # (2) Parse as datetime-like / date strings
        y_dt = pd.to_datetime(y, errors="coerce")
        if y_dt.notna().mean() > 0.5:
            years = y_dt.dt.year
            controls_raw["year"] = years
        else:
            # (3) Excel serial days (origin 1899-12-30)
            y_dt2 = pd.to_datetime(y_num, unit="D", origin="1899-12-30", errors="coerce")
            if y_dt2.notna().mean() > 0.5:
                controls_raw["year"] = y_dt2.dt.year
            else:
                controls_raw["year"] = y_num

#
# Must have company name (because cov i/j are company names like 'Acciona', 'EON', etc.)
if "company" not in controls_raw.columns:
    raise ValueError(
        "Control_variable_final.xlsx must contain a company name column (e.g., 'Company name')."
    )

# Normalize ids
controls_raw["company"] = controls_raw["company"].astype(str).str.strip().map(_norm_id)

# Ticker is optional now (only used for optional conversions when cov uses tickers)
if "ticker" in controls_raw.columns:
    controls_raw["ticker"] = controls_raw["ticker"].astype(str).str.strip().map(_norm_id)

# Numeric coercion helper (handles %, EU decimals)
def _to_num(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("%", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(".", "", regex=False)  # thousand sep
    s = s.str.replace(",", ".", regex=False)  # decimal comma
    return pd.to_numeric(s, errors="coerce")

for c in [
    "total_assets","market_cap","capex_5y_cagr","net_debt","cash",
    "ebit_margin","net_income","green_revenue_pct","common_equity"
]:
    if c in controls_raw.columns:
        controls_raw[c] = _to_num(controls_raw[c])

 # Ensure numeric year after parsing
controls_raw["year"] = pd.to_numeric(controls_raw["year"], errors="coerce")
print("Controls: year non-missing share (after parsing):", float(pd.Series(controls_raw["year"]).notna().mean()))
print("Controls: sample years:", pd.Series(controls_raw["year"]).dropna().astype(int).unique()[:5])
print("Controls: year min/max:", int(pd.Series(controls_raw["year"]).dropna().min()), int(pd.Series(controls_raw["year"]).dropna().max()))
controls_raw["year"] = controls_raw["year"].where((controls_raw["year"] >= 1900) & (controls_raw["year"] <= 2100))
controls_raw = controls_raw.dropna(subset=["company","year"]).copy()
# Drop clearly invalid company values
controls_raw = controls_raw[~controls_raw["company"].isin(["", "0", "nan", "None"])].copy()
controls_raw["year"] = controls_raw["year"].astype(int)

# Build the 8 selected controls
controls_raw["ln_assets"] = np.log(controls_raw["total_assets"].where(controls_raw["total_assets"] > 0))
controls_raw["equity_assets"] = controls_raw["common_equity"] / controls_raw["total_assets"].replace({0: np.nan})

need = [
    "ln_assets","cash","net_debt","ebit_margin","net_income",
    "equity_assets","capex_5y_cagr","green_revenue_pct"
]
missing = [c for c in need if c not in controls_raw.columns]
if missing:
    raise ValueError(f"Missing required controls in Control_variable_final.xlsx: {missing}. Found: {list(controls_raw.columns)}")

# Use company name as the firm identifier for merging (matches cov i/j)
controls_firm_year = (
    controls_raw
    .groupby(["company", "year"], as_index=False)[need]
    .mean()
    .rename(columns={"company": "firm_id"})
)
controls_firm_year["firm_id"] = controls_firm_year["firm_id"].map(_norm_id)

print("Loaded controls firm-year table:", controls_firm_year.shape)
print("Controls sample firm_ids:", controls_firm_year["firm_id"].head(5).tolist())

all_topic_cols = [c for c in tf.columns if c.startswith("topic_loading_")]

# Apply include/exclude rules (single-topic override supported)
if RUN_SINGLE_TOPIC:
    if TOPIC_SINGLE not in all_topic_cols:
        raise ValueError(
            f"TOPIC_SINGLE={TOPIC_SINGLE} not found in extraction summary. "
            f"Example topics: {all_topic_cols[:10]}"
        )
    topic_cols = [TOPIC_SINGLE]
else:
    if TOPIC_INCLUDE:
        missing = [c for c in TOPIC_INCLUDE if c not in all_topic_cols]
        if missing:
            raise ValueError(f"TOPIC_INCLUDE contains columns not found in extraction summary: {missing}")
        topic_cols = list(TOPIC_INCLUDE)
    else:
        topic_cols = [c for c in all_topic_cols if c not in set(TOPIC_EXCLUDE)]

if not topic_cols:
    raise ValueError("No topics selected. Check TOPIC_INCLUDE / TOPIC_EXCLUDE settings.")

print(f"Initial topic pool size before stepwise selection: {len(topic_cols)}")
print("RUN_SINGLE_TOPIC:", RUN_SINGLE_TOPIC, "TOPIC_SINGLE:", TOPIC_SINGLE)
print("PLOT_SINGLE_TOPIC:", PLOT_SINGLE_TOPIC)
print("OUTPUT_TAG:", OUTPUT_TAG)

# ============================
# Step 1: Prepare firm-year TFs
# ============================

tf_firm_year = (
    tf.groupby(["bank", "year"], as_index=False)[topic_cols]
      .mean()
)
# Noise filter: treat very small topic loadings as zero before building pair exposures
for c in topic_cols:
    tf_firm_year[c] = pd.to_numeric(tf_firm_year[c], errors="coerce")
    tf_firm_year[c] = tf_firm_year[c].where(tf_firm_year[c].abs() >= MIN_TOPIC_LOADING, 0.0)

# --------------------------------------------------
# Candidate-topic prefilter for stepwise selection:
# keep the 150 topics with the highest average absolute loading in the rolling text window.
# --------------------------------------------------
if RUN_FORWARD_STEPWISE and not RUN_SINGLE_TOPIC and len(topic_cols) > STEPWISE_MAX_CANDIDATES:
    topic_strength = (
        tf_firm_year[topic_cols]
        .abs()
        .mean(axis=0)
        .sort_values(ascending=False)
    )
    topic_cols = topic_strength.head(STEPWISE_MAX_CANDIDATES).index.tolist()
    tf_firm_year = tf_firm_year[["bank", "year"] + topic_cols].copy()
    print(
        f"Stepwise candidate prefilter applied: keeping top {len(topic_cols)} topics "
        f"by average absolute loading (max candidates = {STEPWISE_MAX_CANDIDATES})."
    )
tf_firm_year["bank"] = tf_firm_year["bank"].map(_norm_id)

# ------------------------------------------------------------
# Topic activation gate: yearly mean topic loading (used to suppress sparse-topic z spikes)
# ------------------------------------------------------------
topic_year_mean = (
    tf_firm_year.groupby("year", as_index=False)[topic_cols].mean()
)
# dict: (topic, year) -> mean loading
topic_year_mean_map = {
    (c, int(y)): float(v)
    for y, row in topic_year_mean.set_index("year")[topic_cols].iterrows()
    for c, v in row.items()
}



# ============================
# Step 2: Align TFs to quarters (same-year)
# ============================

# NOTE: We use same-year alignment (t) rather than t-1.
# Interpretation: Annual Report t is assumed informative from Jan of year t.
cov["q_year"] = cov["quarter"].astype(str).str[:4].astype(int)

# Use same-year alignment:
# Annual Report year t is assumed available from Jan of year t
# so covariances in year t use controls/text from year t
cov["lag_year"] = cov["q_year"]

# If cov identifiers are tickers (contain a dot like 'ANA.MC'), convert them to company names.
# Otherwise, assume they are already company names.
try:
    share_dot_i = float(cov["i"].astype(str).str.contains("\\.").mean())
    share_dot_j = float(cov["j"].astype(str).str.contains("\\.").mean())
except Exception:
    share_dot_i, share_dot_j = 0.0, 0.0

print(f"Dot-share cov ids: i={share_dot_i:.3f}, j={share_dot_j:.3f}")

cov["i"] = cov["i"].astype(str).str.strip().map(_norm_id)
cov["j"] = cov["j"].astype(str).str.strip().map(_norm_id)

# If mostly tickers, map ticker -> company
if (share_dot_i > 0.2) or (share_dot_j > 0.2):
    if not ticker_to_company:
        raise ValueError("ticker_to_company mapping is empty; cannot convert cov tickers to company names.")
    cov["i"] = cov["i"].map(lambda x: ticker_to_company.get(x, x))
    cov["j"] = cov["j"].map(lambda x: ticker_to_company.get(x, x))
    print("Applied ticker->company mapping to cov i/j.")

# Post-mapping diagnostics
print("Cov sample i after mapping:", cov["i"].dropna().astype(str).unique()[:10].tolist())
print("Example unmapped i (still contains a dot):", [x for x in cov["i"].unique() if "." in str(x)][:10])

panel = cov.merge(
    tf_firm_year, left_on=["i", "lag_year"], right_on=["bank", "year"], how="left"
)

panel = panel.merge(
    tf_firm_year, left_on=["j", "lag_year"], right_on=["bank", "year"],
    how="left", suffixes=("_i", "_j")
)

# Merge controls for i and j (lagged by one year)
panel = panel.merge(
    controls_firm_year,
    left_on=["i", "lag_year"],
    right_on=["firm_id", "year"],
    how="left"
).drop(columns=["firm_id", "year"])

panel = panel.merge(
    controls_firm_year,
    left_on=["j", "lag_year"],
    right_on=["firm_id", "year"],
    how="left",
    suffixes=("_i_ctrl", "_j_ctrl")
).drop(columns=["firm_id", "year"])

# Rename control columns to consistent i/j names
# After the second merge, pandas applies suffixes to overlapping columns:
# - i-side columns become *_i_ctrl
# - j-side columns become *_j_ctrl
panel = panel.rename(columns={
    "ln_assets_i_ctrl": "ln_assets_i",
    "cash_i_ctrl": "cash_i",
    "net_debt_i_ctrl": "net_debt_i",
    "ebit_margin_i_ctrl": "ebit_margin_i",
    "net_income_i_ctrl": "net_income_i",
    "equity_assets_i_ctrl": "equity_assets_i",
    "capex_5y_cagr_i_ctrl": "capex_5y_cagr_i",
    "green_revenue_pct_i_ctrl": "green_revenue_pct_i",

    "ln_assets_j_ctrl": "ln_assets_j",
    "cash_j_ctrl": "cash_j",
    "net_debt_j_ctrl": "net_debt_j",
    "ebit_margin_j_ctrl": "ebit_margin_j",
    "net_income_j_ctrl": "net_income_j",
    "equity_assets_j_ctrl": "equity_assets_j",
    "capex_5y_cagr_j_ctrl": "capex_5y_cagr_j",
    "green_revenue_pct_j_ctrl": "green_revenue_pct_j",
})

# Ensure *_miss columns exist even if a control never goes missing (then miss=0)
for base in ["net_income", "equity_assets", "capex_5y_cagr", "green_revenue_pct"]:
    for side in ["i", "j"]:
        col = f"{base}_{side}_miss"
        if col not in panel.columns:
            panel[col] = 0

ctrl_cols = [
    "ln_assets_i","ln_assets_j",
    "cash_i","cash_j",
    "net_debt_i","net_debt_j",
    "ebit_margin_i","ebit_margin_j",
    "net_income_i","net_income_j",
    "equity_assets_i","equity_assets_j",
    "capex_5y_cagr_i","capex_5y_cagr_j",
    "green_revenue_pct_i","green_revenue_pct_j",
]
if all(c in panel.columns for c in ctrl_cols):
    cov_i_match = float(panel["ln_assets_i"].notna().mean())
    cov_j_match = float(panel["ln_assets_j"].notna().mean())
    print("Controls coverage (share non-missing, overall):", float(panel[ctrl_cols].notna().mean().mean()))
    print("Controls match rate i-side:", cov_i_match, "j-side:", cov_j_match)
else:
    raise ValueError(
        f"Expected control columns missing after merge/rename: {[c for c in ctrl_cols if c not in panel.columns]}. "
        "This indicates the controls merge did not match i/j identifiers."
    )

# --------------------------------------------------
# Relax the "all-controls-required" restriction
# --------------------------------------------------
# Goal: keep 2005–2008 rows by not dropping pairs just because some controls are missing.
# Approach:
#  1) Define a smaller set of CORE controls that must be present.
#  2) For the remaining controls, impute missing values within year (lag_year)
#     and add missingness dummies so we don't silently assume missing==0.

CORE_CTRL_COLS = [
    "ln_assets_i","ln_assets_j",
    "ebit_margin_i","ebit_margin_j",
]

# Controls we will impute (everything else)
IMPUTE_CTRL_COLS = [c for c in ctrl_cols if c not in CORE_CTRL_COLS]

# Create missingness dummies + impute within-year median (lag_year)
# NOTE: This is a pragmatic way to maximize sample size.
# If you prefer a stricter approach, remove this block and instead reduce controls.
for c in IMPUTE_CTRL_COLS:
    miss = panel[c].isna()
    panel[f"{c}_miss"] = miss.astype(int)

# Year-median imputation per lag_year (for each control column)
# Falls back to global median if a year has all-missing for that control.
for c in IMPUTE_CTRL_COLS:
    year_median = panel.groupby("lag_year")[c].transform("median")
    global_median = float(panel[c].median(skipna=True)) if panel[c].notna().any() else 0.0
    panel[c] = panel[c].fillna(year_median).fillna(global_median)

# Quick diagnostic: how many rows survive with CORE controls only?
panel["year_q"] = panel["quarter"].astype(str).str[:4].astype(int)

# --------------------------------------------------
# Restrict the regression sample to the rolling text window only.
# Outside this window the topic loadings are unavailable and would otherwise
# enter as zeros, which is not economically meaningful in the dynamic setup.
# --------------------------------------------------
panel = panel[
    panel["lag_year"].between(DYNAMIC_WINDOW_START, DYNAMIC_WINDOW_END)
].copy()

print(
    f"Restricted panel to rolling text window years {DYNAMIC_WINDOW_START}-{DYNAMIC_WINDOW_END}. "
    f"Remaining rows: {len(panel)}"
)
row_ok_core = panel[CORE_CTRL_COLS].notna().all(axis=1)
print("Valid rows by year (CORE controls complete):")
print(panel.loc[row_ok_core].groupby("year_q").size().head(10))


# Existing diagnostic (kept): how many rows have ALL controls vs CORE controls
row_ok_all = panel[ctrl_cols].notna().all(axis=1)
row_ok_core = panel[CORE_CTRL_COLS].notna().all(axis=1)

print("Valid rows by year (ALL 8 controls complete):")
print(panel.loc[row_ok_all].groupby("year_q").size().head(10))

print("Total rows by year:")
print(panel.groupby("year_q").size().head(10))

# ============================
# Optional: VIF test for topic multicollinearity
# Uses the same pairwise topic exposures that enter the regression: topic_i * topic_j
# ============================
if RUN_VIF_TEST:
    print("\nRunning VIF test for topic exposures...")

    vif_exposure = pd.DataFrame({
        k: pd.to_numeric(panel[f"{k}_i"], errors="coerce").fillna(0.0) *
           pd.to_numeric(panel[f"{k}_j"], errors="coerce").fillna(0.0)
        for k in topic_cols
    })

    # Keep rows with at least some topic exposure
    row_mask = (vif_exposure.abs().sum(axis=1) > 0)
    vif_exposure = vif_exposure.loc[row_mask].copy()

    # Drop constant / near-constant columns before VIF
    col_std = vif_exposure.std(axis=0, ddof=0)
    keep_cols = col_std[col_std > 0].index.tolist()
    dropped_cols = [c for c in topic_cols if c not in keep_cols]
    vif_exposure = vif_exposure[keep_cols].copy()

    # Optional subsample for speed
    if len(vif_exposure) > VIF_MAX_ROWS:
        vif_exposure = vif_exposure.sample(VIF_MAX_ROWS, random_state=42)

    # Standardize columns for numerical stability
    vif_exposure = (vif_exposure - vif_exposure.mean(axis=0)) / vif_exposure.std(axis=0, ddof=0)
    vif_exposure = vif_exposure.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    vif_rows = []
    X_vif = vif_exposure.to_numpy(dtype=float)
    vif_cols = vif_exposure.columns.tolist()

    for j, col in enumerate(vif_cols):
        vif_val = variance_inflation_factor(X_vif, j)
        vif_rows.append({"topic": col, "VIF": float(vif_val)})

    vif_results = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)

    out_dir = HERE / "Output"
    out_dir.mkdir(exist_ok=True)
    vif_csv = out_dir / f"topic_vif_{OUTPUT_TAG}.csv"
    vif_results.to_csv(vif_csv, index=False)

    print("Saved VIF results:", vif_csv)
    print("Top topics by VIF:")
    print(vif_results.head(VIF_PRINT_TOP_N))

    if dropped_cols:
        print("Dropped constant topics before VIF:", dropped_cols)


 # Helper: safe scaling for 1D arrays (standardize, avoid NaN if std=0)
def safe_scale_1d(x):
    """Standardize a 1D array; return zeros if std is 0 / NaN."""
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd

# --------------------------------------------------
# Forward stepwise topic selection helper
# --------------------------------------------------
def forward_stepwise_select_topics(panel_df, candidate_topics):
    """
    Forward stepwise selection using pooled panel data.
    Starts with controls-only model and iteratively adds the topic
    that increases R² the most until STEPWISE_MAX_TOPICS is reached.
    """

    selected = []
    remaining = list(candidate_topics)

    # Build control matrix once (match final regression controls)
    y = panel_df["cov_ij_q"].to_numpy(dtype=float)

    Xi_core_raw = np.column_stack([
        (panel_df["ln_assets_i"] * panel_df["ln_assets_j"]).to_numpy(dtype=float),
        (panel_df["cash_i"] * panel_df["cash_j"]).to_numpy(dtype=float),
        (panel_df["net_debt_i"] * panel_df["net_debt_j"]).to_numpy(dtype=float),
        (panel_df["ebit_margin_i"] * panel_df["ebit_margin_j"]).to_numpy(dtype=float),
    ])

    Xi_other_raw = np.column_stack([
        (panel_df["net_income_i"] * panel_df["net_income_j"]).to_numpy(dtype=float),
        (panel_df["equity_assets_i"] * panel_df["equity_assets_j"]).to_numpy(dtype=float),
        (panel_df["capex_5y_cagr_i"] * panel_df["capex_5y_cagr_j"]).to_numpy(dtype=float),
        (panel_df["green_revenue_pct_i"] * panel_df["green_revenue_pct_j"]).to_numpy(dtype=float),
    ])

    Xi_miss_raw = np.column_stack([
        (panel_df["net_income_i_miss"] * panel_df["net_income_j_miss"]).to_numpy(dtype=float),
        (panel_df["equity_assets_i_miss"] * panel_df["equity_assets_j_miss"]).to_numpy(dtype=float),
        (panel_df["capex_5y_cagr_i_miss"] * panel_df["capex_5y_cagr_j_miss"]).to_numpy(dtype=float),
        (panel_df["green_revenue_pct_i_miss"] * panel_df["green_revenue_pct_j_miss"]).to_numpy(dtype=float),
    ])

    Xi_raw = np.column_stack([Xi_core_raw, Xi_other_raw, Xi_miss_raw])

    valid = np.isfinite(y) & np.isfinite(Xi_raw).all(axis=1)
    y0 = y[valid]
    Xi0_raw = Xi_raw[valid]

    Xi0 = np.column_stack([
        safe_scale_1d(Xi0_raw[:, j]) for j in range(Xi0_raw.shape[1])
    ])

    X_base = sm.add_constant(Xi0)
    base_model = sm.OLS(y0, X_base).fit()
    current_r2 = float(base_model.rsquared)

    print(f"Stepwise baseline R2 (controls only): {current_r2:.6f}")

    while remaining and len(selected) < STEPWISE_MAX_TOPICS:
        best_topic = None
        best_r2 = -np.inf
        n_errors = 0
        n_nan_r2 = 0
        sample_errors = []

        for t in remaining:
            s = (panel_df[f"{t}_i"] * panel_df[f"{t}_j"]).to_numpy(dtype=float)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            s0 = safe_scale_1d(s[valid])

            if selected:
                S_prev = np.column_stack([
                    safe_scale_1d(
                        np.nan_to_num(
                            (panel_df[f"{k}_i"] * panel_df[f"{k}_j"]).to_numpy(dtype=float)[valid],
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                    )
                    for k in selected
                ])
                X = sm.add_constant(np.column_stack([Xi0, S_prev, s0]))
            else:
                X = sm.add_constant(np.column_stack([Xi0, s0]))

            try:
                m = sm.OLS(y0, X).fit()
                r2 = float(m.rsquared)
            except Exception as e:
                n_errors += 1
                if len(sample_errors) < 5:
                    sample_errors.append((t, repr(e)))
                continue

            if not np.isfinite(r2):
                n_nan_r2 += 1
                continue

            if r2 > best_r2:
                best_r2 = r2
                best_topic = t

        if best_topic is None:
            print(
                "Stepwise stopping: no usable topic remained. "
                f"errors={n_errors}, nan_r2={n_nan_r2}, remaining={len(remaining)}"
            )
            if sample_errors:
                print("Sample stepwise errors:", sample_errors)
            break

        gain = best_r2 - current_r2

        selected.append(best_topic)
        remaining.remove(best_topic)
        current_r2 = best_r2

        print(
            f"Stepwise ranked {best_topic} | new R2={current_r2:.6f} | gain={gain:.6f} | topics={len(selected)} "
            f"| errors={n_errors} | nan_r2={n_nan_r2}"
        )

    return selected

# ============================
# Step 3b: Forward stepwise topic selection on the pooled rolling-window panel
# ============================
if 'RUN_FORWARD_STEPWISE' in globals() and RUN_FORWARD_STEPWISE and not RUN_SINGLE_TOPIC:
    selected_topics = forward_stepwise_select_topics(panel, topic_cols)

    if not selected_topics:
        raise RuntimeError(
            "Forward stepwise selection returned 0 topics. "
            "Try lowering STEPWISE_MIN_DR2 or checking topic coverage."
        )

    topic_cols = selected_topics
    print(f"Forward stepwise selection completed. Keeping {len(topic_cols)} topics.")
    print("Selected topics:", topic_cols)

    out_dir = HERE / "Output"
    out_dir.mkdir(exist_ok=True)

    pd.DataFrame({"topic": topic_cols}).to_csv(
        out_dir / f"stepwise_selected_topics_{DYNAMIC_TF_RUN_LABEL}_{OUTPUT_TAG}.csv",
        index=False,
    )

    # Restrict the ranking map to selected topics only
    topic_year_mean_map = {
        (c, y): v
        for (c, y), v in topic_year_mean_map.items()
        if c in set(topic_cols)
    }

# ============================
# Step 4: Quarterly regressions (MARGINAL contribution, H&H-style)
# - Baseline: controls only
# - Full: controls + ALL topics (simultaneously)
# - Marginal contribution of topic k: R2(full_all) - R2(full_all_without_k)
# ============================


results_rows = []

# Safety: marginal drop-one is expensive if K is large
if len(topic_cols) > 50:
    print(f"WARNING: You selected {len(topic_cols)} topics. Drop-one marginal R2 will be slow (K+1 regressions per quarter).")

for q, g in panel.groupby("quarter", sort=True):
    y = g["cov_ij_q"].to_numpy(dtype=float)

    Xi_core_raw = np.column_stack([
        (g["ln_assets_i"] * g["ln_assets_j"]).to_numpy(dtype=float),
        (g["cash_i"] * g["cash_j"]).to_numpy(dtype=float),
        (g["net_debt_i"] * g["net_debt_j"]).to_numpy(dtype=float),
        (g["ebit_margin_i"] * g["ebit_margin_j"]).to_numpy(dtype=float),
    ])

    Xi_other_raw = np.column_stack([
        (g["net_income_i"] * g["net_income_j"]).to_numpy(dtype=float),
        (g["equity_assets_i"] * g["equity_assets_j"]).to_numpy(dtype=float),
        (g["capex_5y_cagr_i"] * g["capex_5y_cagr_j"]).to_numpy(dtype=float),
        (g["green_revenue_pct_i"] * g["green_revenue_pct_j"]).to_numpy(dtype=float),
    ])

    # Missingness dummies (products) for the imputed controls
    Xi_miss_raw = np.column_stack([
        (g["net_income_i_miss"] * g["net_income_j_miss"]).to_numpy(dtype=float),
        (g["equity_assets_i_miss"] * g["equity_assets_j_miss"]).to_numpy(dtype=float),
        (g["capex_5y_cagr_i_miss"] * g["capex_5y_cagr_j_miss"]).to_numpy(dtype=float),
        (g["green_revenue_pct_i_miss"] * g["green_revenue_pct_j_miss"]).to_numpy(dtype=float),
    ])

    Xi_raw = np.column_stack([Xi_core_raw, Xi_other_raw, Xi_miss_raw])

    # Require finite y and finite regressors before scaling
    valid_ctrl = np.isfinite(y) & np.isfinite(Xi_raw).all(axis=1)
    if valid_ctrl.sum() < MIN_VALID_PAIRS:
        continue

    y0 = y[valid_ctrl]
    Xi0_raw = Xi_raw[valid_ctrl]

    # Scale ALL control regressors quarter-by-quarter for numerical stability
    Xi0 = np.column_stack([
        safe_scale_1d(Xi0_raw[:, j]) for j in range(Xi0_raw.shape[1])
    ])

    # Controls-only baseline
    X_base = sm.add_constant(Xi0)
    base = sm.OLS(y0, X_base).fit()
    r2_controls = float(base.rsquared)

    # Topic pair exposures matrix S (NxK), fill missing with 0 (no exposure)
    S_cols = []
    for k in topic_cols:
        s = (g[f"{k}_i"] * g[f"{k}_j"]).to_numpy(dtype=float)
        S_cols.append(s)
    S_raw = np.column_stack(S_cols)
    S_raw = np.nan_to_num(S_raw, nan=0.0)

    S0_raw = S_raw[valid_ctrl]
    S0 = np.column_stack([
        safe_scale_1d(S0_raw[:, j]) for j in range(S0_raw.shape[1])
    ])

    # Full model with ALL topics simultaneously
    X_full_all = sm.add_constant(np.column_stack([Xi0, S0]))
    full_all = sm.OLS(y0, X_full_all).fit()
    r2_full_all = float(full_all.rsquared)

    # Indices of topic coefficients inside full_all params:
    # params = [const, ctrl1, ctrl2, ctrl3, topic1, topic2, ...]
    # Compute marginal contribution via drop-one R2
    for idx_k, k in enumerate(topic_cols):
        # Build X without topic k
        if S0.shape[1] == 1:
            # If only one topic, "minus k" becomes controls-only model
            r2_minus_k = r2_controls
        else:
            S_minus = np.delete(S0, idx_k, axis=1)
            X_minus = sm.add_constant(np.column_stack([Xi0, S_minus]))
            m_minus = sm.OLS(y0, X_minus).fit()
            r2_minus_k = float(m_minus.rsquared)

        dR2_marg = r2_full_all - r2_minus_k

        # Also record full-model coefficient + t-stat for topic k (conditional on other topics)
        params = np.asarray(full_all.params)
        tvals = np.asarray(full_all.tvalues)

        K = S0.shape[1]  # number of topics
        beta_topics = params[-K:]
        t_topics = tvals[-K:]

        beta_k = float(beta_topics[idx_k])
        t_k = float(t_topics[idx_k])

        results_rows.append({
            "quarter": q,
            "topic": k,
            "beta": float(beta_k),
            "t_stat": float(t_k),
            "R2_base": r2_controls,           # controls-only baseline R2
            "R2_full": r2_full_all,           # full model (controls + all topics) R2
            "R2_minus_k": r2_minus_k,         # R2 with topic k excluded
            "dR2": float(dR2_marg),           # marginal contribution (H&H concept)
            "n_pairs": int(valid_ctrl.sum()),
        })

results = pd.DataFrame(results_rows)

print(f"Final regression topic count after stepwise selection: {results['topic'].nunique()}")

# --------------------------------------------------
# Optional: Topic pruning based on average marginal contribution (mean dR2)
# This helps reduce the number of topics before running the final model.
# --------------------------------------------------
if PRUNE_TOPICS_BY_DR2 and not results.empty:
    topic_rank = (
        results.groupby("topic")["dR2"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="mean_dR2")
    )

    top_topics = topic_rank.head(TARGET_TOPIC_COUNT)["topic"].tolist()

    print("\nFull ranking of topics by average marginal contribution (mean dR2):")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(topic_rank)

    print("\nSuggested topics to KEEP (top", TARGET_TOPIC_COUNT, "):")
    print(top_topics)

    out_dir = HERE / "Output"
    out_dir.mkdir(exist_ok=True)

    topic_rank.to_csv(out_dir / f"topic_mean_dR2_ranking_{OUTPUT_TAG}.csv", index=False)

    with open(out_dir / f"suggested_topics_{OUTPUT_TAG}.txt", "w") as f:
        for t in top_topics:
            f.write(t + "\n")

    print("Saved topic ranking and suggested topic list to Output folder.")

    # Stop execution so the user can update TOPIC_INCLUDE with the suggested topics
    raise SystemExit(
        f"Topic pruning step completed. Update TOPIC_INCLUDE with the top {TARGET_TOPIC_COUNT} topics and rerun the script."
    )

if results.empty:
    raise RuntimeError("Regression results are empty. Check controls/text coverage and topic selection.")

# Diagnostics: show which years actually have regression output
results["year_q"] = results["quarter"].astype(str).str[:4].astype(int)
year_counts = results.groupby("year_q")["dR2"].size().sort_index()
print("Regression output rows by year (year_q -> n_rows):", year_counts.to_dict())
print("Available regression years:", year_counts.index.tolist())

# ============================
# Step 5: Baseline normalization
# ============================

#
#
# Topic-level baseline normalization (Hanley–Hoberg static approach):
# z_{q,k} = (dR2_{q,k} - mean_baseline_k) / std_baseline_k
baseline_start = DYNAMIC_BASELINE_START_YEAR if BASELINE_START_YEAR is None else BASELINE_START_YEAR
baseline_end = DYNAMIC_BASELINE_END_YEAR if BASELINE_END_YEAR is None else BASELINE_END_YEAR

baseline = results[
    results["year_q"].between(baseline_start, baseline_end)
].copy()

if baseline.empty:
    # Fallback: use the first 3 available years with any regression output
    avail_years = sorted(results["year_q"].unique().tolist())
    if not avail_years:
        raise ValueError(
            f"Baseline period produced 0 rows and no regression years are available. "
            f"Check MIN_VALID_PAIRS={MIN_VALID_PAIRS} and early-year data coverage."
        )
    fb_start = avail_years[0]
    fb_end = min(fb_start + 2, avail_years[-1])
    warnings.warn(
        f"Baseline period {BASELINE_START_YEAR}-{BASELINE_END_YEAR} produced 0 rows. "
        f"Falling back to baseline years {fb_start}-{fb_end} based on available regression output.",
        RuntimeWarning,
    )
    baseline = results[results["year_q"].between(fb_start, fb_end)].copy()

    if baseline.empty:
        raise ValueError(
            f"Fallback baseline years {fb_start}-{fb_end} still produced 0 rows. "
            f"Lower MIN_VALID_PAIRS (currently {MIN_VALID_PAIRS}) or choose a later baseline window."
        )

print(
    "Baseline window used (years):",
    int(baseline["year_q"].min()),
    "-",
    int(baseline["year_q"].max()),
    "; rows=",
    int(len(baseline)),
    "| target year =",
    DYNAMIC_TARGET_YEAR,
)

mu = baseline.groupby("topic")["dR2"].mean()
sd = baseline.groupby("topic")["dR2"].std(ddof=0)  # population std like a stable baseline
cnt = baseline.groupby("topic")["dR2"].count()

# Guardrails: require enough baseline observations.
# Instead of dropping topics with tiny baseline volatility (sd ~ 0),
# we apply a minimum sd to avoid z-scores becoming NaN.
MIN_BASELINE_OBS = 12          # ~3 years of quarters

# Keep sd only when we have enough baseline observations
sd = sd.where(cnt >= MIN_BASELINE_OBS, np.nan)

# Enforce a minimum standard deviation (prevents NaN / infinite z from near-constant dR2)
sd = sd.clip(lower=MIN_BASELINE_SD)

results["z"] = results["dR2"] - results["topic"].map(mu)

# Clean / stabilize z-scores for plotting (numeric safety only)
results = results.replace([np.inf, -np.inf], np.nan).copy()
results["z"] = results["z"].clip(-10, 1000)

# For plotting/aggregation we keep the same z (no activation weighting).
# If a topic cannot be standardized (sd is NaN), z will be NaN and we will plot it as 0.
results["z_plot"] = results["z"]

# --------------------------------------------------
# Target-year ranking: report the highest z-score topics in DYNAMIC_TARGET_YEAR
# --------------------------------------------------
target_year_results = results[results["year_q"] == DYNAMIC_TARGET_YEAR].copy()

if target_year_results.empty:
    warnings.warn(
        f"No regression rows found for target year {DYNAMIC_TARGET_YEAR}. "
        "Top z-score extraction will be empty.",
        RuntimeWarning,
    )
    top_target_year = pd.DataFrame(columns=["quarter", "topic", "z", "dR2", "beta", "t_stat", "n_pairs"])
else:
    top_target_year = (
        target_year_results
        .sort_values(["quarter", "z", "dR2"], ascending=[True, False, False])
        .groupby("quarter", as_index=False, group_keys=False)
        .head(TOP_Z_COUNT)
        .loc[:, [c for c in ["quarter", "topic", "z", "dR2", "beta", "t_stat", "n_pairs"] if c in target_year_results.columns]]
        .copy()
    )

print(f"Top {TOP_Z_COUNT} z-score topics PER QUARTER in target year {DYNAMIC_TARGET_YEAR}:")
print(top_target_year)

# --------------------------------------------------
# Target-year topic ranking aggregated across quarters
# This produces UNIQUE topics for the target year.
# --------------------------------------------------
if target_year_results.empty:
    top_target_year_topics = pd.DataFrame(
        columns=["topic", "z_mean", "z_max", "dR2_mean", "dR2_max", "beta_mean", "t_stat_mean", "quarters_present", "avg_n_pairs"]
    )
else:
    top_target_year_topics = (
        target_year_results
        .groupby("topic", as_index=False)
        .agg(
            z_mean=("z", "mean"),
            z_max=("z", "max"),
            dR2_mean=("dR2", "mean"),
            dR2_max=("dR2", "max"),
            beta_mean=("beta", "mean"),
            t_stat_mean=("t_stat", "mean"),
            quarters_present=("quarter", "nunique"),
            avg_n_pairs=("n_pairs", "mean"),
        )
        .sort_values(["z_mean", "z_max", "dR2_mean"], ascending=[False, False, False])
        .head(TOP_Z_COUNT)
        .copy()
    )

print(f"Top {TOP_Z_COUNT} UNIQUE topics in target year {DYNAMIC_TARGET_YEAR} (aggregated across quarters):")
print(top_target_year_topics)

# Diagnostic: how many quarters have valid z per topic (helps explain "empty" plots)
valid_counts = results.groupby("topic")["z_plot"].apply(lambda s: s.notna().sum()).sort_values()
print("Lowest valid z counts (topics):", valid_counts.head(5).to_dict())

# ============================
# Static risk model (aggregate): average z-score across topics per quarter
# ============================
static_z = (
    results.groupby("quarter", as_index=False)["z_plot"]
           .agg(z_mean="mean", z_median="median")
)

# Convert quarter to timestamp for plotting
static_z["quarter_ts"] = pd.PeriodIndex(static_z["quarter"].astype(str), freq="Q").to_timestamp()
static_z = static_z.sort_values("quarter_ts")

# 4-quarter rolling smoothing (1-year window) for presentation
static_z["z_mean_smooth4"] = static_z["z_mean"].rolling(window=4, min_periods=4).mean()
static_z["z_median_smooth4"] = static_z["z_median"].rolling(window=4, min_periods=4).mean()


out_dir = HERE / "Output"
out_dir.mkdir(exist_ok=True)

out_csv = out_dir / f"topic_dr2_zscores_controls_{OUTPUT_TAG}.csv"
results.to_csv(out_csv, index=False)
print("Saved:", out_csv)

top_z_csv = out_dir / f"top_z_topics_{DYNAMIC_TF_RUN_LABEL}_{OUTPUT_TAG}.csv"
top_target_year.to_csv(top_z_csv, index=False)
print("Saved target-year top z rows per quarter:", top_z_csv)

top_z_topic_csv = out_dir / f"top_z_topics_unique_{DYNAMIC_TF_RUN_LABEL}_{OUTPUT_TAG}.csv"
top_target_year_topics.to_csv(top_z_topic_csv, index=False)
print("Saved target-year top UNIQUE topics:", top_z_topic_csv)


# --------------------------------------------------
# Plot: Static risk model (average z-score across topics)
# --------------------------------------------------
try:
    # Plot window filter (graphs only)
    static_z_plot = static_z.copy()
    static_z_plot["year_plot"] = static_z_plot["quarter_ts"].dt.year
    static_z_plot = static_z_plot[(static_z_plot["year_plot"] >= PLOT_START_YEAR) & (static_z_plot["year_plot"] <= PLOT_END_YEAR)].copy()

    plt.figure(figsize=(14, 4))
    x = np.arange(len(static_z_plot))
    plt.bar(x, static_z_plot["z_mean_smooth4"].values)

    plt.title(STATIC_PLOT_TITLE)
    plt.xlabel("Quarter")
    plt.ylabel("Average z-score")

    # x-axis: show years only (one tick per year)
    years = static_z_plot["quarter_ts"].dt.year
    year_idx = years.ne(years.shift()).to_numpy().nonzero()[0]
    plt.xticks(year_idx, years.iloc[year_idx].astype(str).tolist(), rotation=0)

    plt.tight_layout()
    out_png = out_dir / f"static_risk_model_avg_z_{OUTPUT_TAG}.png"
    plt.savefig(out_png, dpi=200)
    print("Saved plot:", out_png)

    try:
        plt.show()
    except Exception:
        pass

    plt.close()
except Exception as e:
    print("WARNING: Failed to plot static risk model:", e)


# ============================
# Plot: z-scores over time for selected topics (one plot per topic)
# Keep full time axis even when z is NaN (inactive years)
# ============================

if PLOT_SINGLE_TOPIC:
    plot_topics = [TOPIC_SINGLE]
else:
    plot_topics = list(TOPIC_INCLUDE) if TOPIC_INCLUDE else list(topic_cols)
print("Plotting topics:", plot_topics)

# Full quarter timeline (so plots are not truncated)
all_quarters = (
    pd.PeriodIndex(results["quarter"].astype(str), freq="Q")
      .to_timestamp()
      .sort_values()
      .unique()
)
# Plot window filter (graphs only)
all_quarters = pd.to_datetime(all_quarters)
all_quarters = all_quarters[(all_quarters.year >= PLOT_START_YEAR) & (all_quarters.year <= PLOT_END_YEAR)]

plot_df = results.copy()
plot_df["quarter_ts"] = pd.PeriodIndex(plot_df["quarter"].astype(str), freq="Q").to_timestamp()

out_dir = HERE / "Output"
out_dir.mkdir(exist_ok=True)

for t in plot_topics:
    tmp = plot_df[plot_df["topic"] == t].copy()
    if tmp.empty:
        continue

    # Mean z per quarter for this topic and reindex to full timeline
    s = tmp.groupby("quarter_ts")["z_plot"].mean().reindex(all_quarters)

    # For plotting: keep full axis; inactive quarters show as 0-height bars
    z_plot = s.fillna(0.0)

    # 4-quarter rolling smoothing (presentation only)
    z_smooth4 = z_plot.rolling(window=4, min_periods=2).mean()

    plt.figure(figsize=(12, 4))
    x = np.arange(len(all_quarters))
    plt.bar(x, z_smooth4.values)

    plt.axhline(2, linestyle="--")
    plt.axhline(-2, linestyle="--")

    plt.title(TOPIC_PLOT_TITLE_TEMPLATE.format(topic=t))
    plt.xlabel("Year")
    plt.ylabel("z-score")

    # Year-only x-axis labels (one tick per year)
    years = pd.Series(all_quarters).dt.year
    year_idx = years.ne(years.shift()).to_numpy().nonzero()[0]
    plt.xticks(year_idx, years.iloc[year_idx].astype(str).tolist(), rotation=0)

    plt.tight_layout()

    safe_tag = t.replace("topic_loading_", "t")
    out_png = out_dir / f"z_scores_{safe_tag}_{OUTPUT_TAG}.png"
    plt.savefig(out_png, dpi=200)
    print("Saved plot:", out_png)
    plt.close()


# ============================================================
# DIAGNOSTICS / STATISTICS (paste at the END of the script)
# Requires: cov, tf, controls_firm_year, tf_firm_year, panel, results,
#           topic_cols, TOPIC_SINGLE, MIN_VALID_PAIRS, BASELINE_START_YEAR, BASELINE_END_YEAR
# ============================================================
if RUN_DIAGNOSTICS:
    def _pct(x: float) -> str:
        return f"{100*x:.2f}%"

    def _safe_desc(s: pd.Series, name: str) -> dict:
        s = pd.to_numeric(s, errors="coerce")
        out = {
            "name": name,
            "n_nonmissing": int(s.notna().sum()),
            "missing_share": float(s.isna().mean()),
            "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
            "std": float(s.std(skipna=True)) if s.notna().any() else np.nan,
            "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
            "p01": float(s.quantile(0.01)) if s.notna().any() else np.nan,
            "p05": float(s.quantile(0.05)) if s.notna().any() else np.nan,
            "p50": float(s.quantile(0.50)) if s.notna().any() else np.nan,
            "p95": float(s.quantile(0.95)) if s.notna().any() else np.nan,
            "p99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
            "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
        }
        return out

    def _print_df(title: str, df: pd.DataFrame, max_rows: int = 20):
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)
        if df is None or len(df) == 0:
            print("(empty)")
            return
        with pd.option_context(
            "display.max_rows", max_rows,
            "display.max_columns", 200,
            "display.width", 160
        ):
            print(df)

    print("\n\n" + "#" * 90)
    print("DIAGNOSTICS START")
    print("#" * 90)

    # ------------------------------------------------------------
    # 1) INPUT SHAPES + BASIC SANITY
    # ------------------------------------------------------------
    _print_df(
        "1) Dataset shapes",
        pd.DataFrame([
            {"dataset": "cov", "rows": len(cov), "cols": len(cov.columns)},
            {"dataset": "tf (raw)", "rows": len(tf), "cols": len(tf.columns)},
            {"dataset": "tf_firm_year", "rows": len(tf_firm_year), "cols": len(tf_firm_year.columns)},
            {"dataset": "controls_firm_year", "rows": len(controls_firm_year), "cols": len(controls_firm_year.columns)},
            {"dataset": "panel", "rows": len(panel), "cols": len(panel.columns)},
            {"dataset": "results", "rows": len(results), "cols": len(results.columns)},
        ])
    )

    # Check cov duplicates
    if set(["quarter", "i", "j"]).issubset(cov.columns):
        dup_share = float(cov.duplicated(subset=["quarter", "i", "j"]).mean())
        print("\n1b) Cov duplicate share on (quarter,i,j):", _pct(dup_share))
        if dup_share > 0:
            _print_df("1c) Example duplicates (first 20)", cov[cov.duplicated(["quarter","i","j"], keep=False)].head(20))

    # ------------------------------------------------------------
    # 2) IDENTIFIER / YEAR COVERAGE CHECKS
    # ------------------------------------------------------------
    if "quarter" in cov.columns:
        cov_year = cov["quarter"].astype(str).str[:4]
        cov_year = pd.to_numeric(cov_year, errors="coerce")
        print("\n2) Cov year min/max:", int(cov_year.min()), "-", int(cov_year.max()))
        print("2b) Cov unique quarters:", int(cov["quarter"].nunique()))

    if "year" in tf.columns:
        tf_year = pd.to_numeric(tf["year"], errors="coerce")
        print("\n2c) TF year min/max:", int(tf_year.min()), "-", int(tf_year.max()))

    if "year" in controls_firm_year.columns:
        cy = pd.to_numeric(controls_firm_year["year"], errors="coerce")
        print("\n2d) Controls year min/max:", int(cy.min()), "-", int(cy.max()))

    if set(["i","j"]).issubset(cov.columns):
        print("\n2e) Cov unique i:", cov["i"].nunique(), "| unique j:", cov["j"].nunique())
        print("2f) Example i:", cov["i"].dropna().astype(str).unique()[:10].tolist())

    # Panel merge match-rate: do we have any TF / controls on i/j?
    if "year_q" not in panel.columns and "quarter" in panel.columns:
        panel["year_q"] = panel["quarter"].astype(str).str[:4].astype(int)

    # TF match proxy: any topic col non-missing on each side
    tf_i_cols = [f"{k}_i" for k in topic_cols if f"{k}_i" in panel.columns]
    tf_j_cols = [f"{k}_j" for k in topic_cols if f"{k}_j" in panel.columns]

    if tf_i_cols:
        share_tf_i_any = float(panel[tf_i_cols].notna().any(axis=1).mean())
        share_tf_i_all = float(panel[tf_i_cols].notna().all(axis=1).mean())
        print("\n2g) TF coverage i-side: any-topic-nonmissing =", _pct(share_tf_i_any),
              "| all-topics-nonmissing =", _pct(share_tf_i_all))

    if tf_j_cols:
        share_tf_j_any = float(panel[tf_j_cols].notna().any(axis=1).mean())
        share_tf_j_all = float(panel[tf_j_cols].notna().all(axis=1).mean())
        print("2h) TF coverage j-side: any-topic-nonmissing =", _pct(share_tf_j_any),
              "| all-topics-nonmissing =", _pct(share_tf_j_all))

    # Controls coverage by year
    ctrl_cols_present = [c for c in ctrl_cols if c in panel.columns]
    if ctrl_cols_present:
        ctrl_cov_by_year = (
            panel.groupby("year_q")[ctrl_cols_present]
                 .apply(lambda d: float(d.notna().mean().mean()))
                 .rename("controls_nonmissing_share")
                 .reset_index()
        )
        _print_df("2i) Controls coverage by year (mean non-missing share)", ctrl_cov_by_year, max_rows=50)

    # ------------------------------------------------------------
    # 3) DISTRIBUTIONS: Y (cov_ij_q), CONTROLS, TOPIC LOADINGS, EXPOSURES
    # ------------------------------------------------------------
    if "cov_ij_q" in panel.columns:
        _print_df("3) Dependent variable distribution (panel.cov_ij_q)", pd.DataFrame([_safe_desc(panel["cov_ij_q"], "cov_ij_q")]))

    # Controls distributions (pair-level products are what enter regression)
    def _pairprod_desc(col_i: str, col_j: str, name: str):
        if col_i in panel.columns and col_j in panel.columns:
            s = pd.to_numeric(panel[col_i], errors="coerce") * pd.to_numeric(panel[col_j], errors="coerce")
            return _safe_desc(s, name)
        return None

    pairprod_stats = []
    pairprod_stats.append(_pairprod_desc("ln_assets_i","ln_assets_j","ln_assets_i*ln_assets_j"))
    pairprod_stats.append(_pairprod_desc("cash_i","cash_j","cash_i*cash_j"))
    pairprod_stats.append(_pairprod_desc("net_debt_i","net_debt_j","net_debt_i*net_debt_j"))
    pairprod_stats.append(_pairprod_desc("ebit_margin_i","ebit_margin_j","ebit_margin_i*ebit_margin_j"))
    pairprod_stats = [x for x in pairprod_stats if x is not None]
    if pairprod_stats:
        _print_df("3b) Pairwise control products distribution (enter regression)", pd.DataFrame(pairprod_stats), max_rows=50)

    # Topic non-zero shares (after firm-year filter)
    topic_nz = []
    for k in topic_cols:
        s = pd.to_numeric(tf_firm_year[k], errors="coerce").fillna(0.0)
        topic_nz.append({
            "topic": k,
            "nonzero_share": float((s != 0).mean()),
            "mean": float(s.mean()),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
        })
    topic_nz_df = pd.DataFrame(topic_nz).sort_values("nonzero_share")
    _print_df("3c) Topics with LOWEST non-zero share (after MIN_TOPIC_LOADING)", topic_nz_df.head(20))
    _print_df("3d) Topics with HIGHEST non-zero share (after MIN_TOPIC_LOADING)", topic_nz_df.tail(20))

    # Exposure distributions S = topic_i * topic_j for a few topics
    topics_to_check = []
    if "TOPIC_SINGLE" in globals() and TOPIC_SINGLE in topic_cols:
        topics_to_check.append(TOPIC_SINGLE)
    topics_to_check += topic_cols[:4]
    topics_to_check = list(dict.fromkeys(topics_to_check))[:6]

    exp_stats = []
    for k in topics_to_check:
        ki, kj = f"{k}_i", f"{k}_j"
        if ki in panel.columns and kj in panel.columns:
            s = pd.to_numeric(panel[ki], errors="coerce").fillna(0.0) * pd.to_numeric(panel[kj], errors="coerce").fillna(0.0)
            d = _safe_desc(s, f"S={k}_i*{k}_j")
            d["nonzero_share"] = float((s != 0).mean())
            exp_stats.append(d)
    if exp_stats:
        _print_df("3e) Example topic exposure distributions (S = topic_i*topic_j)", pd.DataFrame(exp_stats), max_rows=50)

    # ------------------------------------------------------------
    # 4) REGRESSION OUTPUT CHECKS
    # ------------------------------------------------------------
    _print_df(
        "4) Results basic stats",
        pd.DataFrame([{
            "unique_quarters": int(results["quarter"].nunique()) if "quarter" in results.columns else None,
            "unique_topics": int(results["topic"].nunique()) if "topic" in results.columns else None,
            "expected_topics": int(len(topic_cols)),
            "min_pairs_threshold": int(MIN_VALID_PAIRS),
        }])
    )

    # Rows-per-quarter should be ~ #topics
    if set(["quarter","topic"]).issubset(results.columns):
        rows_per_q = results.groupby("quarter")["topic"].size()
        _print_df("4b) Rows-per-quarter summary (should be close to #topics)", rows_per_q.describe().to_frame("rows_per_quarter"))
        bad_q = rows_per_q[rows_per_q != len(topic_cols)]
        if len(bad_q) > 0:
            _print_df("4c) Quarters with missing topic rows (first 30)", bad_q.head(30).to_frame("rows"))

    # n_pairs distribution + by quarter
    if "n_pairs" in results.columns:
        _print_df("4d) n_pairs distribution (across quarter-topic rows)", pd.DataFrame([_safe_desc(results["n_pairs"], "n_pairs")]))
        med_pairs_by_q = results.groupby("quarter")["n_pairs"].median().rename("median_n_pairs").reset_index()
        _print_df("4e) Median n_pairs by quarter (head)", med_pairs_by_q.head(20))
        _print_df("4f) Median n_pairs by quarter (tail)", med_pairs_by_q.tail(20))

    # R2 + key output distributions
    for col in ["R2_base","R2_full","R2_minus_k","dR2","beta","t_stat","z","z_plot"]:
        if col in results.columns:
            _print_df(f"4g) Distribution: {col}", pd.DataFrame([_safe_desc(results[col], col)]))

    # dR2 should be >= 0 in ideal nesting; small negatives can appear from numerics/collinearity
    if "dR2" in results.columns:
        neg_share = float((results["dR2"] < -1e-12).mean())
        tiny_neg = float(((results["dR2"] < 0) & (results["dR2"] >= -1e-6)).mean())
        print("\n4h) dR2 negative share:", _pct(neg_share), "| tiny negatives (>-1e-6):", _pct(tiny_neg))
        worst = results.sort_values("dR2").head(15)
        _print_df("4i) Most negative dR2 rows (first 15)", worst[["quarter","topic","dR2","R2_full","R2_minus_k","n_pairs"]])

    # Topic ranking by average contribution
    if set(["topic","dR2"]).issubset(results.columns):
        topic_rank = (
            results.groupby("topic")["dR2"]
                   .agg(mean="mean", median="median", p95=lambda s: float(pd.Series(s).quantile(0.95)))
                   .sort_values("mean", ascending=False)
                   .reset_index()
        )
        _print_df("4j) Top 20 topics by mean dR2", topic_rank.head(20))
        _print_df("4k) Bottom 20 topics by mean dR2", topic_rank.tail(20))

    # Single topic time-series (if used)
    if "TOPIC_SINGLE" in globals() and "topic" in results.columns and TOPIC_SINGLE in results["topic"].unique():
        single = results[results["topic"] == TOPIC_SINGLE].sort_values("quarter").copy()
        cols_show = [c for c in ["quarter","n_pairs","beta","t_stat","dR2","z_plot"] if c in single.columns]
        _print_df(f"4l) {TOPIC_SINGLE} time series (head)", single[cols_show].head(20))
        _print_df(f"4m) {TOPIC_SINGLE} time series (tail)", single[cols_show].tail(20))

    # Baseline window diagnostics: do we actually have enough obs per topic?
    if "year_q" in results.columns:
        base = results[results["year_q"].between(BASELINE_START_YEAR, BASELINE_END_YEAR)].copy()
        if len(base) > 0:
            base_counts = base.groupby("topic")["dR2"].count().rename("baseline_obs").reset_index()
            _print_df("4n) Baseline obs per topic (head)", base_counts.sort_values("baseline_obs").head(20))
            _print_df("4o) Baseline obs per topic (tail)", base_counts.sort_values("baseline_obs").tail(20))
        else:
            print("\n4n) Baseline window produced 0 rows (you already have fallback logic above).")

    print("\n" + "#" * 90)
    print("DIAGNOSTICS DONE ✅")
    print("#" * 90)