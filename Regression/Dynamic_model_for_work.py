# ===========================================================
# Importing packages
# ===========================================================
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
import warnings
import matplotlib.pyplot as plt

# ===========================================================
# GLOBAL SETTINGS
# ===========================================================
# This block controls the dynamic-regression run.
# Main workflow:
#   1) choose the rolling text window,
#   2) choose topic-selection mode,
#   3) choose diagnostics / optional pruning,
#   4) choose plotting / output behavior.

# --------------------------------------------------
# Rolling text window used in the dynamic model
# --------------------------------------------------
# The regression target year uses a rolling text window ending in this year.
# Example:
#   DYNAMIC_TARGET_YEAR = 2020
#   DYNAMIC_WINDOW_LENGTH = 5
# gives the extraction summary built from 2016-2020.
DYNAMIC_TARGET_YEAR = 2020
DYNAMIC_WINDOW_LENGTH = 5
DYNAMIC_WINDOW_START = DYNAMIC_TARGET_YEAR - DYNAMIC_WINDOW_LENGTH + 1
DYNAMIC_WINDOW_END = DYNAMIC_TARGET_YEAR
DYNAMIC_TF_RUN_LABEL = f"{DYNAMIC_WINDOW_START}_{DYNAMIC_WINDOW_END}"

# Baseline years used to normalize topic-level dR2 in the dynamic model.
# By default, the baseline is the full rolling window except the target year.
DYNAMIC_BASELINE_START_YEAR = DYNAMIC_WINDOW_START
DYNAMIC_BASELINE_END_YEAR = DYNAMIC_TARGET_YEAR - 1

# Number of top-z topics saved/reported for the target year.
TOP_Z_COUNT = 10

# --------------------------------------------------
# Output naming and single-topic mode
# --------------------------------------------------
# Short tag appended to output filenames.
# Example values: "run1", "gfc", "covid", "v2".
OUTPUT_TAG = "run1"

# If True, run the full regression / plotting workflow for one topic only.
# Useful for debugging a single topic time series.
RUN_SINGLE_TOPIC = False

# Topic used when RUN_SINGLE_TOPIC=True.
TOPIC_SINGLE = "topic_loading_204"

# Manual title used in the single-topic plot.
TOPIC_PLOT_TITLE_TEMPLATE = "Interest rate"

# If True, only generate the plot for TOPIC_SINGLE.
# If False, generate plots for all selected topics.
PLOT_SINGLE_TOPIC = True

# --------------------------------------------------
# Optional diagnostics and pruning
# --------------------------------------------------
# If True, run the large diagnostics/statistics block at the end of the script.
RUN_DIAGNOSTICS = False

# If True, compute VIFs for pairwise topic exposures to inspect multicollinearity.
RUN_VIF_TEST = False

# If True, rank topics by average marginal contribution (mean dR2),
# save the ranking, and stop so TOPIC_INCLUDE can be updated manually.
PRUNE_TOPICS_BY_DR2 = False
TARGET_TOPIC_COUNT = 35

# VIF settings
VIF_MAX_ROWS = 20000
VIF_PRINT_TOP_N = 30

# --------------------------------------------------
# Topic-loading and plotting settings
# --------------------------------------------------
# Minimum absolute firm-year topic loading.
# Values below this threshold are treated as zero before pair exposures are built.
MIN_TOPIC_LOADING = 0

# Minimum baseline standard deviation used in the z-score denominator.
# This avoids NaN / explosive z-scores when baseline dR2 is nearly constant.
MIN_BASELINE_SD = 0.00005

# Plotting range for exported topic plots.
PLOT_START_YEAR = 2005
PLOT_END_YEAR = 2024

# --------------------------------------------------
# Forward stepwise topic selection for the dynamic model
# --------------------------------------------------
# If True, topics are selected on the pooled rolling-window panel using
# forward stepwise improvement in model R2.
RUN_FORWARD_STEPWISE = True

# Maximum number of candidate topics entering stepwise selection.
# If more topics are available, the script first keeps those with the
# highest average absolute firm-year loading.
STEPWISE_MAX_CANDIDATES = 150

# Maximum number of topics kept after stepwise selection.
STEPWISE_MAX_TOPICS = 30

# Optional minimum dR2 gain required to keep adding topics.
# Currently not enforced when set to None.
STEPWISE_MIN_DR2 = None

# Topics to exclude manually when TOPIC_INCLUDE is not used.
TOPIC_EXCLUDE = []  # e.g. ["topic_loading_12"]

# Minimum number of valid firm pairs required to estimate a quarter-topic regression.
# For production runs, a higher threshold than 1 is usually preferable.
MIN_VALID_PAIRS = 1

# --------------------------------------------------
# Manual topic list
# --------------------------------------------------
# Explicit topic list for manual testing / locked specifications.
# This list is used when RUN_SINGLE_TOPIC=False and TOPIC_INCLUDE is non-empty.
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

# ============================
# Step 0: Locate and load the three required input files
# - covariance panel (quarter-pair level)
# - rolling-window textual-factor extraction summary
# - firm-year control variables
# ============================
HERE = Path(__file__).resolve().parent   # Folder containing this regression script
REPO_ROOT = HERE.parent                  # Project root: .../Emerging-Credit-Risk_1

# Covariance panel candidates
# We check both the local Output folder and the Regression/Output folder
# so the script still works if it is moved or run from a slightly different location.
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

# Rolling-window textual-factor extraction summary candidates.
# Preferred file name includes the explicit run label used in the text stage.
# Fallbacks are included to support older naming conventions.
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

# Firm-year controls candidates.
# These controls are merged later to the i and j firms in the pairwise covariance panel.
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

print("Loading covariance panel from:", COV_PATH)
print(f"Loading rolling-window extraction summary ({DYNAMIC_TF_RUN_LABEL}) from:", TF_PATH)
print("Loading controls from:", CTRL_PATH)

# Load the three raw input tables.
# - cov: quarterly pairwise covariance panel
# - tf: firm-year textual factor loadings from the rolling text window
# - controls_raw: raw firm-year financial controls from Excel
cov = pd.read_csv(COV_PATH)
tf = pd.read_csv(TF_PATH)
controls_raw = pd.read_excel(CTRL_PATH)

print(
    f"Dynamic text window in use: {DYNAMIC_WINDOW_START}-{DYNAMIC_WINDOW_END} "
    f"(target year = {DYNAMIC_TARGET_YEAR})"
)

# Compact textual-factor diagnostics.
# These help verify that the loaded extraction summary matches the intended rolling window.
try:
    tf["year"] = pd.to_numeric(tf["year"], errors="coerce")
    print("TF_PATH:", TF_PATH)
    print("TF year min/max:", int(tf["year"].min()), int(tf["year"].max()))
    print("TF rows:", int(len(tf)))
except Exception as _e:
    print("WARNING: TF diagnostics failed:", _e)


# -----------------------------
# Step 1: Prepare controls
# -----------------------------

controls_raw["company"] = controls_raw["company"].astype(str).str.strip().map(_norm_id)
controls_raw["year"] = controls_raw["year"].astype(int)

# Derived controls
controls_raw["ln_assets"] = np.log(controls_raw["total_assets"])
controls_raw["equity_assets"] = controls_raw["common_equity"] / controls_raw["total_assets"]

controls_firm_year = (
    controls_raw
    .groupby(["company","year"], as_index=False)[
        [
            "ln_assets",
            "cash",
            "net_debt",
            "ebit_margin",
            "net_income",
            "equity_assets",
            "capex_5y_cagr",
            "green_revenue_pct",
        ]
    ]
    .mean()
    .rename(columns={"company":"firm_id"})
)

# ============================
# Step 2: Prepare firm-year TFs
# ============================

# Collapse to one row per firm-year
tf_firm_year = tf.groupby(["bank", "year"], as_index=False)[topic_cols].mean()

# Normalize firm ids
tf_firm_year["bank"] = tf_firm_year["bank"].map(_norm_id)

# Ensure numeric topic loadings and apply noise filter
tf_firm_year[topic_cols] = tf_firm_year[topic_cols].apply(pd.to_numeric, errors="coerce")
tf_firm_year[topic_cols] = tf_firm_year[topic_cols].where(
    tf_firm_year[topic_cols].abs() >= MIN_TOPIC_LOADING, 0.0
)

# Optional prefilter before forward stepwise selection:
# keep the strongest topics by average absolute loading
if RUN_FORWARD_STEPWISE and len(topic_cols) > STEPWISE_MAX_CANDIDATES:    topic_cols = (
        tf_firm_year[topic_cols]
        .abs()
        .mean()
        .nlargest(STEPWISE_MAX_CANDIDATES)
        .index
        .tolist()
    )
    tf_firm_year = tf_firm_year[["bank", "year"] + topic_cols].copy()
    print(
        f"Stepwise candidate prefilter applied: keeping top {len(topic_cols)} topics "
        f"(max candidates = {STEPWISE_MAX_CANDIDATES})."
    )

# Yearly mean topic loading map: (topic, year) -> mean loading
topic_year_mean_map = {
    (topic, int(year)): float(value)
    for year, row in (
        tf_firm_year.groupby("year")[topic_cols].mean().iterrows()
    )
    for topic, value in row.items()
}


# ============================
# Step 3: Align TFs and controls to the quarterly covariance panel
# ============================
# Same-year alignment:
# annual report year t is assumed informative for quarters in year t.
# Therefore, quarter yyyyQx uses text/control information from year yyyy.
cov["q_year"] = cov["quarter"].astype(str).str[:4].astype(int)
cov["lag_year"] = cov["q_year"]

# Normalize firm identifiers so they match the firm-year text and control tables.
cov["i"] = cov["i"].astype(str).str.strip().map(_norm_id)
cov["j"] = cov["j"].astype(str).str.strip().map(_norm_id)

print("Cov sample i after normalization:", cov["i"].dropna().astype(str).unique()[:10].tolist())
print("Cov sample j after normalization:", cov["j"].dropna().astype(str).unique()[:10].tolist())

# Merge textual factors for i and j.
panel = cov.merge(
    tf_firm_year,
    left_on=["i", "lag_year"],
    right_on=["bank", "year"],
    how="left"
)

panel = panel.merge(
    tf_firm_year,
    left_on=["j", "lag_year"],
    right_on=["bank", "year"],
    how="left",
    suffixes=("_i", "_j")
)

# Merge firm-year controls for i and j.
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

# Rename control columns to consistent *_i / *_j names.
control_bases = [
    "ln_assets",
    "cash",
    "net_debt",
    "ebit_margin",
    "net_income",
    "equity_assets",
    "capex_5y_cagr",
    "green_revenue_pct",
]
panel = panel.rename(columns={
    **{f"{c}_i_ctrl": f"{c}_i" for c in control_bases},
    **{f"{c}_j_ctrl": f"{c}_j" for c in control_bases},
})

ctrl_cols = [f"{c}_{side}" for c in control_bases for side in ["i", "j"]]
missing_ctrl_cols = [c for c in ctrl_cols if c not in panel.columns]
if missing_ctrl_cols:
    raise ValueError(
        f"Expected control columns missing after merge/rename: {missing_ctrl_cols}. "
        "This indicates the controls merge did not match i/j identifiers."
    )

print("Controls coverage (share non-missing, overall):", float(panel[ctrl_cols].notna().mean().mean()))
print("Controls match rate i-side:", float(panel["ln_assets_i"].notna().mean()))
print("Controls match rate j-side:", float(panel["ln_assets_j"].notna().mean()))

# --------------------------------------------------
# Control handling: require a small core set, impute the remaining controls,
# and add missingness dummies for the imputed variables.
# --------------------------------------------------
CORE_CTRL_COLS = [
    "ln_assets_i", "ln_assets_j",
    "ebit_margin_i", "ebit_margin_j",
]
IMPUTE_CTRL_COLS = [c for c in ctrl_cols if c not in CORE_CTRL_COLS]

for c in IMPUTE_CTRL_COLS:
    panel[f"{c}_miss"] = panel[c].isna().astype(int)
    year_median = panel.groupby("lag_year")[c].transform("median")
    global_median = float(panel[c].median(skipna=True)) if panel[c].notna().any() else 0.0
    panel[c] = panel[c].fillna(year_median).fillna(global_median)

# Restrict the regression sample to the rolling text window only.
panel["year_q"] = panel["quarter"].astype(str).str[:4].astype(int)
panel = panel[panel["lag_year"].between(DYNAMIC_WINDOW_START, DYNAMIC_WINDOW_END)].copy()

print(
    f"Restricted panel to rolling text window years {DYNAMIC_WINDOW_START}-{DYNAMIC_WINDOW_END}. "
    f"Remaining rows: {len(panel)}"
)

row_ok_core = panel[CORE_CTRL_COLS].notna().all(axis=1)
row_ok_all = panel[ctrl_cols].notna().all(axis=1)

print("Valid rows by year (CORE controls complete):")
print(panel.loc[row_ok_core].groupby("year_q").size().head(10))

print("Valid rows by year (ALL 8 controls complete):")
print(panel.loc[row_ok_all].groupby("year_q").size().head(10))

print("Total rows by year:")
print(panel.groupby("year_q").size().head(10))


# --------------------------------------------------
# Step 4: Forward stepwise topic selection helper
# --------------------------------------------------
def forward_stepwise_select_topics(panel_df, candidate_topics):
    """
    Forward stepwise selection on the pooled rolling-window panel.
    Starts from the controls-only model and iteratively adds the topic
    with the largest increase in pooled R² until STEPWISE_MAX_TOPICS is reached.
    """

    selected = []
    remaining = list(candidate_topics)

    # Dependent variable
    y = panel_df["cov_ij_q"].to_numpy(dtype=float)

    # Pairwise controls used in the final regression
    Xi_raw = np.column_stack([
        (panel_df["ln_assets_i"] * panel_df["ln_assets_j"]).to_numpy(dtype=float),
        (panel_df["cash_i"] * panel_df["cash_j"]).to_numpy(dtype=float),
        (panel_df["net_debt_i"] * panel_df["net_debt_j"]).to_numpy(dtype=float),
        (panel_df["ebit_margin_i"] * panel_df["ebit_margin_j"]).to_numpy(dtype=float),
        (panel_df["net_income_i"] * panel_df["net_income_j"]).to_numpy(dtype=float),
        (panel_df["equity_assets_i"] * panel_df["equity_assets_j"]).to_numpy(dtype=float),
        (panel_df["capex_5y_cagr_i"] * panel_df["capex_5y_cagr_j"]).to_numpy(dtype=float),
        (panel_df["green_revenue_pct_i"] * panel_df["green_revenue_pct_j"]).to_numpy(dtype=float),
        (panel_df["net_income_i_miss"] * panel_df["net_income_j_miss"]).to_numpy(dtype=float),
        (panel_df["equity_assets_i_miss"] * panel_df["equity_assets_j_miss"]).to_numpy(dtype=float),
        (panel_df["capex_5y_cagr_i_miss"] * panel_df["capex_5y_cagr_j_miss"]).to_numpy(dtype=float),
        (panel_df["green_revenue_pct_i_miss"] * panel_df["green_revenue_pct_j_miss"]).to_numpy(dtype=float),
    ])

    valid = np.isfinite(y) & np.isfinite(Xi_raw).all(axis=1)
    y0 = y[valid]
    Xi0_raw = Xi_raw[valid]
    Xi0 = np.column_stack([safe_scale_1d(Xi0_raw[:, j]) for j in range(Xi0_raw.shape[1])])

    # Precompute all candidate topic pair exposures once.
    S_raw = np.column_stack([
        np.nan_to_num(
            (panel_df[f"{t}_i"] * panel_df[f"{t}_j"]).to_numpy(dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for t in candidate_topics
    ])
    S0 = np.column_stack([safe_scale_1d(S_raw[valid, j]) for j in range(S_raw.shape[1])])
    topic_to_idx = {t: j for j, t in enumerate(candidate_topics)}

    # Controls-only baseline
    base_model = sm.OLS(y0, sm.add_constant(Xi0)).fit()
    current_r2 = float(base_model.rsquared)
    print(f"Stepwise baseline R2 (controls only): {current_r2:.6f}")

    while remaining and len(selected) < STEPWISE_MAX_TOPICS:
        best_topic = None
        best_r2 = -np.inf
        n_errors = 0
        n_nan_r2 = 0
        sample_errors = []

        selected_idx = [topic_to_idx[t] for t in selected]
        S_prev = S0[:, selected_idx] if selected_idx else None

        for t in remaining:
            s0 = S0[:, topic_to_idx[t]]

            if S_prev is None:
                X = sm.add_constant(np.column_stack([Xi0, s0]))
            else:
                X = sm.add_constant(np.column_stack([Xi0, S_prev, s0]))

            try:
                r2 = float(sm.OLS(y0, X).fit().rsquared)
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
        if STEPWISE_MIN_DR2 is not None and gain < STEPWISE_MIN_DR2:
            print(
                f"Stepwise stopping: best gain {gain:.6f} is below "
                f"STEPWISE_MIN_DR2={STEPWISE_MIN_DR2:.6f}."
            )
            break

        selected.append(best_topic)
        remaining.remove(best_topic)
        current_r2 = best_r2

        print(
            f"Stepwise ranked {best_topic} | new R2={current_r2:.6f} | gain={gain:.6f} | "
            f"topics={len(selected)} | errors={n_errors} | nan_r2={n_nan_r2}"
        )

    return selected

# ============================
# Step 4b: Forward stepwise topic selection on the pooled rolling-window panel
# ============================
if RUN_FORWARD_STEPWISE:
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

    # Restrict the ranking map to selected topics only.
    topic_year_mean_map = {
        (c, y): v
        for (c, y), v in topic_year_mean_map.items()
        if c in set(topic_cols)
    }

# ============================
# Step 5: Quarterly regressions (MARGINAL contribution, H&H-style)
# - Baseline: controls only
# - Full: controls + ALL topics (simultaneously)
# - Marginal contribution of topic k: R2(full_all) - R2(full_all_without_k)
# ============================

results_rows = []

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

    # Topic pair exposures matrix S (N x K); missing values imply no exposure.
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

    # Topic coefficients/t-stats from the full model.
    # params are ordered as: [const, controls..., topics...]
    K = S0.shape[1]
    params = np.asarray(full_all.params)
    tvals = np.asarray(full_all.tvalues)
    beta_topics = params[-K:]
    t_topics = tvals[-K:]

    # Marginal contribution of topic k: R2(full model) - R2(full model without k)
    for idx_k, k in enumerate(topic_cols):
        if K == 1:
            # With only one topic, excluding k leaves the controls-only model.
            r2_minus_k = r2_controls
        else:
            S_minus = np.delete(S0, idx_k, axis=1)
            X_minus = sm.add_constant(np.column_stack([Xi0, S_minus]))
            r2_minus_k = float(sm.OLS(y0, X_minus).fit().rsquared)

        dR2_marg = r2_full_all - r2_minus_k
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
# Saves a ranked topic list and stops, so TOPIC_INCLUDE can be updated manually.
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
# Step 6: Baseline normalization
# ============================

baseline_start = DYNAMIC_BASELINE_START_YEAR
baseline_end = DYNAMIC_BASELINE_END_YEAR

baseline = results[
    results["year_q"].between(baseline_start, baseline_end)
].copy()

if baseline.empty:
    avail_years = sorted(results["year_q"].unique().tolist())
    if not avail_years:
        raise ValueError(
            f"Baseline period produced 0 rows and no regression years are available. "
            f"Check MIN_VALID_PAIRS={MIN_VALID_PAIRS} and early-year data coverage."
        )
    fb_start = avail_years[0]
    fb_end = min(fb_start + 2, avail_years[-1])
    warnings.warn(
        f"Baseline period {baseline_start}-{baseline_end} produced 0 rows. "
        f"Falling back to baseline years {fb_start}-{fb_end}.",
        RuntimeWarning,
    )
    baseline = results[results["year_q"].between(fb_start, fb_end)].copy()

    if baseline.empty:
        raise ValueError(
            f"Fallback baseline years {fb_start}-{fb_end} still produced 0 rows."
        )

mu = baseline.groupby("topic")["dR2"].mean()
sd = baseline.groupby("topic")["dR2"].std(ddof=0)
cnt = baseline.groupby("topic")["dR2"].count()

MIN_BASELINE_OBS = 12
sd = sd.where(cnt >= MIN_BASELINE_OBS, np.nan)
sd = sd.clip(lower=MIN_BASELINE_SD)

results["z"] = (
    results["dR2"] - results["topic"].map(mu)
) / results["topic"].map(sd)

results = results.replace([np.inf, -np.inf], np.nan).copy()
results["z"] = results["z"].clip(-10, 1000)
results["z_plot"] = results["z"]


# ============================================================
# DIAGNOSTICS / STATISTICS
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