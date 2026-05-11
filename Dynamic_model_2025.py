import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# ============================================================
# Helper functions
# ============================================================
# These helper functions support the data preparation and regression steps.
# _norm_id standardizes firm identifiers for merging, while safe_scale_1d standardizes
# numeric arrays and avoids division-by-zero problems when a variable has no variation.
def first_existing(paths, label):
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Could not find {label}. Tried:\n" + "\n".join(map(str, paths)))
    return path

def _norm_id(x):
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().split())

def safe_scale_1d(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


# ============================================================
# Paths
# ============================================================
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# ============================================================
# Dynamic text window settings
# ============================================================
TEXT_TARGET_YEAR = 2024
OUTPUT_TARGET_YEAR = 2025
DYNAMIC_WINDOW_LENGTH = 5

DYNAMIC_WINDOW_START = TEXT_TARGET_YEAR - DYNAMIC_WINDOW_LENGTH + 1
DYNAMIC_WINDOW_END = TEXT_TARGET_YEAR
DYNAMIC_TF_RUN_LABEL = f"{DYNAMIC_WINDOW_START}_{DYNAMIC_WINDOW_END}"

# Baseline period used to standardize z-scores
DYNAMIC_BASELINE_START_YEAR = DYNAMIC_WINDOW_START
DYNAMIC_BASELINE_END_YEAR = TEXT_TARGET_YEAR - 1

# Number of top topics reported in output tables
TOP_Z_COUNT = 10

# ============================================================
# Output and model settings
# ============================================================
OUTPUT_TAG = "run1"
MIN_BASELINE_OBS = 12
MIN_BASELINE_SD = 1e-8

# Minimum absolute firm-year topic loading retained in the panel
MIN_TOPIC_LOADING = 1

# --------------------------------------------------
# Forward stepwise topic selection
# --------------------------------------------------
STEPWISE_MAX_CANDIDATES = 150
STEPWISE_MAX_TOPICS = 40
TOPIC_EXCLUDE = []

# Minimum number of valid pairs required in a quarter
MIN_VALID_PAIRS = 1

TOPIC_INCLUDE = [
    "topic_loading_14",
    "topic_loading_16",
    "topic_loading_18",
    "topic_loading_23",
    "topic_loading_29",
    "topic_loading_33",
    "topic_loading_37",
    "topic_loading_42",
    "topic_loading_49",
    "topic_loading_54",
    "topic_loading_56",
    "topic_loading_57",
    "topic_loading_58",
    "topic_loading_70",
    "topic_loading_82",
    "topic_loading_94",
    "topic_loading_97",
    "topic_loading_98",
    "topic_loading_110",
    "topic_loading_113",
    "topic_loading_117",
    "topic_loading_121",
    "topic_loading_137",
    "topic_loading_139",
    "topic_loading_142",
    "topic_loading_149",
    "topic_loading_157",
    "topic_loading_163",
    "topic_loading_164",
    "topic_loading_166",
    "topic_loading_170",
    "topic_loading_172",
    "topic_loading_185",
    "topic_loading_193",
    "topic_loading_202",
    "topic_loading_207",
    "topic_loading_216",
    "topic_loading_220",
    "topic_loading_223",
    "topic_loading_235",
    "topic_loading_245",
    "topic_loading_247",
    "topic_loading_256",
    "topic_loading_261",
    "topic_loading_269",
    "topic_loading_270",
    "topic_loading_271",
    "topic_loading_274",
    "topic_loading_276",
    "topic_loading_277",
    "topic_loading_290",
    "topic_loading_291",
    "topic_loading_303",
    "topic_loading_307",
    "topic_loading_333",
    "topic_loading_349",
    "topic_loading_404",
    "topic_loading_439",
    "topic_loading_468",
    "topic_loading_482",
    "topic_loading_548",
    "topic_loading_631",
    "topic_loading_659",
    "topic_loading_877",
]

# ============================
# Step 0: Load data
# ============================
# This step loads the covariance, text factor, and control variable datasets used in the dynamic model.
# The control variables are standardized and collapsed to firm-year level, while the relevant topic
# columns are selected from the rolling text factor file.
COV_PATH = first_existing([
    HERE / "Output" / "quarterly_pairwise_covariance_2025.csv",
    REPO_ROOT / "Regression" / "Output" / "quarterly_pairwise_covariance_2025.csv",
], "quarterly_pairwise_covariance_2025.csv")

TF_PATH = first_existing([
    REPO_ROOT / "Text analytics" / "outputs_textual_factors" / f"extraction_summary_{DYNAMIC_TF_RUN_LABEL}_V1.csv",
    REPO_ROOT / "Text analytics" / "Scripts" / "outputs_textual_factors" / f"extraction_summary_{DYNAMIC_TF_RUN_LABEL}_V1.csv",
], f"rolling extraction summary for {DYNAMIC_TF_RUN_LABEL}")

CTRL_PATH = first_existing([
    HERE / "Data" / "Control_variable_final.xlsx",
    REPO_ROOT / "Regression" / "Data" / "Control_variable_final.xlsx",
], "Control_variable_final.xlsx")

cov = pd.read_csv(COV_PATH)
tf = pd.read_csv(TF_PATH)
controls_raw = pd.read_excel(CTRL_PATH)

controls_raw.columns = [str(c).strip().lower() for c in controls_raw.columns]
controls_raw = controls_raw.rename(columns={
    "company name": "company",
    "ticker name": "ticker",
    "stock code": "ticker",
    "year date": "year",
    "date": "year",
    "total assets": "total_assets",
    "company market capitalization": "market_cap",
    "market cap": "market_cap",
    "market capitalization": "market_cap",
    "capital expenditures - total, 5 yr cagr": "capex_5y_cagr",
    "net debt - mean": "net_debt",
    "net debt": "net_debt",
    "cash & cash equivalents - total": "cash",
    "cash & cash equivalents": "cash",
    "cash and cash equivalents": "cash",
    "ebit margin - %": "ebit_margin",
    "ebit margin": "ebit_margin",
    "net income after tax": "net_income",
    "net income": "net_income",
    "company green revenue percentage": "green_revenue_pct",
    "common equity - total": "common_equity",
    "common equity": "common_equity",
})

required_controls = [
    "company", "year", "total_assets", "common_equity", "cash", "net_debt",
    "ebit_margin", "net_income", "capex_5y_cagr", "green_revenue_pct"
]
missing_controls = [c for c in required_controls if c not in controls_raw.columns]
if missing_controls:
    raise ValueError(f"Missing required control columns: {missing_controls}")

controls_raw["company"] = controls_raw["company"].map(_norm_id)
controls_raw["year"] = controls_raw["year"].astype(int)

controls_raw["ln_assets"] = np.log(controls_raw["total_assets"].where(controls_raw["total_assets"] > 0))
controls_raw["equity_assets"] = (
    controls_raw["common_equity"] / controls_raw["total_assets"].replace({0: np.nan})
)

need = [
    "ln_assets", "cash", "net_debt", "ebit_margin",
    "net_income", "equity_assets", "capex_5y_cagr", "green_revenue_pct"
]

controls_firm_year = (
    controls_raw.groupby(["company", "year"], as_index=False)[need]
    .mean()
    .rename(columns={"company": "firm_id"})
)
controls_firm_year["firm_id"] = controls_firm_year["firm_id"].map(_norm_id)

all_topic_cols = [c for c in tf.columns if c.startswith("topic_loading_")]

if TOPIC_INCLUDE:
    missing = [c for c in TOPIC_INCLUDE if c not in all_topic_cols]
    if missing:
        raise ValueError(f"TOPIC_INCLUDE contains missing columns: {missing}")
    topic_cols = list(TOPIC_INCLUDE)
else:
    topic_cols = [c for c in all_topic_cols if c not in set(TOPIC_EXCLUDE)]

# ============================
# Step 1: Prepare firm-year TFs
# ============================
# This step prepares the firm-year topic exposures used in the dynamic model.
# Topic loadings are converted to numeric values, small loadings are treated as zero to reduce noise,
# firm identifiers are standardized for merging, and the topic pool is prefiltered if it exceeds the
# maximum number of stepwise candidates.
tf_firm_year = tf[["bank", "year"] + topic_cols].copy()

for c in topic_cols:
    tf_firm_year[c] = pd.to_numeric(tf_firm_year[c], errors="coerce")
    tf_firm_year[c] = tf_firm_year[c].where(tf_firm_year[c].abs() >= MIN_TOPIC_LOADING, 0.0)

tf_firm_year["bank"] = tf_firm_year["bank"].map(_norm_id)

if len(topic_cols) > STEPWISE_MAX_CANDIDATES:
    topic_strength = tf_firm_year[topic_cols].abs().mean(axis=0).sort_values(ascending=False)
    topic_cols = topic_strength.head(STEPWISE_MAX_CANDIDATES).index.tolist()
    tf_firm_year = tf_firm_year[["bank", "year"] + topic_cols].copy()

# ============================
# Step 2: Align TFs and controls to quarters
# ============================
# This step links quarterly covariance observations to lagged firm-year information.
# First, the quarter is converted into a calendar year and shifted back by one year to implement
# the t-1 alignment. The firm identifiers in the covariance panel are then standardized, after
# which lagged textual factor exposures and firm-year controls are merged in for both firms in
# each pair. Finally, the panel is restricted to observations that fall within the selected
# rolling text window, and the full set of pair-level control variables is defined for later use
# in the regression framework.
cov["q_year"] = cov["quarter"].astype(str).str[:4].astype(int)
cov["lag_year"] = cov["q_year"] - 1

cov["i"] = cov["i"].map(_norm_id)
cov["j"] = cov["j"].map(_norm_id)

panel = cov.merge(
    tf_firm_year, left_on=["i", "lag_year"], right_on=["bank", "year"], how="left"
).merge(
    tf_firm_year, left_on=["j", "lag_year"], right_on=["bank", "year"],
    how="left", suffixes=("_i", "_j")
)

panel = panel.merge(
    controls_firm_year,
    left_on=["i", "lag_year"],
    right_on=["firm_id", "year"],
    how="left"
).drop(columns=["firm_id", "year"]).merge(
    controls_firm_year,
    left_on=["j", "lag_year"],
    right_on=["firm_id", "year"],
    how="left",
    suffixes=("_i", "_j")
).drop(columns=["firm_id", "year"])

panel = panel[
    panel["lag_year"].between(DYNAMIC_WINDOW_START, DYNAMIC_WINDOW_END)
].copy()

ctrl_cols = [
    "ln_assets_i", "ln_assets_j",
    "cash_i", "cash_j",
    "net_debt_i", "net_debt_j",
    "ebit_margin_i", "ebit_margin_j",
    "net_income_i", "net_income_j",
    "equity_assets_i", "equity_assets_j",
    "capex_5y_cagr_i", "capex_5y_cagr_j",
    "green_revenue_pct_i", "green_revenue_pct_j",
]

missing = [c for c in ctrl_cols if c not in panel.columns]
if missing:
    raise ValueError(f"Missing control columns after merge: {missing}")

# --------------------------------------------------
# Step 3: Forward stepwise topic selection
# --------------------------------------------------
# This function selects the most relevant topics for the dynamic model using a forward stepwise procedure.
# Starting from a controls-only specification, topics are added one at a time according to the increase
# in explanatory power they provide. The process continues until no further topic can be added or the
# maximum number of selected topics is reached.
def forward_stepwise_select_topics(panel_df, candidate_topics):
    selected = []
    remaining = list(candidate_topics)

    y = panel_df["cov_ij_q"].to_numpy(dtype=float)

    X_ctrl_raw = np.column_stack([
        (panel_df["ln_assets_i"] * panel_df["ln_assets_j"]).to_numpy(dtype=float),
        (panel_df["cash_i"] * panel_df["cash_j"]).to_numpy(dtype=float),
        (panel_df["net_debt_i"] * panel_df["net_debt_j"]).to_numpy(dtype=float),
        (panel_df["ebit_margin_i"] * panel_df["ebit_margin_j"]).to_numpy(dtype=float),
        (panel_df["net_income_i"] * panel_df["net_income_j"]).to_numpy(dtype=float),
        (panel_df["equity_assets_i"] * panel_df["equity_assets_j"]).to_numpy(dtype=float),
        (panel_df["capex_5y_cagr_i"] * panel_df["capex_5y_cagr_j"]).to_numpy(dtype=float),
        (panel_df["green_revenue_pct_i"] * panel_df["green_revenue_pct_j"]).to_numpy(dtype=float),
    ])

    valid = np.isfinite(y) & np.isfinite(X_ctrl_raw).all(axis=1)
    y0 = y[valid]
    X_ctrl = np.column_stack([safe_scale_1d(X_ctrl_raw[valid, j]) for j in range(X_ctrl_raw.shape[1])])

    current_r2 = float(sm.OLS(y0, sm.add_constant(X_ctrl)).fit().rsquared)

    while remaining and len(selected) < STEPWISE_MAX_TOPICS:
        best_topic = None
        best_r2 = -np.inf

        for t in remaining:
            s = np.nan_to_num(
                (panel_df[f"{t}_i"] * panel_df[f"{t}_j"]).to_numpy(dtype=float),
                nan=0.0, posinf=0.0, neginf=0.0
            )
            s0 = safe_scale_1d(s[valid])

            if selected:
                S_prev = np.column_stack([
                    safe_scale_1d(np.nan_to_num(
                        (panel_df[f"{k}_i"] * panel_df[f"{k}_j"]).to_numpy(dtype=float)[valid],
                        nan=0.0, posinf=0.0, neginf=0.0
                    ))
                    for k in selected
                ])
                X = sm.add_constant(np.column_stack([X_ctrl, S_prev, s0]))
            else:
                X = sm.add_constant(np.column_stack([X_ctrl, s0]))

            try:
                r2 = float(sm.OLS(y0, X).fit().rsquared)
            except Exception:
                continue

            if np.isfinite(r2) and r2 > best_r2:
                best_r2 = r2
                best_topic = t

        if best_topic is None:
            break

        selected.append(best_topic)
        remaining.remove(best_topic)
        current_r2 = best_r2

    return selected

# This step applies forward stepwise selection to the merged panel and stores the final set of
# topics used in the dynamic specification.
topic_cols = forward_stepwise_select_topics(panel, topic_cols)

if not topic_cols:
    raise RuntimeError("Forward stepwise selection returned 0 topics.")

out_dir = HERE / "Output"
out_dir.mkdir(exist_ok=True)

pd.DataFrame({"topic": topic_cols}).to_csv(
    out_dir / f"stepwise_selected_topics_{DYNAMIC_TF_RUN_LABEL}_{OUTPUT_TAG}.csv",
    index=False,
)

# ============================
# Step 4: Quarterly regressions
# ============================
# This step runs the regression framework quarter by quarter using the selected topics and the full
# set of pairwise control variables. For each quarter, it first estimates a controls-only model and
# then a full model that includes all selected topic exposures. It subsequently removes each topic
# one at a time, re-estimates the model, and records how much explanatory power is lost when that
# topic is excluded. The resulting output is a quarter-topic table containing coefficients, t-statistics,
# baseline and full-model R² values, drop-one R² values, marginal dR² contributions, and the number
# of firm pairs used in each quarterly regression.
results_rows = []

for q, g in panel.groupby("quarter", sort=True):
    y = g["cov_ij_q"].to_numpy(dtype=float)

    X_ctrl_raw = np.column_stack([
        (g["ln_assets_i"] * g["ln_assets_j"]).to_numpy(dtype=float),
        (g["cash_i"] * g["cash_j"]).to_numpy(dtype=float),
        (g["net_debt_i"] * g["net_debt_j"]).to_numpy(dtype=float),
        (g["ebit_margin_i"] * g["ebit_margin_j"]).to_numpy(dtype=float),
        (g["net_income_i"] * g["net_income_j"]).to_numpy(dtype=float),
        (g["equity_assets_i"] * g["equity_assets_j"]).to_numpy(dtype=float),
        (g["capex_5y_cagr_i"] * g["capex_5y_cagr_j"]).to_numpy(dtype=float),
        (g["green_revenue_pct_i"] * g["green_revenue_pct_j"]).to_numpy(dtype=float),
    ])

    valid = np.isfinite(y) & np.isfinite(X_ctrl_raw).all(axis=1)
    if valid.sum() < MIN_VALID_PAIRS:
        continue

    y0 = y[valid]
    X_ctrl = np.column_stack([safe_scale_1d(X_ctrl_raw[valid, j]) for j in range(X_ctrl_raw.shape[1])])

    base = sm.OLS(y0, sm.add_constant(X_ctrl)).fit()
    r2_base = float(base.rsquared)

    S_raw = np.column_stack([
        np.nan_to_num((g[f"{k}_i"] * g[f"{k}_j"]).to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        for k in topic_cols
    ])
    S = np.column_stack([safe_scale_1d(S_raw[valid, j]) for j in range(S_raw.shape[1])])

    full = sm.OLS(y0, sm.add_constant(np.column_stack([X_ctrl, S]))).fit()
    r2_full = float(full.rsquared)

    K = S.shape[1]
    beta_topics = np.asarray(full.params)[-K:]
    t_topics = np.asarray(full.tvalues)[-K:]

    for idx_k, k in enumerate(topic_cols):
        if K == 1:
            r2_minus_k = r2_base
        else:
            S_minus = np.delete(S, idx_k, axis=1)
            r2_minus_k = float(
                sm.OLS(y0, sm.add_constant(np.column_stack([X_ctrl, S_minus]))).fit().rsquared
            )

        results_rows.append({
            "quarter": q,
            "topic": k,
            "beta": float(beta_topics[idx_k]),
            "t_stat": float(t_topics[idx_k]),
            "R2_base": r2_base,
            "R2_full": r2_full,
            "R2_minus_k": r2_minus_k,
            "dR2": float(r2_full - r2_minus_k),
            "n_pairs": int(valid.sum()),
        })

results = pd.DataFrame(results_rows)

if results.empty:
    raise RuntimeError("Regression results are empty.")

results["year_q"] = results["quarter"].astype(str).str[:4].astype(int)

# ============================
# Step 5: Baseline normalization and output tables
# ============================
# This step first defines the baseline sample used for normalization and, if necessary,
# replaces it with an earliest available fallback period. It then computes topic-specific
# baseline means and standard deviations of dR², uses these to calculate z-scores for all
# quarter-topic observations, and filters the results to the selected output year. Finally,
# it creates two output tables: one containing the top topics within each quarter of the
# output year, and one containing the top unique topics aggregated across the output year,
# before saving all result tables to the output folder.
baseline = results[
    results["year_q"].between(DYNAMIC_BASELINE_START_YEAR, DYNAMIC_BASELINE_END_YEAR)
].copy()

if baseline.empty:
    avail_years = sorted(results["year_q"].unique().tolist())
    if not avail_years:
        raise ValueError("No regression years available for baseline construction.")
    fb_start = avail_years[0]
    fb_end = min(fb_start + 2, avail_years[-1])
    baseline = results[results["year_q"].between(fb_start, fb_end)].copy()
    if baseline.empty:
        raise ValueError("Fallback baseline window is also empty.")

mu = baseline.groupby("topic")["dR2"].mean()
sd = baseline.groupby("topic")["dR2"].std(ddof=0)
cnt = baseline.groupby("topic")["dR2"].count()

sd = sd.where(cnt >= MIN_BASELINE_OBS, np.nan)
sd = sd.clip(lower=MIN_BASELINE_SD)

results["z"] = (results["dR2"] - results["topic"].map(mu)) / results["topic"].map(sd)
results = results.replace([np.inf, -np.inf], np.nan).copy()

target_year_results = results[results["year_q"] == OUTPUT_TARGET_YEAR].copy()

top_target_year = (
    target_year_results
    .sort_values(["quarter", "z", "dR2"], ascending=[True, False, False])
    .groupby("quarter", as_index=False, group_keys=False)
    .head(TOP_Z_COUNT)
    [["quarter", "topic", "z", "dR2", "beta", "t_stat", "n_pairs"]]
    .copy()
    if not target_year_results.empty
    else pd.DataFrame(columns=["quarter", "topic", "z", "dR2", "beta", "t_stat", "n_pairs"])
)

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
    if not target_year_results.empty
    else pd.DataFrame(columns=[
        "topic", "z_mean", "z_max", "dR2_mean", "dR2_max",
        "beta_mean", "t_stat_mean", "quarters_present", "avg_n_pairs"
    ])
)

out_dir = HERE / "Output"
out_dir.mkdir(exist_ok=True)

results.to_csv(out_dir / f"topic_dr2_zscores_controls_{OUTPUT_TAG}.csv", index=False)
top_target_year.to_csv(
    out_dir / f"top_z_topics_output_{OUTPUT_TARGET_YEAR}_text_{DYNAMIC_TF_RUN_LABEL}_{OUTPUT_TAG}.csv",
    index=False,
)
top_target_year_topics.to_csv(
    out_dir / f"top_z_topics_unique_output_{OUTPUT_TARGET_YEAR}_text_{DYNAMIC_TF_RUN_LABEL}_{OUTPUT_TAG}.csv",
    index=False,
)