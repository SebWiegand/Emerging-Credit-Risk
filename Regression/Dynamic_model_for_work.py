# ===========================================================
# Importing packages
# ===========================================================
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import warnings

# ===========================================================
# GLOBAL SETTINGS
# ===========================================================
# The regression target year uses a rolling text window ending in this year.
DYNAMIC_TARGET_YEAR = 2020
DYNAMIC_WINDOW_LENGTH = 5
DYNAMIC_WINDOW_START = DYNAMIC_TARGET_YEAR - DYNAMIC_WINDOW_LENGTH + 1
DYNAMIC_WINDOW_END = DYNAMIC_TARGET_YEAR
DYNAMIC_TF_RUN_LABEL = f"{DYNAMIC_WINDOW_START}_{DYNAMIC_WINDOW_END}"

# Baseline years used to normalize topic-level dR2 in the dynamic model. By default, the baseline is the full rolling window except the target year.
DYNAMIC_BASELINE_START_YEAR = DYNAMIC_WINDOW_START
DYNAMIC_BASELINE_END_YEAR = DYNAMIC_TARGET_YEAR - 1

# Number of top-z topics saved/reported for the target year.
TOP_Z_COUNT = 10

# Short tag appended to output filenames.
OUTPUT_TAG = "run1"

# --------------------------------------------------
# Optional diagnostics and pruning
# --------------------------------------------------
# If True, run the large diagnostics/statistics block at the end of the script.
RUN_DIAGNOSTICS = False

# If True, rank topics by average marginal contribution (mean dR2), save the ranking, and stop so TOPIC_INCLUDE can be updated manually.
PRUNE_TOPICS_BY_DR2 = False
TARGET_TOPIC_COUNT = 35

# --------------------------------------------------
# Topic-loading and plotting settings
# --------------------------------------------------
# Minimum absolute firm-year topic loading. # Values below this threshold are treated as zero before pair exposures are built.
MIN_TOPIC_LOADING = 0

# Minimum baseline standard deviation used in the z-score denominator. # This avoids NaN / explosive z-scores when baseline dR2 is nearly constant.
MIN_BASELINE_SD = 0.00005

# --------------------------------------------------
# Forward stepwise topic selection for the dynamic model
# --------------------------------------------------

# Maximum number of candidate topics entering stepwise selection.
STEPWISE_MAX_CANDIDATES = 150

# Maximum number of topics kept after stepwise selection.
STEPWISE_MAX_TOPICS = 30

# Optional minimum dR2 gain required to keep adding topics. # Currently not enforced when set to None.
STEPWISE_MIN_DR2 = None

# Topics to exclude manually when TOPIC_INCLUDE is not used.
TOPIC_EXCLUDE = []  # e.g. ["topic_loading_12"]

# Minimum number of valid firm pairs required to estimate a quarter-topic regression. # For production runs, a higher threshold than 1 is usually preferable.
MIN_VALID_PAIRS = 1

# --------------------------------------------------
# Manual topic list
# --------------------------------------------------
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
]

# ============================
# Step 0: Locate and load the three required input files
# ============================
HERE = Path(__file__).resolve().parent   # Folder containing this regression script
REPO_ROOT = HERE.parent                  # Project root: .../Emerging-Credit-Risk_1

# Covariance panel candidates
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

# Firm-year controls candidates. These controls are merged later to the i and j firms in the pairwise covariance panel.
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
if len(topic_cols) > STEPWISE_MAX_CANDIDATES:
    topic_cols = (
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
# Same-year alignment: annual report year t is assumed informative for quarters in year t.
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
    how="left",
)
panel = panel.merge(
    tf_firm_year,
    left_on=["j", "lag_year"],
    right_on=["bank", "year"],
    how="left",
    suffixes=("_i", "_j"),
)
# Merge firm-year controls for i and j.
panel = panel.merge(
    controls_firm_year,
    left_on=["i", "lag_year"],
    right_on=["firm_id", "year"],
    how="left",
).drop(columns=["firm_id", "year"])
panel = panel.merge(
    controls_firm_year,
    left_on=["j", "lag_year"],
    right_on=["firm_id", "year"],
    how="left",
    suffixes=("_i_ctrl", "_j_ctrl"),
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

# Restrict the regression sample to the rolling text window only.
panel["year_q"] = panel["quarter"].astype(str).str[:4].astype(int)
panel = panel[panel["lag_year"].between(DYNAMIC_WINDOW_START, DYNAMIC_WINDOW_END)].copy()

# Keep only rows with complete control information.
row_ok_all = panel[ctrl_cols].notna().all(axis=1)
panel = panel.loc[row_ok_all].copy()

print(
    f"Restricted panel to rolling text window years {DYNAMIC_WINDOW_START}-{DYNAMIC_WINDOW_END}. "
    f"Remaining rows after requiring complete controls: {len(panel)}"
)

print("Valid rows by year (ALL 8 controls complete):")
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

    Xi_raw = np.column_stack([
        (panel_df["ln_assets_i"] * panel_df["ln_assets_j"]).to_numpy(dtype=float),
        (panel_df["cash_i"] * panel_df["cash_j"]).to_numpy(dtype=float),
        (panel_df["net_debt_i"] * panel_df["net_debt_j"]).to_numpy(dtype=float),
        (panel_df["ebit_margin_i"] * panel_df["ebit_margin_j"]).to_numpy(dtype=float),
        (panel_df["net_income_i"] * panel_df["net_income_j"]).to_numpy(dtype=float),
        (panel_df["equity_assets_i"] * panel_df["equity_assets_j"]).to_numpy(dtype=float),
        (panel_df["capex_5y_cagr_i"] * panel_df["capex_5y_cagr_j"]).to_numpy(dtype=float),
        (panel_df["green_revenue_pct_i"] * panel_df["green_revenue_pct_j"]).to_numpy(dtype=float),
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
# Step 5: Quarterly regressions
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

    Xi_raw = np.column_stack([Xi_core_raw, Xi_other_raw])

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
# DIAGNOSTICS / STATISTICS (optional)
# Light diagnostic summary for development / sanity checks.
# ============================================================
if RUN_DIAGNOSTICS:
    def _pct(x: float) -> str:
        return f"{100 * x:.2f}%"

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
            "display.width", 160,
        ):
            print(df)

    print("\n\n" + "#" * 90)
    print("DIAGNOSTICS START")
    print("#" * 90)

    # 1) Core dataset shapes
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

    # 2) Basic coverage / merge checks
    if set(["quarter", "i", "j"]).issubset(cov.columns):
        dup_share = float(cov.duplicated(subset=["quarter", "i", "j"]).mean())
        print("\n2a) Cov duplicate share on (quarter, i, j):", _pct(dup_share))

    if "year_q" not in panel.columns and "quarter" in panel.columns:
        panel["year_q"] = panel["quarter"].astype(str).str[:4].astype(int)

    tf_i_cols = [f"{k}_i" for k in topic_cols if f"{k}_i" in panel.columns]
    tf_j_cols = [f"{k}_j" for k in topic_cols if f"{k}_j" in panel.columns]

    if tf_i_cols:
        print("2b) TF coverage i-side (any topic non-missing):", _pct(float(panel[tf_i_cols].notna().any(axis=1).mean())))
    if tf_j_cols:
        print("2c) TF coverage j-side (any topic non-missing):", _pct(float(panel[tf_j_cols].notna().any(axis=1).mean())))

    ctrl_cols_present = [c for c in ctrl_cols if c in panel.columns]
    if ctrl_cols_present:
        ctrl_cov_by_year = (
            panel.groupby("year_q")[ctrl_cols_present]
                 .apply(lambda d: float(d.notna().mean().mean()))
                 .rename("controls_nonmissing_share")
                 .reset_index()
        )
        _print_df("2d) Controls coverage by year", ctrl_cov_by_year, max_rows=50)

    # 3) Topic sparsity after firm-year filtering
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
    _print_df("3a) Lowest non-zero-share topics", topic_nz_df.head(15))
    _print_df("3b) Highest non-zero-share topics", topic_nz_df.tail(15))

    # 4) Regression output checks
    if set(["quarter", "topic"]).issubset(results.columns):
        rows_per_q = results.groupby("quarter")["topic"].size().rename("rows_per_quarter")
        _print_df("4a) Rows per quarter summary", rows_per_q.describe().to_frame())

    if "n_pairs" in results.columns:
        n_pairs_summary = pd.DataFrame([{
            "min": float(results["n_pairs"].min()),
            "median": float(results["n_pairs"].median()),
            "mean": float(results["n_pairs"].mean()),
            "max": float(results["n_pairs"].max()),
        }])
        _print_df("4b) n_pairs summary", n_pairs_summary)

    if "dR2" in results.columns:
        neg_share = float((results["dR2"] < -1e-12).mean())
        tiny_neg = float(((results["dR2"] < 0) & (results["dR2"] >= -1e-6)).mean())
        print("\n4c) dR2 negative share:", _pct(neg_share), "| tiny negatives (>-1e-6):", _pct(tiny_neg))

        topic_rank = (
            results.groupby("topic")["dR2"]
                   .agg(mean="mean", median="median")
                   .sort_values("mean", ascending=False)
                   .reset_index()
        )
        _print_df("4d) Top 15 topics by mean dR2", topic_rank.head(15))
        _print_df("4e) Bottom 15 topics by mean dR2", topic_rank.tail(15))

    # 5) Baseline coverage
    if "year_q" in results.columns:
        base = results[
            results["year_q"].between(DYNAMIC_BASELINE_START_YEAR, DYNAMIC_BASELINE_END_YEAR)
        ].copy()
        if len(base) > 0:
            base_counts = base.groupby("topic")["dR2"].count().rename("baseline_obs").reset_index()
            _print_df("5) Baseline observations per topic", base_counts.sort_values("baseline_obs").head(20))
        else:
            print("\n5) Baseline window produced 0 rows (fallback logic should already have handled this above).")

    print("\n" + "#" * 90)
    print("DIAGNOSTICS DONE ✅")
    print("#" * 90)