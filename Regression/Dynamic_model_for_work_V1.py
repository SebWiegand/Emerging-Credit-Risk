# ===========================================================
# Imports
# ===========================================================
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import warnings

# ===========================================================
# Settings
# ===========================================================
DYNAMIC_TARGET_YEAR = 2020
DYNAMIC_WINDOW_LENGTH = 5
DYNAMIC_WINDOW_START = DYNAMIC_TARGET_YEAR - DYNAMIC_WINDOW_LENGTH + 1
DYNAMIC_WINDOW_END = DYNAMIC_TARGET_YEAR
DYNAMIC_TF_RUN_LABEL = f"{DYNAMIC_WINDOW_START}_{DYNAMIC_WINDOW_END}"

DYNAMIC_BASELINE_START_YEAR = DYNAMIC_WINDOW_START
DYNAMIC_BASELINE_END_YEAR = DYNAMIC_TARGET_YEAR - 1

TOP_Z_COUNT = 10
OUTPUT_TAG = "run1"

MIN_TOPIC_LOADING = 0
MIN_BASELINE_SD = 0.00005
MIN_VALID_PAIRS = 1

STEPWISE_MAX_CANDIDATES = 150
STEPWISE_MAX_TOPICS = 30
STEPWISE_MIN_DR2 = None

PRUNE_TOPICS_BY_DR2 = False
TARGET_TOPIC_COUNT = 35
RUN_DIAGNOSTICS = False

TOPIC_EXCLUDE = []
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

# ===========================================================
# Step 0: Load files
# ===========================================================

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

COV_PATH = HERE / "Output" / "quarterly_pairwise_covariance.csv"
TF_PATH = REPO_ROOT / "Text analytics" / "outputs_textual_factors" / f"extraction_summary_{DYNAMIC_TF_RUN_LABEL}_V1.csv"
CTRL_PATH = HERE / "Data" / "Control_variable_final.xlsx"

cov = pd.read_csv(COV_PATH)
tf = pd.read_csv(TF_PATH)
controls_raw = pd.read_excel(CTRL_PATH)

# ===========================================================
# Step 1: Prepare controls
# ===========================================================

controls_raw["company"] = controls_raw["company"].astype(str).str.strip().map(_norm_id)
controls_raw["year"] = controls_raw["year"].astype(int)

controls_raw["ln_assets"] = np.log(controls_raw["total_assets"])
controls_raw["equity_assets"] = controls_raw["common_equity"] / controls_raw["total_assets"]

control_cols = [
    "ln_assets",
    "cash",
    "net_debt",
    "ebit_margin",
    "net_income",
    "equity_assets",
    "capex_5y_cagr",
    "green_revenue_pct",
]

controls_firm_year = (
    controls_raw
    .groupby(["company", "year"], as_index=False)[control_cols]
    .mean()
    .rename(columns={"company": "firm_id"})
)

# ===========================================================
# Step 2: Prepare controls
# ===========================================================
all_topic_cols = [c for c in tf.columns if c.startswith("topic_loading_")]
topic_cols = list(TOPIC_INCLUDE) if TOPIC_INCLUDE else [c for c in all_topic_cols if c not in set(TOPIC_EXCLUDE)]

tf_firm_year = tf.groupby(["bank", "year"], as_index=False)[topic_cols].mean()
tf_firm_year["bank"] = tf_firm_year["bank"].map(_norm_id)

tf_firm_year[topic_cols] = tf_firm_year[topic_cols].apply(pd.to_numeric, errors="coerce")
tf_firm_year[topic_cols] = tf_firm_year[topic_cols].where(
    tf_firm_year[topic_cols].abs() >= MIN_TOPIC_LOADING, 0.0
)

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

topic_year_mean_map = {
    (topic, int(year)): float(value)
    for year, row in tf_firm_year.groupby("year")[topic_cols].mean().iterrows()
    for topic, value in row.items()
}

# ===========================================================
# Step 3: Merge onto pair panel
# ===========================================================

cov["q_year"] = cov["quarter"].astype(str).str[:4].astype(int)
cov["lag_year"] = cov["q_year"]
cov["i"] = cov["i"].astype(str).str.strip().map(_norm_id)
cov["j"] = cov["j"].astype(str).str.strip().map(_norm_id)

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

control_bases = [
    "ln_assets", "cash", "net_debt", "ebit_margin",
    "net_income", "equity_assets", "capex_5y_cagr", "green_revenue_pct"
]

panel = panel.rename(columns={
    **{f"{c}_i_ctrl": f"{c}_i" for c in control_bases},
    **{f"{c}_j_ctrl": f"{c}_j" for c in control_bases},
})

ctrl_cols = [f"{c}_{side}" for c in control_bases for side in ["i", "j"]]

panel["year_q"] = panel["quarter"].astype(str).str[:4].astype(int)
panel = panel[panel["lag_year"].between(DYNAMIC_WINDOW_START, DYNAMIC_WINDOW_END)].copy()
panel = panel.loc[panel[ctrl_cols].notna().all(axis=1)].copy()

# ===========================================================
# Step 4: Stepwise selection
# ===========================================================

def forward_stepwise_select_topics(panel_df, candidate_topics):
    selected = []
    remaining = list(candidate_topics)

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
    Xi0 = np.column_stack([safe_scale_1d(Xi_raw[valid, j]) for j in range(Xi_raw.shape[1])])

    S_raw = np.column_stack([
        np.nan_to_num(
            (panel_df[f"{t}_i"] * panel_df[f"{t}_j"]).to_numpy(dtype=float),
            nan=0.0, posinf=0.0, neginf=0.0
        )
        for t in candidate_topics
    ])
    S0 = np.column_stack([safe_scale_1d(S_raw[valid, j]) for j in range(S_raw.shape[1])])
    topic_to_idx = {t: j for j, t in enumerate(candidate_topics)}

    current_r2 = float(sm.OLS(y0, sm.add_constant(Xi0)).fit().rsquared)

    while remaining and len(selected) < STEPWISE_MAX_TOPICS:
        best_topic = None
        best_r2 = -np.inf

        selected_idx = [topic_to_idx[t] for t in selected]
        S_prev = S0[:, selected_idx] if selected_idx else None

        for t in remaining:
            s0 = S0[:, topic_to_idx[t]]
            X = sm.add_constant(np.column_stack([Xi0, s0] if S_prev is None else [Xi0, S_prev, s0]))
            r2 = float(sm.OLS(y0, X).fit().rsquared)

            if np.isfinite(r2) and r2 > best_r2:
                best_r2 = r2
                best_topic = t

        if best_topic is None:
            break

        gain = best_r2 - current_r2
        if STEPWISE_MIN_DR2 is not None and gain < STEPWISE_MIN_DR2:
            break

        selected.append(best_topic)
        remaining.remove(best_topic)
        current_r2 = best_r2

    return selected


topic_cols = forward_stepwise_select_topics(panel, topic_cols)
if not topic_cols:
    raise RuntimeError("Forward stepwise selection returned 0 topics.")

out_dir = HERE / "Output"
out_dir.mkdir(exist_ok=True)

pd.DataFrame({"topic": topic_cols}).to_csv(
    out_dir / f"stepwise_selected_topics_{DYNAMIC_TF_RUN_LABEL}_{OUTPUT_TAG}.csv",
    index=False,
)

topic_year_mean_map = {
    (c, y): v
    for (c, y), v in topic_year_mean_map.items()
    if c in set(topic_cols)
}

# ===========================================================
# Step 5: Quarterly regressions
# ===========================================================

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

    valid_ctrl = np.isfinite(y) & np.isfinite(Xi_raw).all(axis=1)
    if valid_ctrl.sum() < MIN_VALID_PAIRS:
        continue

    y0 = y[valid_ctrl]
    Xi0 = np.column_stack([safe_scale_1d(Xi_raw[valid_ctrl, j]) for j in range(Xi_raw.shape[1])])

    r2_controls = float(sm.OLS(y0, sm.add_constant(Xi0)).fit().rsquared)

    S_raw = np.column_stack([
        np.nan_to_num((g[f"{k}_i"] * g[f"{k}_j"]).to_numpy(dtype=float), nan=0.0)
        for k in topic_cols
    ])
    S0 = np.column_stack([safe_scale_1d(S_raw[valid_ctrl, j]) for j in range(S_raw.shape[1])])

    full_all = sm.OLS(y0, sm.add_constant(np.column_stack([Xi0, S0]))).fit()
    r2_full_all = float(full_all.rsquared)

    K = S0.shape[1]
    beta_topics = np.asarray(full_all.params)[-K:]
    t_topics = np.asarray(full_all.tvalues)[-K:]

    for idx_k, k in enumerate(topic_cols):
        if K == 1:
            r2_minus_k = r2_controls
        else:
            S_minus = np.delete(S0, idx_k, axis=1)
            r2_minus_k = float(sm.OLS(y0, sm.add_constant(np.column_stack([Xi0, S_minus]))).fit().rsquared)

        results_rows.append({
            "quarter": q,
            "topic": k,
            "beta": float(beta_topics[idx_k]),
            "t_stat": float(t_topics[idx_k]),
            "R2_base": r2_controls,
            "R2_full": r2_full_all,
            "R2_minus_k": r2_minus_k,
            "dR2": float(r2_full_all - r2_minus_k),
            "n_pairs": int(valid_ctrl.sum()),
        })

results = pd.DataFrame(results_rows)
if results.empty:
    raise RuntimeError("Regression results are empty.")

# ===========================================================
# Step 5: Quarterly regressions
# ===========================================================

results["year_q"] = results["quarter"].astype(str).str[:4].astype(int)

baseline = results[
    results["year_q"].between(DYNAMIC_BASELINE_START_YEAR, DYNAMIC_BASELINE_END_YEAR)
].copy()

if baseline.empty:
    raise ValueError("Baseline window produced 0 rows.")

mu = baseline.groupby("topic")["dR2"].mean()
sd = baseline.groupby("topic")["dR2"].std(ddof=0)
cnt = baseline.groupby("topic")["dR2"].count()

MIN_BASELINE_OBS = 12
sd = sd.where(cnt >= MIN_BASELINE_OBS, np.nan)
sd = sd.clip(lower=MIN_BASELINE_SD)

results["z"] = (results["dR2"] - results["topic"].map(mu)) / results["topic"].map(sd)
results = results.replace([np.inf, -np.inf], np.nan).copy()
results["z"] = results["z"].clip(-10, 1000)
results["z_plot"] = results["z"]

