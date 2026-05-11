from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================================================
# PATHS
# ============================================================
BASE = Path(__file__).resolve().parent
DATA = BASE / "Data"
OUT = BASE / "Output"
TEXT_OUT = BASE.parent / "Text analytics" / "outputs_textual_factors"
OUT.mkdir(exist_ok=True)

TF_PATH = TEXT_OUT / "extraction_summary_ALL_Final_V1.csv"
CONTROLS_PATH = DATA / "Control_variable_final.xlsx"
TOPIC_PATH = OUT / "topic_dr2_zscores_controls_Run5_2025_final.csv"
RETURNS_PATH = DATA / "Stock_data_final.xlsx"


# ============================================================
# SETTINGS
# ============================================================
TSTAT_FILTER = 1.5
MIN_DAILY_OBS_PER_QUARTER = 20
MIN_FMB_OBS_PER_QUARTER = 10
SAMPLE_START_YEAR = 2010
SAMPLE_END_YEAR = 2025
USE_Z_IF_AVAILABLE = True


# ============================================================
# HELPERS
# ============================================================
def _find_col(cols: list[str], candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"No {label} column found. Available columns: {cols}")


def _to_num(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace("\xa0", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _standardize_within_quarter(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    def _z(x: pd.Series) -> pd.Series:
        mu = x.mean()
        sd = x.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - mu) / sd

    df[out_col] = df.groupby("quarter")[col].transform(_z)
    return df


def fama_macbeth(df: pd.DataFrame, y_col: str, x_cols: list[str], min_obs: int = 10) -> dict:
    betas = []
    for q, g in df.groupby("quarter"):
        g = g.dropna(subset=[y_col] + x_cols)
        if len(g) < min_obs:
            continue
        y = g[y_col].astype(float)
        X = sm.add_constant(g[x_cols].astype(float), has_constant="add")
        try:
            res = sm.OLS(y, X).fit()
        except Exception:
            continue
        row = {"quarter": q, "n": len(g), "r2": res.rsquared}
        for c in X.columns:
            row[c] = res.params.get(c, np.nan)
        betas.append(row)

    if not betas:
        return {
            "months": 0,
            "avg_n": np.nan,
            "avg_r2": np.nan,
            "coef_exposure": np.nan,
            "t_exposure": np.nan,
        }

    b = pd.DataFrame(betas)
    coef = b["er_score_z"].mean()
    se = b["er_score_z"].std(ddof=1) / np.sqrt(len(b)) if len(b) > 1 else np.nan
    tval = coef / se if pd.notna(se) and se != 0 else np.nan
    return {
        "months": int(len(b)),
        "avg_n": float(b["n"].mean()),
        "avg_r2": float(b["r2"].mean()),
        "coef_exposure": float(coef),
        "t_exposure": float(tval) if pd.notna(tval) else np.nan,
    }


# ============================================================
# LOAD TOPIC / TEXT DATA
# ============================================================
print("Loading TF extraction summary from:", TF_PATH)
tf = pd.read_csv(TF_PATH)
tf.columns = tf.columns.str.strip().str.lower()

print("Loading topic-quarter scores from:", TOPIC_PATH)
topic = pd.read_csv(TOPIC_PATH)
topic.columns = topic.columns.str.strip().str.lower()

quarter_col = _find_col(topic.columns.tolist(), ["quarter"], "quarter")
topic_col = _find_col(topic.columns.tolist(), ["topic"], "topic")
tstat_col = _find_col(topic.columns.tolist(), ["t_stat", "tstat"], "t-stat")
score_col = "z" if USE_Z_IF_AVAILABLE and "z" in topic.columns else "beta"
if score_col not in topic.columns:
    raise KeyError(f"No z or beta column found in topic file. Available columns: {topic.columns.tolist()}")

topic = topic[[quarter_col, topic_col, score_col, tstat_col]].copy()
topic[quarter_col] = topic[quarter_col].astype(str)
topic[tstat_col] = pd.to_numeric(topic[tstat_col], errors="coerce")
topic[score_col] = pd.to_numeric(topic[score_col], errors="coerce")
topic = topic.loc[topic[tstat_col].abs() >= TSTAT_FILTER].copy()
print(f"Applying topic t-stat filter: |t_stat| >= {TSTAT_FILTER}")
print("Topics surviving filter:", topic[topic_col].nunique())
print("First few surviving topics:", sorted(topic[topic_col].dropna().unique().tolist())[:10])

firm_col_tf = _find_col(tf.columns.tolist(), ["firm", "company", "company name", "name", "bank"], "firm")
year_col_tf = _find_col(tf.columns.tolist(), ["year", "fyear"], "year")

possible_topic_cols = [c for c in tf.columns if c.startswith("topic_loading_")]
used_topics = sorted(set(possible_topic_cols).intersection(set(topic[topic_col].unique())))
if not used_topics:
    raise ValueError("No overlapping topic_loading columns found between extraction summary and topic file.")
print("Topics used in exposure construction:", len(used_topics))
print("First few used topics:", used_topics[:10])

tf = tf[[firm_col_tf, year_col_tf] + used_topics].copy()
tf = tf.rename(columns={firm_col_tf: "firm", year_col_tf: "year"})
tf["firm"] = tf["firm"].astype(str).str.upper().str.strip()
tf["year"] = pd.to_numeric(tf["year"], errors="coerce").astype("Int64")
for c in used_topics:
    tf[c] = pd.to_numeric(tf[c], errors="coerce")

# Build firm-quarter ER score: sum(topic loading * quarter score)
exposure_rows = []
for quarter, gq in topic.groupby(quarter_col):
    score_map = gq.set_index(topic_col)[score_col].reindex(used_topics).fillna(0.0)
    scores = score_map.to_numpy(dtype=float)
    tmp = tf[["firm", "year"] + used_topics].copy()
    X = tmp[used_topics].fillna(0.0).to_numpy(dtype=float)
    tmp["quarter"] = quarter
    tmp["er_score"] = X @ scores
    tmp["quarter_year"] = pd.PeriodIndex(tmp["quarter"], freq="Q").year
    tmp = tmp.loc[tmp["year"] == tmp["quarter_year"]].copy()
    exposure_rows.append(tmp[["firm", "year", "quarter", "er_score"]])

exposure = pd.concat(exposure_rows, ignore_index=True)
exposure = _standardize_within_quarter(exposure, "er_score", "er_score_z")
print("Constructed firm-quarter exposure panel:", exposure.shape)


# ============================================================
# LOAD CONTROLS
# ============================================================
print("Loading controls from:", CONTROLS_PATH)
controls = pd.read_excel(CONTROLS_PATH)
controls.columns = controls.columns.str.lower().str.strip()
print("Controls columns:", controls.columns.tolist())

firm_col_c = _find_col(controls.columns.tolist(), ["company name", "company", "firm", "name", "bank"], "firm")
year_col_c = _find_col(controls.columns.tolist(), ["year", "fyear"], "year")
assets_col = _find_col(controls.columns.tolist(), ["total assets", "assets", "total_assets"], "total assets")

net_debt_col = None
for c in controls.columns:
    if c.startswith("net debt"):
        net_debt_col = c
        break
if net_debt_col is None:
    raise KeyError(f"No net debt column found. Available columns: {controls.columns.tolist()}")

cash_col = None
for c in controls.columns:
    if c == "cash" or c.startswith("cash"):
        cash_col = c
        break
if cash_col is None:
    raise KeyError(f"No cash column found. Available columns: {controls.columns.tolist()}")

controls = controls.rename(columns={firm_col_c: "firm", year_col_c: "year"}).copy()
controls["firm"] = controls["firm"].astype(str).str.upper().str.strip()
controls["year"] = pd.to_numeric(controls["year"], errors="coerce")
for c in [assets_col, net_debt_col, cash_col]:
    controls[c] = _to_num(controls[c])
controls["log_assets"] = np.log(controls[assets_col])
controls["leverage"] = controls[net_debt_col] / controls[assets_col]
controls["cash_ratio"] = controls[cash_col] / controls[assets_col]
controls = controls[["firm", "year", "log_assets", "leverage", "cash_ratio"]].dropna()

# OPTIONAL: industry FE (if available)
industry_col = None
for c in controls.columns:
    if "sector" in c or "industry" in c:
        industry_col = c
        break

if industry_col is not None:
    controls[industry_col] = controls[industry_col].astype(str)
    dummies = pd.get_dummies(controls[industry_col], prefix="ind", drop_first=True)
    controls = pd.concat([controls, dummies], axis=1)
    industry_dummies = list(dummies.columns)
else:
    industry_dummies = []


# ============================================================
# LOAD RETURNS AND BUILD QUARTERLY OUTCOMES
# ============================================================
print("Loading returns from:", RETURNS_PATH)
ret = pd.read_excel(RETURNS_PATH)
ret.columns = ret.columns.str.strip()
print("Returns columns:", ret.columns.tolist())

firm_col_r = _find_col(ret.columns.tolist(), ["Company", "Company name", "company", "company name"], "firm")
date_col_r = _find_col(ret.columns.tolist(), ["Date", "date"], "date")
price_col_r = _find_col(ret.columns.tolist(), ["Price Close", "price close", "Close", "close"], "price")

ret["firm"] = ret[firm_col_r].astype(str).str.upper().str.strip()
ret["date"] = pd.to_datetime(ret[date_col_r], dayfirst=True, errors="coerce")
ret[price_col_r] = _to_num(ret[price_col_r])
ret = ret.dropna(subset=["firm", "date", price_col_r]).copy()
ret = ret.sort_values(["firm", "date"])
ret["daily_ret"] = ret.groupby("firm")[price_col_r].pct_change(fill_method=None)

# --- BUILD MONTHLY RETURNS FOR MOMENTUM ---
ret["month"] = ret["date"].dt.to_period("M")
monthly_ret = (
    ret.groupby(["firm", "month"])["daily_ret"]
    .apply(lambda x: np.prod(1.0 + x) - 1.0)
    .reset_index(name="m_ret")
)

# momentum = cumulative return from t-12 to t-2 (skip most recent month)
monthly_ret = monthly_ret.sort_values(["firm", "month"])
monthly_ret["mom_12_2"] = (
    monthly_ret.groupby("firm")["m_ret"]
    .transform(lambda x: (1 + x).rolling(11).apply(np.prod, raw=True).shift(1) - 1)
)

ret["quarter"] = ret["date"].dt.to_period("Q").astype(str)
ret = ret.dropna(subset=["daily_ret"])

quarterly = (
    ret.groupby(["firm", "quarter"])["daily_ret"]
    .agg(q_vol=lambda x: np.std(x, ddof=1),
         n_days="count",
         q_ret=lambda x: np.prod(1.0 + x) - 1.0)
    .reset_index()
)
quarterly = quarterly.loc[quarterly["n_days"] >= MIN_DAILY_OBS_PER_QUARTER].copy()
quarterly["quarter_period"] = pd.PeriodIndex(quarterly["quarter"], freq="Q")
quarterly["year"] = quarterly["quarter_period"].dt.year
quarterly["log_q_vol"] = np.log(quarterly["q_vol"].where(quarterly["q_vol"] > 0))
quarterly["log_q_ret"] = np.log1p(quarterly["q_ret"])
quarterly = quarterly.sort_values(["firm", "quarter_period"])
quarterly["lead1_ret"] = quarterly.groupby("firm")["q_ret"].shift(-1)
quarterly["lead1_log_q_ret"] = quarterly.groupby("firm")["log_q_ret"].shift(-1)
quarterly["lead1_log_q_vol"] = quarterly.groupby("firm")["log_q_vol"].shift(-1)
print("Quarterly returns/volatility panel:", quarterly.shape)

# --- MAP MOMENTUM TO QUARTERLY ---
monthly_ret["quarter"] = monthly_ret["month"].dt.to_period("Q").astype(str)
momentum_q = (
    monthly_ret.groupby(["firm", "quarter"])["mom_12_2"]
    .last()
    .reset_index()
)


# ============================================================
# MERGE PANEL
# ============================================================
panel = exposure.merge(controls, on=["firm", "year"], how="left")
panel = panel.merge(
    quarterly[["firm", "quarter", "lead1_ret", "lead1_log_q_ret", "lead1_log_q_vol", "q_ret", "log_q_ret", "log_q_vol"]],
    on=["firm", "quarter"],
    how="inner",
)

# merge momentum
panel = panel.merge(momentum_q, on=["firm", "quarter"], how="left")

panel["quarter_period"] = pd.PeriodIndex(panel["quarter"], freq="Q")
panel = panel.loc[
    (panel["quarter_period"].dt.year >= SAMPLE_START_YEAR)
    & (panel["quarter_period"].dt.year <= SAMPLE_END_YEAR)
].copy()
print("Merged panel shape:", panel.shape)
print("Merged firms:", panel["firm"].nunique())
print("Quarter range:", panel["quarter"].min(), "to", panel["quarter"].max())

panel.to_csv(OUT / "quarterly_er_exposure_panel_simple.csv", index=False)


# ============================================================
# REGRESSIONS (MULTIPLE LAGS)
# ============================================================
results = []
controls_x = ["log_assets", "leverage", "cash_ratio", "mom_12_2"] + industry_dummies
LAGS = [1, 2, 4, 8]  # 1Q, 2Q, 1Y, 2Y

panel = panel.sort_values(["firm", "quarter_period"]).copy()

for lag in LAGS:
    panel[f"lead{lag}_ret"] = panel.groupby("firm")["q_ret"].shift(-lag)
    panel[f"lead{lag}_log_q_ret"] = panel.groupby("firm")["log_q_ret"].shift(-lag)
    panel[f"lead{lag}_log_q_vol"] = panel.groupby("firm")["log_q_vol"].shift(-lag)

for lag in LAGS:
    for dep_base, dep_name in [("ret", "return"), ("log_q_ret", "log_return"), ("log_q_vol", "volatility")]:
        dep = f"lead{lag}_{dep_base}"

        pooled = panel.dropna(subset=[dep, "er_score_z"] + controls_x).copy()
        if len(pooled) > 0:
            y = pooled[dep].astype(float)
            X = sm.add_constant(pooled[["er_score_z"] + controls_x].astype(float), has_constant="add")
            fit = sm.OLS(y, X).fit(cov_type="HC1")
            results.append({
                "method": "pooled_ols_hc1",
                "lag_quarters": lag,
                "dependent": dep_name,
                "coef_exposure": float(fit.params.get("er_score_z", np.nan)),
                "t_exposure": float(fit.tvalues.get("er_score_z", np.nan)),
                "r2": float(fit.rsquared),
                "n_obs": int(fit.nobs),
            })

        fm = fama_macbeth(panel, dep, ["er_score_z"] + controls_x, min_obs=MIN_FMB_OBS_PER_QUARTER)
        results.append({
            "method": "fama_macbeth",
            "lag_quarters": lag,
            "dependent": dep_name,
            "coef_exposure": fm["coef_exposure"],
            "t_exposure": fm["t_exposure"],
            "r2": fm["avg_r2"],
            "n_obs": fm["months"],
            "avg_n_per_quarter": fm["avg_n"],
        })

results = pd.DataFrame(results)
results.to_csv(OUT / "predicting_volatility_lagged_results.csv", index=False)

print("\nPreview of results:")
print(results.to_string(index=False))
print("\nSaved merged panel to:", OUT / "quarterly_er_exposure_panel_simple.csv")
print("Saved regression results to:", OUT / "predicting_volatility_lagged_results.csv")