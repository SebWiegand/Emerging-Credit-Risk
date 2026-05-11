import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================
HERE = Path(__file__).resolve().parent              # .../Regression
REPO_ROOT = HERE.parent                             # .../Emerging-Credit-Risk_1

# ============================================================
# Helpers
# ============================================================

def _norm_id(x) -> str:
    return str(x).strip().lower()

def _to_num_series(s: pd.Series) -> pd.Series:
    """Robust numeric parse: handles decimal commas and thousand separators."""
    return pd.to_numeric(
        s.astype(str)
         .str.replace(".", "", regex=False)     # thousands sep
         .str.replace(",", ".", regex=False),   # decimal comma
        errors="coerce"
    )

def _load_returns_from_stockdata(path: Path) -> pd.DataFrame:
    """Load returns from stock workbook.
    Accepts identifiers like firm_id / Company / Ticker and date-price columns like
    date / Date and stock price / Price Close. Computes simple returns within firm.
    Returns: firm_id, date, ret
    """
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Accept several identifier columns
    # Prefer Company because the predictive-crisis merge is company-based in this script
    if "firm_id" in df.columns:
        id_col = "firm_id"
    elif "Company" in df.columns:
        id_col = "Company"
    elif "Ticker" in df.columns:
        id_col = "Ticker"
    else:
        raise ValueError(
            f"{path.name}: expected an identifier column among ['firm_id', 'Company', 'Ticker'] but found {list(df.columns)}"
        )

    # Accept several date/price column names
    date_col = None
    for c in ["date", "Date"]:
        if c in df.columns:
            date_col = c
            break

    price_col = None
    for c in ["stock price", "Price Close", "Close", "PX_LAST"]:
        if c in df.columns:
            price_col = c
            break

    if date_col is None or price_col is None:
        raise ValueError(
            f"{path.name}: expected date column in ['date','Date'] and price column in ['stock price','Price Close','Close','PX_LAST'] but found {list(df.columns)}"
        )

    df = df.rename(columns={id_col: "firm_id", date_col: "date", price_col: "px"}).copy()
    df["firm_id"] = df["firm_id"].map(_norm_id)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["px"] = pd.to_numeric(df["px"], errors="coerce")
    df = df.dropna(subset=["firm_id", "date", "px"]).sort_values(["firm_id", "date"])

    df["ret"] = df.groupby("firm_id")["px"].pct_change()
    out = df[["firm_id", "date", "ret"]].dropna().sort_values(["firm_id", "date"])
    return out

def compute_window_return(df_returns: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Per-firm cumulative return over [start, end] using simple returns."""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    sub = df_returns[(df_returns["date"] >= start_dt) & (df_returns["date"] <= end_dt)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["firm_id", "window_ret"])

    out = (
        sub.groupby("firm_id")["ret"]
           .apply(lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0))
           .reset_index()
           .rename(columns={"ret": "window_ret"})
    )
    return out

def winsorize_window_returns(df_window: pd.DataFrame, lower_q: float, upper_q: float) -> pd.DataFrame:
    """Winsorize event-window returns cross-sectionally within a window."""
    out = df_window.copy()
    if out.empty:
        out["window_ret_raw"] = np.nan
        return out

    out["window_ret_raw"] = out["window_ret"]
    lo = pd.to_numeric(out["window_ret"], errors="coerce").quantile(lower_q)
    hi = pd.to_numeric(out["window_ret"], errors="coerce").quantile(upper_q)
    out["window_ret"] = pd.to_numeric(out["window_ret"], errors="coerce").clip(lower=lo, upper=hi)
    return out

# ============================================================
# USER SETTINGS
# ============================================================

# Baseline window for topic z-scores (calibration)
BASELINE_START_YEAR = 2005
BASELINE_END_YEAR = 2008

# Returns source (ONLY Excel)
STOCKDATA_CANDIDATES = [
    HERE / "Data" / "Stock_data_final.xlsx",
    REPO_ROOT / "Regression" / "Data" / "Stock_data_final.xlsx",
    REPO_ROOT / "Data" / "Stock_data_final.xlsx",
    HERE / "Output" / "Stock_data_final.xlsx",
    HERE / "Data" / "Stockdata1.xlsx",
    REPO_ROOT / "Regression" / "Data" / "Stockdata1.xlsx",
    REPO_ROOT / "Data" / "Stockdata1.xlsx",
    HERE / "Output" / "Stockdata1.xlsx",
]

# Event windows
WINDOW1_START = "2008-09-01"   # 2008Q2
WINDOW1_END   = "2012-06-30"   # 2009Q2

WINDOW2_START = "2015-12-01"   # 2014Q3
WINDOW2_END   = "2018-12-1"   # 2016Q2

# COVID crash window: use the sharp initial shock period only
WINDOW3_START = "2020-12-01"   # 2020Q1
WINDOW3_END   = "2022-09-30"   # 2020Q1


WINDOW4_START = "2024-01-01"
WINDOW4_END   = "2025-01-01"

# Helper to validate event windows
def _validate_window(label: str, start: str, end: str) -> None:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    if end_dt < start_dt:
        raise ValueError(f"{label}: end date {end} is before start date {start}")

# Predictor-quarter ranges (IMPORTANT: predictors must be PRE-event)
# Predictor quarters strictly BEFORE each evaluation window (predictive timing)
WINDOW1_PRED_START = "2005Q1"
WINDOW1_PRED_END   = "2008Q1"

WINDOW2_PRED_START = "2014Q1"
WINDOW2_PRED_END   = "2015Q2"

# Sharper predictive cutoff: all predictor quarters must end before the event starts.
# Event starts on 2020-04-01, so the last strictly pre-event quarter is 2019Q4.
WINDOW3_PRED_START = "2019Q4"
WINDOW3_PRED_END   = "2020Q3"

# Recent-window test: predictors must be strictly before the 2023-06-01 event start.
# Therefore the last admissible quarter is 2023Q1.
WINDOW4_PRED_START = "2022Q1"
WINDOW4_PRED_END   = "2023Q4"

_validate_window("WINDOW1", WINDOW1_START, WINDOW1_END)
_validate_window("WINDOW2", WINDOW2_START, WINDOW2_END)
_validate_window("WINDOW3", WINDOW3_START, WINDOW3_END)
_validate_window("WINDOW4", WINDOW4_START, WINDOW4_END)

# Helper to validate predictor quarter ranges are strictly pre-event
def _quarter_end_ts(q: str) -> pd.Timestamp:
    return pd.Period(q, freq="Q").end_time.normalize()


def _validate_predictive_range(label: str, pred_start: str, pred_end: str, event_start: str) -> None:
    pred_start_p = pd.Period(pred_start, freq="Q")
    pred_end_p = pd.Period(pred_end, freq="Q")
    event_start_dt = pd.to_datetime(event_start)
    if pred_end_p < pred_start_p:
        raise ValueError(f"{label}: predictor end quarter {pred_end} is before predictor start quarter {pred_start}")
    if _quarter_end_ts(pred_end) >= event_start_dt:
        raise ValueError(
            f"{label}: predictor end quarter {pred_end} overlaps or follows event start {event_start}; use a strictly pre-event quarter"
        )

_validate_predictive_range("WINDOW1", WINDOW1_PRED_START, WINDOW1_PRED_END, WINDOW1_START)
_validate_predictive_range("WINDOW2", WINDOW2_PRED_START, WINDOW2_PRED_END, WINDOW2_START)
_validate_predictive_range("WINDOW3", WINDOW3_PRED_START, WINDOW3_PRED_END, WINDOW3_START)
_validate_predictive_range("WINDOW4", WINDOW4_PRED_START, WINDOW4_PRED_END, WINDOW4_START)

# Include firm controls in Table-6 regressions
TABLE6_USE_CONTROLS = True
TABLE6_COV_TYPE = "HC1"  # robust SE

# Winsorize event-window returns before crisis tests
WINSORIZE_EVENT_RETURNS = True
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

# Minimum firm-year topic loading treated as non-noise
MIN_TOPIC_LOADING = 0.01

# Baseline SD settings
MIN_BASELINE_SD = 0
MIN_BASELINE_OBS = 0

# Plot window (graphs only)
PLOT_START_YEAR = 2005
PLOT_END_YEAR = 2025

TOPIC_INCLUDE = [
    "topic_loading_46",
    "topic_loading_66",
    "topic_loading_76",
    "topic_loading_398",
    "topic_loading_186",
    "topic_loading_204",
    "topic_loading_216",
    "topic_loading_221",
    "topic_loading_228",
    "topic_loading_232",
    "topic_loading_244",
    "topic_loading_247",
    "topic_loading_251",
    "topic_loading_289",
    "topic_loading_295",
    "topic_loading_349",
    "topic_loading_416",
    "topic_loading_454",
    "topic_loading_515",
    "topic_loading_518",
    "topic_loading_535",
    "topic_loading_627",
    "topic_loading_828",
    "topic_loading_871",
    "topic_loading_872",
    "topic_loading_1201",
]
TOPIC_EXCLUDE = []
MIN_VALID_PAIRS = 10

# ============================================================
# Step 0: Load data
# ============================================================

COV_CANDIDATES = [
    HERE / "Output" / "quarterly_pairwise_covariance_2025.csv",
    REPO_ROOT / "Regression" / "Output" / "quarterly_pairwise_covariance_2025.csv",
]
COV_PATH = next((p for p in COV_CANDIDATES if p.exists()), None)
if COV_PATH is None:
    raise FileNotFoundError("Could not find quarterly_pairwise_covariance_2025.csv")

TF_CANDIDATES = [
    REPO_ROOT / "Text analytics" / "outputs_textual_factors" / "extraction_summary_ALL_Final_V1.csv",
    REPO_ROOT / "Text analytics" / "Scripts" / "outputs_textual_factors" / "extraction_summary_ALL_Final_V1.csv",
    HERE / "Output" / "extraction_summary_ALL_Final_V1.csv",
]
TF_PATH = next((p for p in TF_CANDIDATES if p.exists()), None)
if TF_PATH is None:
    raise FileNotFoundError("Could not find extraction_summary_ALL_Final_V1.csv")

CTRL_CANDIDATES = [
    HERE / "Data" / "Control_variable_final.xlsx",
    REPO_ROOT / "Regression" / "Data" / "Control_variable_final.xlsx",
    HERE / "Data" / "Controls.xlsx",
    REPO_ROOT / "Regression" / "Data" / "Controls.xlsx",
]
CTRL_PATH = next((p for p in CTRL_CANDIDATES if p.exists()), None)
if CTRL_PATH is None:
    raise FileNotFoundError("Could not find Control_variable_final.xlsx or Controls.xlsx")

print("Loading covariance panel from:", COV_PATH)
print("Loading extraction summary from:", TF_PATH)
print("Loading controls from:", CTRL_PATH)

cov = pd.read_csv(COV_PATH)
tf = pd.read_csv(TF_PATH)
controls_raw = pd.read_excel(CTRL_PATH)

# Normalize IDs
cov["i"] = cov["i"].map(_norm_id)
cov["j"] = cov["j"].map(_norm_id)
if "bank" in tf.columns:
    tf["bank"] = tf["bank"].map(_norm_id)

# ============================================================
# Load returns (Stockdata1.xlsx)
# ============================================================
STOCKDATA_PATH = next((p for p in STOCKDATA_CANDIDATES if p.exists()), None)
if STOCKDATA_PATH is None:
    raise FileNotFoundError("Stock_data_final.xlsx / Stockdata1.xlsx not found.")
print("Loading returns from:", STOCKDATA_PATH)
returns = _load_returns_from_stockdata(STOCKDATA_PATH)
print("Returns loaded:", returns.shape, "firms=", returns["firm_id"].nunique())

# ============================================================
# Controls (firm-year)
# ============================================================
controls_raw.columns = [str(c).strip() for c in controls_raw.columns]

# Normalize likely identifier columns
if "Company name" in controls_raw.columns:
    controls_raw["Company name"] = controls_raw["Company name"].map(_norm_id)
if "Ticker name" in controls_raw.columns:
    controls_raw["Ticker name"] = controls_raw["Ticker name"].map(_norm_id)

_colmap = {c.lower(): c for c in controls_raw.columns}
def _pick(*names):
    for n in names:
        if n.lower() in _colmap:
            return _colmap[n.lower()]
    return None

company_col = _pick("company", "company name", "name")
ticker_col = _pick("ticker", "ticker name", "stock code", "stock_code")
year_col = _pick("year", "year date", "date", "report date")
assets_col = _pick("total assets", "assets")
marketcap_col = _pick("company market capitalization", "market cap", "market capitalization")
capex5y_col = _pick("capital expenditures - total, 5 yr cagr")
netdebt_col = _pick("net debt - mean", "net debt")
cash_col = _pick("cash & cash equivalents - total", "cash & cash equivalents", "cash")
greenrev_col = _pick("company green revenue percentage", "green revenue percentage")
ebitmargin_col = _pick("ebit margin - %", "ebit margin")
netincome_col = _pick("net income after tax", "net income")
common_equity_col = _pick("common equity - total", "common equity")

if company_col is None or year_col is None or assets_col is None:
    raise ValueError(f"Controls file missing required cols. Found: {list(controls_raw.columns)}")

# Use company name as merge key because covariance / text identifiers are company-based in this script
ID_COL = company_col

# Extract year robustly
_year_series = controls_raw[year_col]
year_num = pd.to_numeric(_year_series, errors="coerce")
year_dt = pd.to_datetime(_year_series, errors="coerce").dt.year
year_num = year_num.where((year_num >= 1900) & (year_num <= 2100))
controls_raw["year"] = year_num.fillna(year_dt).astype("Int64")

controls_raw = controls_raw.dropna(subset=[ID_COL, "year"]).copy()
controls_raw["year"] = controls_raw["year"].astype(int)

# Parse numeric columns used in the current script
controls_raw["assets"] = _to_num_series(controls_raw[assets_col])
controls_raw["market_cap"] = _to_num_series(controls_raw[marketcap_col]) if marketcap_col else np.nan
controls_raw["capex_5y_cagr"] = _to_num_series(controls_raw[capex5y_col]) if capex5y_col else np.nan
controls_raw["net_debt"] = _to_num_series(controls_raw[netdebt_col]) if netdebt_col else np.nan
controls_raw["cash_total"] = _to_num_series(controls_raw[cash_col]) if cash_col else np.nan
controls_raw["green_revenue_pct"] = _to_num_series(controls_raw[greenrev_col]) if greenrev_col else np.nan
controls_raw["ebit_margin"] = _to_num_series(controls_raw[ebitmargin_col]) if ebitmargin_col else np.nan
controls_raw["net_income"] = _to_num_series(controls_raw[netincome_col]) if netincome_col else np.nan
controls_raw["common_equity"] = _to_num_series(controls_raw[common_equity_col]) if common_equity_col else np.nan

# Variables used by this script's predictive regressions
controls_raw["log_assets"] = np.log(controls_raw["assets"].where(controls_raw["assets"] > 0))
controls_raw["leverage"] = controls_raw["net_debt"] / controls_raw["assets"]
controls_raw["cash_ratio"] = controls_raw["cash_total"] / controls_raw["assets"]

controls_raw[ID_COL] = controls_raw[ID_COL].astype(str).str.strip().map(_norm_id)

controls_firm_year = (
    controls_raw.groupby([ID_COL, "year"], as_index=False)[["log_assets", "leverage", "cash_ratio"]]
               .mean()
).rename(columns={ID_COL: "firm_id"})
controls_firm_year["firm_id"] = controls_firm_year["firm_id"].map(_norm_id)

print("Loaded controls firm-year table:", controls_firm_year.shape)
print("Controls ID column used:", ID_COL)

# ============================================================
# Step 1: Prepare firm-year TFs
# ============================================================
tf = tf[tf.get("status", "ok") == "ok"].copy()
all_topic_cols = [c for c in tf.columns if c.startswith("topic_loading_")]

if TOPIC_INCLUDE:
    missing = [c for c in TOPIC_INCLUDE if c not in all_topic_cols]
    if missing:
        raise ValueError(f"TOPIC_INCLUDE contains missing columns: {missing}")
    topic_cols = list(TOPIC_INCLUDE)
else:
    topic_cols = [c for c in all_topic_cols if c not in set(TOPIC_EXCLUDE)]

tf_firm_year = (
    tf.groupby(["bank", "year"], as_index=False)[topic_cols]
      .mean()
)
for c in topic_cols:
    tf_firm_year[c] = pd.to_numeric(tf_firm_year[c], errors="coerce")
    tf_firm_year[c] = tf_firm_year[c].where(tf_firm_year[c].abs() >= MIN_TOPIC_LOADING, 0.0)

# ============================================================
# Step 2: Align TFs + controls to quarters
# ============================================================
cov["q_year"] = cov["quarter"].astype(str).str[:4].astype(int)
cov["lag_year"] = cov["q_year"] -1  # same-year alignment

panel = cov.merge(tf_firm_year, left_on=["i", "lag_year"], right_on=["bank", "year"], how="left")
panel = panel.merge(tf_firm_year, left_on=["j", "lag_year"], right_on=["bank", "year"], how="left", suffixes=("_i", "_j"))

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

panel = panel.rename(columns={
    "log_assets_i_ctrl": "log_assets_i",
    "leverage_i_ctrl": "leverage_i",
    "cash_ratio_i_ctrl": "cash_ratio_i",
    "log_assets_j_ctrl": "log_assets_j",
    "leverage_j_ctrl": "leverage_j",
    "cash_ratio_j_ctrl": "cash_ratio_j",
})

# ============================================================
# Step 3: Quarterly covariance regressions and marginal dR2
# ============================================================
results_rows = []

for q, g in panel.groupby("quarter", sort=True):
    y = g["cov_ij_q"].to_numpy(dtype=float)

    Xi = np.column_stack([
        (g["log_assets_i"] * g["log_assets_j"]).to_numpy(dtype=float),
        (g["leverage_i"]   * g["leverage_j"]).to_numpy(dtype=float),
        (g["cash_ratio_i"] * g["cash_ratio_j"]).to_numpy(dtype=float),
    ])

    valid = np.isfinite(y) & np.isfinite(Xi).all(axis=1)
    if valid.sum() < MIN_VALID_PAIRS:
        continue

    y0 = y[valid]
    Xi0 = Xi[valid]

    base = sm.OLS(y0, sm.add_constant(Xi0)).fit()
    r2_controls = float(base.rsquared)

    S = []
    for k in topic_cols:
        s = (g.get(f"{k}_i") * g.get(f"{k}_j")).to_numpy(dtype=float)
        S.append(np.nan_to_num(s, nan=0.0))
    S0 = np.column_stack(S)[valid]

    full_all = sm.OLS(y0, sm.add_constant(np.column_stack([Xi0, S0]))).fit()
    r2_full_all = float(full_all.rsquared)

    k0 = 1 + Xi0.shape[1]
    for idx_k, k in enumerate(topic_cols):
        if S0.shape[1] == 1:
            r2_minus_k = r2_controls
        else:
            S_minus = np.delete(S0, idx_k, axis=1)
            m_minus = sm.OLS(y0, sm.add_constant(np.column_stack([Xi0, S_minus]))).fit()
            r2_minus_k = float(m_minus.rsquared)

        dR2 = r2_full_all - r2_minus_k
        beta_k = float(full_all.params[k0 + idx_k])
        t_k = float(full_all.tvalues[k0 + idx_k])

        results_rows.append({
            "quarter": q,
            "topic": k,
            "beta": beta_k,
            "t_stat": t_k,
            "R2_base": r2_controls,
            "R2_full": r2_full_all,
            "R2_minus_k": r2_minus_k,
            "dR2": dR2,
            "n_pairs": int(valid.sum()),
        })

results = pd.DataFrame(results_rows)
if results.empty:
    raise RuntimeError("No regression results produced. Check data coverage.")

# ============================================================
# Step 4: Topic z-scores (baseline-normalized dR2)
# ============================================================
baseline = results[results["quarter"].astype(str).str[:4].astype(int).between(BASELINE_START_YEAR, BASELINE_END_YEAR)].copy()
if baseline.empty:
    raise ValueError("Baseline period empty. Check BASELINE_START_YEAR/END_YEAR.")

mu = baseline.groupby("topic")["dR2"].mean()
sd = baseline.groupby("topic")["dR2"].std(ddof=0)
cnt = baseline.groupby("topic")["dR2"].count()

sd = sd.where(cnt >= MIN_BASELINE_OBS, np.nan)
sd = sd.clip(lower=MIN_BASELINE_SD)

results["z"] = (results["dR2"] - results["topic"].map(mu)) / results["topic"].map(sd)
results = results.replace([np.inf, -np.inf], np.nan).copy()
results["z_plot"] = results["z"].fillna(0.0).clip(-10, 100000)

# ============================================================
# Outputs\n# ============================================================
out_dir = HERE / "Output"
out_dir.mkdir(exist_ok=True)

out_csv = out_dir / "topic_dr2_zscores_controls.csv"
results.to_csv(out_csv, index=False)
print("Saved:", out_csv)

# ============================================================
# Static risk model (aggregate)\n# ============================================================
static_z = results.groupby("quarter", as_index=False)["z_plot"].agg(z_mean="mean", z_median="median")
static_z["quarter_ts"] = pd.PeriodIndex(static_z["quarter"].astype(str), freq="Q").to_timestamp()
static_z = static_z.sort_values("quarter_ts")
static_z["z_mean_smooth4"] = static_z["z_mean"].rolling(window=4, min_periods=4).mean()

# ============================================================
# Step 5: TABLE 6 cross-sectional return regressions\n# ============================================================
beta_wide = results.pivot_table(index="quarter", columns="topic", values="beta", aggfunc="mean")
for c in topic_cols:
    if c not in beta_wide.columns:
        beta_wide[c] = 0.0
beta_wide = beta_wide[topic_cols]

load_wide = tf_firm_year[["bank", "year"] + topic_cols].copy()
load_wide = load_wide.rename(columns={"bank": "firm_id", "year": "lag_year"})

qmap = pd.DataFrame({"quarter": beta_wide.index.astype(str)})
qmap["q_year"] = qmap["quarter"].astype(str).str[:4].astype(int)
qmap["lag_year"] = qmap["q_year"]

bq = qmap.merge(beta_wide.reset_index(), on="quarter", how="left")
bq = bq.merge(load_wide, on="lag_year", how="inner", suffixes=("_beta", "_load"))

beta_cols = [f"{c}_beta" if f"{c}_beta" in bq.columns else c for c in topic_cols]
load_cols = [f"{c}_load" if f"{c}_load" in bq.columns else c for c in topic_cols]
bq["er_exposure"] = np.sum(bq[beta_cols].to_numpy(float) * bq[load_cols].to_numpy(float), axis=1)

bq = bq.merge(
    controls_firm_year,
    left_on=["firm_id", "lag_year"],
    right_on=["firm_id", "year"],
    how="left"
).drop(columns=["year"])

ret_w1 = compute_window_return(returns, WINDOW1_START, WINDOW1_END)
ret_w2 = compute_window_return(returns, WINDOW2_START, WINDOW2_END)
ret_w3 = compute_window_return(returns, WINDOW3_START, WINDOW3_END)
ret_w4 = compute_window_return(returns, WINDOW4_START, WINDOW4_END)

if WINSORIZE_EVENT_RETURNS:
    ret_w1 = winsorize_window_returns(ret_w1, WINSOR_LOWER, WINSOR_UPPER)
    ret_w2 = winsorize_window_returns(ret_w2, WINSOR_LOWER, WINSOR_UPPER)
    ret_w3 = winsorize_window_returns(ret_w3, WINSOR_LOWER, WINSOR_UPPER)
    ret_w4 = winsorize_window_returns(ret_w4, WINSOR_LOWER, WINSOR_UPPER)

    print(f"Winsorizing event-window returns: True ({WINSOR_LOWER:.0%}/{WINSOR_UPPER:.0%})")
    for label, dfx in [
        ("WINDOW1", ret_w1),
        ("WINDOW2", ret_w2),
        ("WINDOW3", ret_w3),
        ("WINDOW4", ret_w4),
    ]:
        raw = pd.to_numeric(dfx.get("window_ret_raw"), errors="coerce")
        win = pd.to_numeric(dfx.get("window_ret"), errors="coerce")
        print(
            f"{label} raw mean/std=({float(raw.mean(skipna=True)):.6f}, {float(raw.std(skipna=True)):.6f}) | "
            f"winsor mean/std=({float(win.mean(skipna=True)):.6f}, {float(win.std(skipna=True)):.6f})"
        )
else:
    print("Winsorizing event-window returns: False")

def _run_table6(window_name: str, y_df: pd.DataFrame, q_start: str, q_end: str) -> pd.DataFrame:
    out_rows = []
    q_start_p = pd.Period(q_start, freq="Q")
    q_end_p = pd.Period(q_end, freq="Q")

    tmp = bq.merge(y_df, on="firm_id", how="inner")
    tmp["q_period"] = pd.PeriodIndex(tmp["quarter"].astype(str), freq="Q")
    tmp = tmp[(tmp["q_period"] >= q_start_p) & (tmp["q_period"] <= q_end_p)].copy()

    for q, gq in tmp.groupby("quarter", sort=True):
        y = pd.to_numeric(gq["window_ret"], errors="coerce")
        x = pd.to_numeric(gq["er_exposure"], errors="coerce")

        # KEY FIX: standardize exposure within quarter
        x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

        y_below = np.minimum(0.0, y - np.nanmean(y))

        X_cols = {"er_exposure": x}
        if TABLE6_USE_CONTROLS:
            X_cols.update({
                "log_assets": gq.get("log_assets"),
                "leverage": gq.get("leverage"),
                "cash_ratio": gq.get("cash_ratio"),
            })
        X = pd.DataFrame(X_cols)

        msk = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if msk.sum() >= 10:
            fit = sm.OLS(y.loc[msk].to_numpy(float), sm.add_constant(X.loc[msk], has_constant="add")).fit(cov_type=TABLE6_COV_TYPE)
            out_rows.append({"window": window_name, "panel": "A_raw", "quarter": q,
                             "beta_er": float(fit.params.get("er_exposure", np.nan)),
                             "t_er": float(fit.tvalues.get("er_exposure", np.nan)),
                             "n": int(msk.sum())})

        msk2 = np.isfinite(y_below) & np.isfinite(X).all(axis=1)
        if msk2.sum() >= 10:
            fit2 = sm.OLS(np.asarray(y_below[msk2], float), sm.add_constant(X.loc[msk2], has_constant="add")).fit(cov_type=TABLE6_COV_TYPE)
            out_rows.append({"window": window_name, "panel": "B_below_mean", "quarter": q,
                             "beta_er": float(fit2.params.get("er_exposure", np.nan)),
                             "t_er": float(fit2.tvalues.get("er_exposure", np.nan)),
                             "n": int(msk2.sum())})

    return pd.DataFrame(out_rows)


# =======================
# Portfolio-style Table 6 (High/Low spread)
# =======================
def _tstat_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    n = len(s)
    if n < 2:
        return np.nan
    sd = float(np.std(s, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    return float(np.mean(s) / (sd / np.sqrt(n)))


def _run_table6_spreads(window_name: str, y_df: pd.DataFrame, q_start: str, q_end: str) -> pd.DataFrame:
    """Portfolio-style Table 6:
    - Split firms each quarter into High and Low exposure using the within-quarter median.
    - Report mean raw returns, mean below-mean returns, and High-Low spreads.
    - High-Low spread t-stats are computed from a robust OLS of returns on a High dummy.
    """
    out_rows = []
    q_start_p = pd.Period(q_start, freq="Q")
    q_end_p = pd.Period(q_end, freq="Q")

    tmp = bq.merge(y_df, on="firm_id", how="inner")
    tmp["q_period"] = pd.PeriodIndex(tmp["quarter"].astype(str), freq="Q")
    tmp = tmp[(tmp["q_period"] >= q_start_p) & (tmp["q_period"] <= q_end_p)].copy()

    for q, gq in tmp.groupby("quarter", sort=True):
        x = pd.to_numeric(gq["er_exposure"], errors="coerce")
        y = pd.to_numeric(gq["window_ret"], errors="coerce")

        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 10:
            continue

        xq = x.loc[valid].astype(float)
        yq = y.loc[valid].astype(float)
        y_below = np.minimum(0.0, yq - np.nanmean(yq))

        med = float(np.nanmedian(xq))
        high = xq > med
        low = xq <= med

        if high.sum() < 5 or low.sum() < 5:
            continue

        # Raw return portfolios
        y_high = yq.loc[high]
        y_low = yq.loc[low]
        out_rows.append({
            "window": window_name,
            "panel": "High_raw",
            "quarter": q,
            "beta_er": float(np.mean(y_high)),
            "t_er": _tstat_mean(y_high),
            "n": int(len(y_high)),
        })
        out_rows.append({
            "window": window_name,
            "panel": "Low_raw",
            "quarter": q,
            "beta_er": float(np.mean(y_low)),
            "t_er": _tstat_mean(y_low),
            "n": int(len(y_low)),
        })

        fit_spread_raw = sm.OLS(
            yq.to_numpy(float),
            sm.add_constant(high.astype(int).to_numpy(float), has_constant="add")
        ).fit(cov_type=TABLE6_COV_TYPE)
        out_rows.append({
            "window": window_name,
            "panel": "HighMinusLow_raw",
            "quarter": q,
            "beta_er": float(fit_spread_raw.params[1]),
            "t_er": float(fit_spread_raw.tvalues[1]),
            "n": int(len(yq)),
        })

        # Below-mean return portfolios
        yb_high = pd.Series(y_below, index=yq.index).loc[high]
        yb_low = pd.Series(y_below, index=yq.index).loc[low]
        out_rows.append({
            "window": window_name,
            "panel": "High_below_mean",
            "quarter": q,
            "beta_er": float(np.mean(yb_high)),
            "t_er": _tstat_mean(yb_high),
            "n": int(len(yb_high)),
        })
        out_rows.append({
            "window": window_name,
            "panel": "Low_below_mean",
            "quarter": q,
            "beta_er": float(np.mean(yb_low)),
            "t_er": _tstat_mean(yb_low),
            "n": int(len(yb_low)),
        })

        fit_spread_below = sm.OLS(
            np.asarray(y_below, float),
            sm.add_constant(high.astype(int).to_numpy(float), has_constant="add")
        ).fit(cov_type=TABLE6_COV_TYPE)
        out_rows.append({
            "window": window_name,
            "panel": "HighMinusLow_below_mean",
            "quarter": q,
            "beta_er": float(fit_spread_below.params[1]),
            "t_er": float(fit_spread_below.tvalues[1]),
            "n": int(len(yq)),
        })

    return pd.DataFrame(out_rows)



table6_w1 = _run_table6("2008_2009", ret_w1, WINDOW1_PRED_START, WINDOW1_PRED_END)
table6_w2 = _run_table6("2014_2016", ret_w2, WINDOW2_PRED_START, WINDOW2_PRED_END)
table6_w3 = _run_table6("2020_Q1", ret_w3, WINDOW3_PRED_START, WINDOW3_PRED_END)
table6_w4 = _run_table6("2024_2025", ret_w4, WINDOW4_PRED_START, WINDOW4_PRED_END)
table6 = pd.concat([table6_w1, table6_w2, table6_w3, table6_w4], ignore_index=True)

out_table6 = out_dir / "table6_cross_sectional_returns.csv"
table6.to_csv(out_table6, index=False)
print("Saved Table-6 style regressions:", out_table6)

# Portfolio-style High / Low / High-Low version
spread_w1 = _run_table6_spreads("2008_2009", ret_w1, WINDOW1_PRED_START, WINDOW1_PRED_END)
spread_w2 = _run_table6_spreads("2014_2016", ret_w2, WINDOW2_PRED_START, WINDOW2_PRED_END)
spread_w3 = _run_table6_spreads("2020_Q1", ret_w3, WINDOW3_PRED_START, WINDOW3_PRED_END)
spread_w4 = _run_table6_spreads("2024_2025", ret_w4, WINDOW4_PRED_START, WINDOW4_PRED_END)
table6_spreads = pd.concat([spread_w1, spread_w2, spread_w3, spread_w4], ignore_index=True)

out_table6_spreads = out_dir / "table6_portfolio_spreads.csv"
table6_spreads.to_csv(out_table6_spreads, index=False)
print("Saved Table-6 portfolio spreads:", out_table6_spreads)

# ============================================================
# Plot: Static risk model\n# ============================================================
try:
    static_z_plot = static_z.copy()
    static_z_plot["year_plot"] = static_z_plot["quarter_ts"].dt.year
    static_z_plot = static_z_plot[(static_z_plot["year_plot"] >= PLOT_START_YEAR) & (static_z_plot["year_plot"] <= PLOT_END_YEAR)].copy()

    plt.figure(figsize=(14, 4))
    x = np.arange(len(static_z_plot))
    plt.bar(x, static_z_plot["z_mean_smooth4"].values)
    plt.title("Static risk model (4-quarter smoothed average topic z-score)")
    plt.xlabel("Quarter")
    plt.ylabel("Average z-score")

    years = static_z_plot["quarter_ts"].dt.year
    year_idx = years.ne(years.shift()).to_numpy().nonzero()[0]
    plt.xticks(year_idx, years.iloc[year_idx].astype(str).tolist(), rotation=0)

    plt.tight_layout()
    out_png = out_dir / "static_risk_model_avg_z.png"
    plt.savefig(out_png, dpi=200)
    print("Saved plot:", out_png)
    plt.close()
except Exception as e:
    print("WARNING: Failed to plot static risk model:", e)