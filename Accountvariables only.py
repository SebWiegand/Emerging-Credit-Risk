from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ============================================================
# Accounting-variables-only covariance model (quarterly)
# Uses:
# - quarterly_pairwise_covariance.csv
# - Control_variable_final.xlsx
# Controls used on both i and j sides:
#   ln_assets, cash, net_debt, ebit_margin, net_income,
#   equity_assets, capex_5y_cagr, green_revenue_pct
# Model each quarter:
#   cov_ij_q ~ (control_i * control_j)
# Output:
# - quarterly R² series
# - yearly R² series
# - Full_sample-style bar chart
# ============================================================

HERE = Path(__file__).resolve().parent

MIN_VALID_PAIRS = 1
START_YEAR = 2005
END_YEAR = 2024

TITLE = "Accounting variables only"
YLABEL = "R²"

OUTPUT_DIR = HERE / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COV_CANDIDATES = [
    HERE / "Output" / "quarterly_pairwise_covariance.csv",
    HERE / "quarterly_pairwise_covariance.csv",
]
CTRL_CANDIDATES = [
    HERE / "Data" / "Control_variable_final.xlsx",
]

COV_PATH = next((p for p in COV_CANDIDATES if p.exists()), None)
if COV_PATH is None:
    raise FileNotFoundError(
        "Could not find quarterly_pairwise_covariance.csv. Tried:\n" + "\n".join(str(p) for p in COV_CANDIDATES)
    )

CTRL_PATH = next((p for p in CTRL_CANDIDATES if p.exists()), None)
if CTRL_PATH is None:
    raise FileNotFoundError(
        "Could not find Control_variable_final.xlsx. Tried:\n" + "\n".join(str(p) for p in CTRL_CANDIDATES)
    )


# ---------------- Helpers ----------------
def _norm_id(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    s = " ".join(s.split())
    return s.lower()


def _to_num(s: pd.Series) -> pd.Series:
    def parse_mixed_number(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        z = str(x).strip().replace("%", "").replace(" ", "")
        if z == "":
            return np.nan
        if "," in z:
            z = z.replace(".", "").replace(",", ".")
        return pd.to_numeric(z, errors="coerce")

    return s.apply(parse_mixed_number)


def safe_scale_1d(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


print("Loading covariance panel from:", COV_PATH)
print("Loading accounting controls from:", CTRL_PATH)
print("RUNNING PATCHED ACCOUNTING SCRIPT")

cov = pd.read_csv(COV_PATH)
controls_raw = pd.read_excel(CTRL_PATH)

for col in ["i", "j"]:
    if col in cov.columns:
        cov[col] = cov[col].astype(str).str.strip().map(_norm_id)

controls_raw.columns = [str(c).strip().lower() for c in controls_raw.columns]

col_map = {
    "company": "company",
    "company name": "company",
    "name": "company",
    "ticker": "ticker",
    "year": "year",
    "year date": "year",
    "date": "date",
    "total assets": "total_assets",
    "total_assets": "total_assets",
    "cash & cash equivalents - total": "cash",
    "cash & cash equivalents": "cash",
    "cash and cash equivalents": "cash",
    "cash": "cash",
    "net debt - mean": "net_debt",
    "net debt": "net_debt",
    "net_debt": "net_debt",
    "ebit margin - %": "ebit_margin",
    "ebit margin": "ebit_margin",
    "ebit_margin": "ebit_margin",
    "net income after tax": "net_income",
    "net income": "net_income",
    "net_income": "net_income",
    "common equity - total": "common_equity",
    "common equity": "common_equity",
    "common_equity": "common_equity",
    "capital expenditures - total, 5 yr cagr": "capex_5y_cagr",
    "capex_5y_cagr": "capex_5y_cagr",
    "company green revenue percentage": "green_revenue_pct",
    "green revenue percentage": "green_revenue_pct",
    "green_revenue_pct": "green_revenue_pct",
}
controls_raw = controls_raw.rename(columns={c: col_map.get(c, c) for c in controls_raw.columns})

if "year" not in controls_raw.columns and "date" in controls_raw.columns:
    controls_raw["year"] = pd.to_datetime(controls_raw["date"], errors="coerce", dayfirst=True).dt.year

if "year" in controls_raw.columns:
    y = controls_raw["year"]
    y_num = pd.to_numeric(y, errors="coerce")
    share_plain_year = float(((y_num >= 1900) & (y_num <= 2100)).mean()) if len(y_num) else 0.0
    if share_plain_year > 0.8:
        controls_raw["year"] = y_num
    else:
        y_dt = pd.to_datetime(y, errors="coerce", dayfirst=True)
        if y_dt.notna().mean() > 0.5:
            controls_raw["year"] = y_dt.dt.year
        else:
            y_dt2 = pd.to_datetime(y_num, unit="D", origin="1899-12-30", errors="coerce")
            controls_raw["year"] = y_dt2.dt.year

if "company" not in controls_raw.columns:
    raise ValueError("Control_variable_final.xlsx must contain a company/company name column.")

controls_raw["company"] = controls_raw["company"].astype(str).str.strip().map(_norm_id)
controls_raw["year"] = pd.to_numeric(controls_raw["year"], errors="coerce")
controls_raw["year"] = controls_raw["year"].where((controls_raw["year"] >= 1900) & (controls_raw["year"] <= 2100))
controls_raw = controls_raw.dropna(subset=["company", "year"]).copy()
controls_raw["year"] = controls_raw["year"].astype(int)

for c in [
    "total_assets", "cash", "net_debt", "ebit_margin", "net_income",
    "common_equity", "capex_5y_cagr", "green_revenue_pct"
]:
    if c in controls_raw.columns:
        controls_raw[c] = _to_num(controls_raw[c])

controls_raw["ln_assets"] = np.log(controls_raw["total_assets"].where(controls_raw["total_assets"] > 0))
controls_raw["equity_assets"] = controls_raw["common_equity"] / controls_raw["total_assets"].replace({0: np.nan})

need = [
    "ln_assets", "cash", "net_debt", "ebit_margin", "net_income",
    "equity_assets", "capex_5y_cagr", "green_revenue_pct"
]
missing = [c for c in need if c not in controls_raw.columns]
if missing:
    raise ValueError(f"Missing required control variables in Control_variable_final.xlsx: {missing}")

controls_firm_year = (
    controls_raw
    .groupby(["company", "year"], as_index=False)[need]
    .mean()
    .rename(columns={"company": "firm_id"})
)
controls_firm_year["firm_id"] = controls_firm_year["firm_id"].map(_norm_id)

print("Controls firm-year rows:", len(controls_firm_year))
print("Unique firms in controls:", controls_firm_year["firm_id"].nunique())
print("Unique years in controls:", controls_firm_year["year"].nunique())
print("Earliest control year:", controls_firm_year["year"].min())
print("Latest control year:", controls_firm_year["year"].max())
coverage_by_year = controls_firm_year.groupby("year")[need].apply(lambda x: x.notna().mean())
coverage_csv = OUTPUT_DIR / "accounting_control_coverage_by_year.csv"
coverage_by_year.to_csv(coverage_csv)
print("Saved control coverage by year:", coverage_csv)

cov["q_year"] = cov["quarter"].astype(str).str[:4].astype(int)
if START_YEAR is not None:
    cov = cov[cov["q_year"] >= int(START_YEAR)]
if END_YEAR is not None:
    cov = cov[cov["q_year"] <= int(END_YEAR)]

print("Covariance quarter range in raw filtered file:", cov["quarter"].min(), "to", cov["quarter"].max())
print("Number of covariance rows after year filter:", len(cov))

# Same-year alignment
cov["lag_year"] = cov["q_year"]

panel = cov.merge(
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

ctrl_cols = [
    "ln_assets_i", "cash_i", "net_debt_i", "ebit_margin_i", "net_income_i", "equity_assets_i", "capex_5y_cagr_i", "green_revenue_pct_i",
    "ln_assets_j", "cash_j", "net_debt_j", "ebit_margin_j", "net_income_j", "equity_assets_j", "capex_5y_cagr_j", "green_revenue_pct_j",
]
missing_after_merge = [c for c in ctrl_cols if c not in panel.columns]
if missing_after_merge:
    raise ValueError(f"Missing merged control columns: {missing_after_merge}")

coverage = panel[ctrl_cols].notna().mean().mean()
print("Controls coverage (avg non-missing across i/j controls):", float(coverage))

quarter_pair_counts = panel.groupby("quarter").size().reset_index(name="n_rows")
print("Merged panel quarter range:", quarter_pair_counts["quarter"].min(), "to", quarter_pair_counts["quarter"].max())
print("First 12 merged quarters and row counts:")
print(quarter_pair_counts.head(12).to_string(index=False))

results_rows = []
for q, g in panel.groupby("quarter", sort=True):
    y = pd.to_numeric(g["cov_ij_q"], errors="coerce").to_numpy(dtype=float)

    Xi_raw = np.column_stack([
        (pd.to_numeric(g["ln_assets_i"], errors="coerce") * pd.to_numeric(g["ln_assets_j"], errors="coerce")).to_numpy(dtype=float),
        (pd.to_numeric(g["cash_i"], errors="coerce") * pd.to_numeric(g["cash_j"], errors="coerce")).to_numpy(dtype=float),
        (pd.to_numeric(g["net_debt_i"], errors="coerce") * pd.to_numeric(g["net_debt_j"], errors="coerce")).to_numpy(dtype=float),
        (pd.to_numeric(g["ebit_margin_i"], errors="coerce") * pd.to_numeric(g["ebit_margin_j"], errors="coerce")).to_numpy(dtype=float),
        (pd.to_numeric(g["net_income_i"], errors="coerce") * pd.to_numeric(g["net_income_j"], errors="coerce")).to_numpy(dtype=float),
        (pd.to_numeric(g["equity_assets_i"], errors="coerce") * pd.to_numeric(g["equity_assets_j"], errors="coerce")).to_numpy(dtype=float),
        (pd.to_numeric(g["capex_5y_cagr_i"], errors="coerce") * pd.to_numeric(g["capex_5y_cagr_j"], errors="coerce")).to_numpy(dtype=float),
        (pd.to_numeric(g["green_revenue_pct_i"], errors="coerce") * pd.to_numeric(g["green_revenue_pct_j"], errors="coerce")).to_numpy(dtype=float),
    ])

    # Use as much data as possible: only require finite covariance.
    valid = np.isfinite(y)
    if int(valid.sum()) < MIN_VALID_PAIRS:
        continue

    y0 = y[valid]
    Xi0_raw = Xi_raw[valid]

    # Scale each interaction using available values, then fill remaining missing values with 0.
    Xi0 = np.column_stack([safe_scale_1d(Xi0_raw[:, j]) for j in range(Xi0_raw.shape[1])])
    Xi0 = np.where(np.isfinite(Xi0), Xi0, 0.0)

    X = sm.add_constant(Xi0, has_constant="add")
    try:
        reg = sm.OLS(y0, X).fit(cov_type="HC1")
    except Exception:
        continue

    results_rows.append({
        "quarter": q,
        "R2_base": float(reg.rsquared),
        "adj_r2": float(reg.rsquared_adj),
        "n_pairs": int(valid.sum()),
    })

if results_rows:
    debug_df = pd.DataFrame(results_rows)
    print("First 12 estimated quarters:")
    print(debug_df[["quarter", "n_pairs"]].head(12).to_string(index=False))

results = pd.DataFrame(results_rows)
if results.empty:
    raise RuntimeError("No accounting-only regression results were produced. Check merge coverage.")

results["quarter_ts"] = pd.PeriodIndex(results["quarter"].astype(str), freq="Q").to_timestamp()
results["year"] = results["quarter_ts"].dt.year

print("Accounting-only results rows:", len(results))
print("Quarter range:", results["quarter"].min(), "to", results["quarter"].max())
print("R2_base min/max:", results["R2_base"].min(), results["R2_base"].max())
print("Unique years in estimated results:", sorted(results["year"].unique().tolist()))

quarterly_out = results[["quarter", "quarter_ts", "year", "R2_base", "adj_r2", "n_pairs"]].copy()
quarterly_csv = OUTPUT_DIR / "accounting_only_r2_quarterly.csv"
quarterly_out.to_csv(quarterly_csv, index=False)
print("Saved quarterly accounting-only series:", quarterly_csv)

fig_df = results.groupby("year", as_index=False)[["R2_base"]].mean()
fig_csv = OUTPUT_DIR / "accounting_only_r2_yearly.csv"
fig_df.to_csv(fig_csv, index=False)
print("Saved yearly accounting-only series:", fig_csv)

# --- Plot in same style as Full_sample_V1.py ---
plt.rcParams["axes.grid"] = False

# Full quarterly timeline
all_quarters = pd.period_range(results["quarter"].min(), results["quarter"].max(), freq="Q").to_timestamp()

# Mean R2 per quarter and reindex to full timeline
s = results.groupby("quarter_ts")["R2_base"].mean().reindex(all_quarters)

# For plotting: keep full axis; inactive quarters show as 0-height bars
r2_plot = s.fillna(0.0)

# 4-quarter rolling smoothing (presentation only)
r2_smooth4 = r2_plot.rolling(window=4, min_periods=2).mean()

plt.figure(figsize=(12, 4))
x = np.arange(len(all_quarters))
plt.bar(x, r2_smooth4.values)

plt.title(TITLE)
plt.xlabel("Year")
plt.ylabel(YLABEL)

# Year-only x-axis labels (one tick per year)
years = pd.Series(all_quarters).dt.year
year_idx = years.ne(years.shift()).to_numpy().nonzero()[0]
plt.xticks(year_idx, years.iloc[year_idx].astype(str).tolist(), rotation=0)

plt.tight_layout()

png_path = OUTPUT_DIR / "figure_accounting_variables_only.png"
plt.savefig(png_path, dpi=200)
plt.close()
print("Saved figure:", png_path)
print("Dependent variable: cov_ij_q")
