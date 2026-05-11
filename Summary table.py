import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# Table 3 – Summary statistics (A and B)
# ============================================================

HERE = Path(__file__).resolve().parent

# Robustly detect repo root by walking upwards until we find the 'Regression' folder
REPO_ROOT = None
for p in [HERE] + list(HERE.parents):
    if (p / "Regression").exists() and (p / "Regression").is_dir():
        REPO_ROOT = p
        break
if REPO_ROOT is None:
    raise FileNotFoundError(
        f"Could not locate repo root above {HERE}. Expected a 'Regression' folder in a parent directory."
    )

DATA_DIR = REPO_ROOT / "Regression" / "Data"

# Controls file (financial controls). Prefer Controls.xlsx; otherwise try to auto-detect.
CANDIDATE_CONTROLS = [
    DATA_DIR / "Control_variable_final.xlsx",
    DATA_DIR / "Control_variable_final.xlsx",
]

PATH_CONTROLS = None
for cand in CANDIDATE_CONTROLS:
    if cand.exists():
        PATH_CONTROLS = cand
        break

# Fallback: pick the first xlsx file that looks like controls (but NOT stock price files)
if PATH_CONTROLS is None:
    for cand in sorted(DATA_DIR.glob("*.xlsx")):
        name = cand.name.lower()
        if "control" in name and "stock" not in name and "vix" not in name and "epu" not in name:
            PATH_CONTROLS = cand
            break

if PATH_CONTROLS is None:
    raise FileNotFoundError(
        f"Could not find a Controls Excel file in {DATA_DIR}. "
        f"Expected Controls.xlsx (or a file containing 'control' in the name)."
    )

# Stock price file for building index volatility (optional)
PATH_STOCK = DATA_DIR / "Stock_data_final.xlsx"

# Path to VIX (Euro Stoxx) index data
PATH_VIX = REPO_ROOT / "Regression" / "Data" / "VIX Europa.xlsx"

PATH_COV = REPO_ROOT / "Regression" / "Output" / "quarterly_pairwise_covariance.csv"

# Optional reference to EPU index script
PATH_EPU_SCRIPT = REPO_ROOT / "Regression" / "EPU_index.py"  # optional reference

print("Loaded controls file:", PATH_CONTROLS)

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

ctr = pd.read_excel(PATH_CONTROLS)

# Sanity check: controls file should contain firm fundamentals like Total Assets
if "Total Assets" not in ctr.columns and "total assets" not in [str(c).lower() for c in ctr.columns]:
    raise ValueError(
        f"The selected controls file {PATH_CONTROLS.name} does not appear to be a controls dataset. "
        "It is missing 'Total Assets'. Make sure you point PATH_CONTROLS to your fundamentals/controls file."
    )

# Clean column names (strip + remove non-breaking spaces)
ctr.columns = [str(c).replace("\u00a0", " ").strip() for c in ctr.columns]


# Detect company and year/date columns (new format)
if "Company name" in ctr.columns:
    ctr = ctr.rename(columns={"Company name": "company"})
elif "company" not in ctr.columns:
    raise ValueError("Could not find company column.")

# Prefer a direct Year column if present
if "Year" in ctr.columns:
    ctr["year"] = pd.to_numeric(ctr["Year"], errors="coerce")
elif "year" in ctr.columns:
    ctr["year"] = pd.to_numeric(ctr["year"], errors="coerce")
elif "Year date" in ctr.columns:
    ctr["Date"] = pd.to_datetime(ctr["Year date"], errors="coerce", dayfirst=True)
    ctr["year"] = ctr["Date"].dt.year
elif "Date" in ctr.columns:
    ctr["Date"] = pd.to_datetime(ctr["Date"], errors="coerce", dayfirst=True)
    ctr["year"] = ctr["Date"].dt.year
else:
    raise ValueError("Could not find a year/date column. Expected one of: 'Year', 'year', 'Year date', or 'Date'.")

# Drop rows without company or year
ctr = ctr.dropna(subset=["company", "year"]).copy()
ctr["year"] = ctr["year"].astype(int)

# ------------------------------------------------------------
# Remove artificial year-end rows with no financial data
# (e.g. 31.12.YYYY rows where key financial fields are NULL)
# We require at least Total Assets to be present
# ------------------------------------------------------------
ctr["Total Assets"] = pd.to_numeric(ctr["Total Assets"], errors="coerce")
ctr = ctr.dropna(subset=["Total Assets"]).copy()
print("Rows after removing NULL year-end rows:", len(ctr))

# ------------------------------------------------------------
# Section A – Firm characteristics
# ------------------------------------------------------------

# ln(Total assets)
ctr["ln_total_assets"] = np.log(pd.to_numeric(ctr["Total Assets"], errors="coerce"))

# Capital expenditures (use available 5Y CAGR proxy)
capex_col = None
for name in [
    "Capital Expenditures - Total, 5 Yr CAGR",
    "Capital Expenditures - Total. 5 Yr CAGR"
]:
    if name in ctr.columns:
        capex_col = name
        break

ctr["capital_expenditures"] = pd.to_numeric(
    ctr[capex_col], errors="coerce"
) if capex_col else np.nan

# Firm age = current year − first observed year per firm + 1
first_year = ctr.groupby("company")["year"].transform("min")
ctr["firm_age"] = ctr["year"] - first_year + 1

# ------------------------------------------------------------
# Section B – Financial structure
# ------------------------------------------------------------

# Debt / total assets
if "Total Debt Percentage of Total Assets" in ctr.columns:
    ctr["debt_assets"] = pd.to_numeric(
        ctr["Total Debt Percentage of Total Assets"], errors="coerce"
    )
elif "Debt - Total" in ctr.columns:
    ctr["debt_assets"] = (
        pd.to_numeric(ctr["Debt - Total"], errors="coerce") /
        pd.to_numeric(ctr["Total Assets"], errors="coerce")
    )
else:
    ctr["debt_assets"] = np.nan

# Cash / total assets
if "Cash & Cash Equivalents to Total Assets" in ctr.columns:
    ctr["cash_assets"] = pd.to_numeric(
        ctr["Cash & Cash Equivalents to Total Assets"], errors="coerce"
    )
elif "Cash & Cash Equivalents - Total" in ctr.columns:
    ctr["cash_assets"] = (
        pd.to_numeric(ctr["Cash & Cash Equivalents - Total"], errors="coerce") /
        pd.to_numeric(ctr["Total Assets"], errors="coerce")
    )
else:
    ctr["cash_assets"] = np.nan

# EBIT margin (%)
ctr["ebit_margin"] = pd.to_numeric(
    ctr["EBIT Margin - %"], errors="coerce"
)


# Capital (Equity / assets)
ctr["capital_equity_assets"] = (
    pd.to_numeric(ctr["Common Equity - Total"], errors="coerce") /
    pd.to_numeric(ctr["Total Assets"], errors="coerce")
)

# ------------------------------------------------------------
# Panel D – Renewable / geographic exposure variables
# ------------------------------------------------------------

 # Green revenue percentage (robust column detection + numeric parsing)
green_col = None
for c in ctr.columns:
    name = str(c).replace("\u00a0", " ").strip().lower()
    if name == "company green revenue percentage":
        green_col = c
        break

if green_col is not None:
    ctr["green_revenue_pct"] = (
        ctr[green_col]
        .astype(str)
        .str.strip()
        .str.replace("NULL", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    ctr["green_revenue_pct"] = pd.to_numeric(ctr["green_revenue_pct"], errors="coerce")
else:
    ctr["green_revenue_pct"] = np.nan

# ESG score (robust)
esg_col = None
for c in ctr.columns:
    name = str(c).replace("\u00a0", " ").strip().lower()
    if name == "esg score":
        esg_col = c
        break

if esg_col is not None:
    ctr["esg_score"] = (
        ctr[esg_col]
        .astype(str)
        .str.strip()
        .str.replace("NULL", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    ctr["esg_score"] = pd.to_numeric(ctr["esg_score"], errors="coerce")
else:
    ctr["esg_score"] = np.nan


# Debug print for non-missing counts
print("Green revenue pct non-missing:", int(pd.to_numeric(ctr["green_revenue_pct"], errors="coerce").notna().sum()))
print("ESG score non-missing:", int(pd.to_numeric(ctr["esg_score"], errors="coerce").notna().sum()))

# Extra diagnostics: show candidate columns and sample raw values
print("\n[Diag] Columns containing 'green':", [c for c in ctr.columns if 'green' in str(c).lower()])
print("[Diag] Columns containing 'esg':", [c for c in ctr.columns if 'esg' in str(c).lower()])

if green_col is not None:
    raw = ctr[green_col].astype(str).str.strip().head(10).tolist()
    print("[Diag] Sample raw Green Revenue values:", raw)
else:
    print("[Diag] green_col not detected")

if esg_col is not None:
    raw = ctr[esg_col].astype(str).str.strip().head(10).tolist()
    print("[Diag] Sample raw ESG values:", raw)
else:
    print("[Diag] esg_col not detected")

# How many rows are non-empty strings before numeric coercion?
if green_col is not None:
    nonempty = ctr[green_col].astype(str).str.strip().replace({'NULL':'', 'nan':'', 'NaN':''}).ne('').sum()
    print("[Diag] Green Revenue non-empty raw cells:", int(nonempty))
if esg_col is not None:
    nonempty = ctr[esg_col].astype(str).str.strip().replace({'NULL':'', 'nan':'', 'NaN':''}).ne('').sum()
    print("[Diag] ESG non-empty raw cells:", int(nonempty))

# ------------------------------------------------------------
# Stock return volatility (annual, index-level)
# ------------------------------------------------------------

if PATH_STOCK.exists():
    stock = pd.read_excel(PATH_STOCK)
    stock.columns = [str(c).strip() for c in stock.columns]

    # Detect date column
    if "date" in stock.columns:
        stock["date"] = pd.to_datetime(stock["date"], errors="coerce", dayfirst=True)
    elif "Date" in stock.columns:
        stock["date"] = pd.to_datetime(stock["Date"], errors="coerce", dayfirst=True)
    else:
        # fall back: first column that looks like date
        stock["date"] = pd.to_datetime(stock[stock.columns[1]], errors="coerce", dayfirst=True)

    # Detect price column (prefer one containing 'price')
    price_col = None
    for c in stock.columns:
        if "price" in str(c).lower():
            price_col = c
            break
    if price_col is None:
        # assume third column if present (Company, date, price)
        price_col = stock.columns[2] if len(stock.columns) > 2 else stock.columns[-1]

    stock["price"] = (
        stock[price_col]
        .astype(str)
        .str.strip()
        .str.replace("NULL", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    stock["price"] = pd.to_numeric(stock["price"], errors="coerce")

    stock = stock.dropna(subset=["date", "price"]).copy()

    # If multiple companies exist, aggregate to a single index price per day
    stock = stock.groupby("date", as_index=False)["price"].mean().sort_values("date")

    # Daily log returns for the index
    stock["log_return"] = np.log(stock["price"]).diff()

    # ------------------------------------------------------------
    # Monthly return volatility (index-level)
    # ------------------------------------------------------------

    stock["month"] = stock["date"].dt.to_period("M")

    # Monthly log returns (sum of daily log returns per month)
    monthly = (
        stock.dropna(subset=["log_return"])
             .groupby("month")["log_return"]
             .sum()
             .reset_index(name="monthly_log_return")
    )

    monthly["year"] = monthly["month"].dt.year

    # SD of monthly returns per year
    monthly_sd = (
        monthly.groupby("year")["monthly_log_return"]
               .std(ddof=1)
               .reset_index(name="sd_monthly_returns")
    )

    ctr = ctr.merge(monthly_sd, on="year", how="left")

    # Financial-only monthly volatility proxy (currently same as overall)
    ctr["sd_monthly_returns_fin"] = ctr["sd_monthly_returns"]

    stock["year"] = stock["date"].dt.year

    vol_year = (
        stock.dropna(subset=["log_return"])
             .groupby("year")["log_return"]
             .std(ddof=1)
             .reset_index()
    )
    vol_year = vol_year.rename(columns={"log_return": "stock_return_volatility"})

    # Optional annualization (uncomment if desired)
    # vol_year["stock_return_volatility"] = vol_year["stock_return_volatility"] * np.sqrt(252)

    # Merge year-level volatility onto controls by year
    ctr = ctr.merge(vol_year, on="year", how="left")
else:
    print("Stockdata1.xlsx not found — skipping stock return volatility.")
    ctr["stock_return_volatility"] = np.nan
    if "sd_monthly_returns" not in ctr.columns:
        ctr["sd_monthly_returns"] = np.nan
    ctr["sd_monthly_returns_fin"] = np.nan

if "stock_return_volatility" not in ctr.columns:
    ctr["stock_return_volatility"] = np.nan
if "sd_monthly_returns" not in ctr.columns:
    ctr["sd_monthly_returns"] = np.nan

# ------------------------------------------------------------
# VIX index (Euro Stoxx / V2TX) – annual average level
# Source: VIX Europa.xlsx (Date, Symbol, Indexvalue)
# ------------------------------------------------------------

if PATH_VIX.exists():
    vix = pd.read_excel(PATH_VIX)
    vix.columns = [str(c).strip() for c in vix.columns]

    # Standardize column names
    # Expect: Date, Symbol, Indexvalue
    date_col = "Date" if "Date" in vix.columns else ("date" if "date" in vix.columns else vix.columns[0])
    sym_col = "Symbol" if "Symbol" in vix.columns else ("symbol" if "symbol" in vix.columns else None)
    val_col = "Indexvalue" if "Indexvalue" in vix.columns else ("IndexValue" if "IndexValue" in vix.columns else None)

    vix["date"] = pd.to_datetime(vix[date_col], errors="coerce", dayfirst=True)

    if val_col is None:
        # fallback: pick first numeric-like column not date/symbol
        candidates = [c for c in vix.columns if c not in [date_col, sym_col]]
        val_col = candidates[-1]

    vix["value"] = (
        vix[val_col]
        .astype(str)
        .str.strip()
        .str.replace("NULL", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    vix["value"] = pd.to_numeric(vix["value"], errors="coerce")

    vix = vix.dropna(subset=["date", "value"]).copy()

    # Filter symbol if available (V2TX = EURO STOXX 50 Volatility Index)
    if sym_col is not None and sym_col in vix.columns:
        vix = vix[vix[sym_col].astype(str).str.upper() == "V2TX"].copy()

    vix["year"] = vix["date"].dt.year

    vix_year = (
        vix.groupby("year")["value"]
           .mean()
           .reset_index(name="vix_index")
    )

    # Avoid _x/_y suffixes
    if "vix_index" in ctr.columns:
        ctr = ctr.drop(columns=["vix_index"])

    ctr = ctr.merge(vix_year, on="year", how="left")
else:
    print("VIX Europa.xlsx not found — skipping VIX index.")
    if "vix_index" not in ctr.columns:
        ctr["vix_index"] = np.nan

# Ensure column exists
if "vix_index" not in ctr.columns:
    ctr["vix_index"] = np.nan

# ------------------------------------------------------------
# Pairwise covariance (annual aggregates from quarterly data)
# ------------------------------------------------------------

if PATH_COV.exists():
    cov_df = pd.read_csv(PATH_COV)
    cov_df.columns = [str(c).strip() for c in cov_df.columns]

    if "quarter" in cov_df.columns and "cov_ij_q" in cov_df.columns:
        # Extract year from quarter string (e.g. 2005Q1)
        cov_df["year"] = cov_df["quarter"].astype(str).str[:4].astype(int)
        cov_df["cov_ij_q"] = pd.to_numeric(cov_df["cov_ij_q"], errors="coerce")

        # Annual mean covariance
        cov_year_mean = (
            cov_df.groupby("year")["cov_ij_q"]
                  .mean()
                  .reset_index(name="avg_pair_covariance")
        )

        # Annual std of covariance
        cov_year_sd = (
            cov_df.groupby("year")["cov_ij_q"]
                  .std(ddof=1)
                  .reset_index(name="sd_pair_covariance")
        )

        # Merge into controls by year
        ctr = ctr.merge(cov_year_mean, on="year", how="left")
        ctr = ctr.merge(cov_year_sd, on="year", how="left")
    else:
        print("quarterly_pairwise_covariance.csv missing required columns.")
        ctr["avg_pair_covariance"] = np.nan
        ctr["sd_pair_covariance"] = np.nan
else:
    print("quarterly_pairwise_covariance.csv not found — skipping covariance statistics.")
    ctr["avg_pair_covariance"] = np.nan
    ctr["sd_pair_covariance"] = np.nan

# Ensure columns exist
if "avg_pair_covariance" not in ctr.columns:
    ctr["avg_pair_covariance"] = np.nan
if "sd_pair_covariance" not in ctr.columns:
    ctr["sd_pair_covariance"] = np.nan

# ------------------------------------------------------------
# EPU index (annual, from monthly data)
# ------------------------------------------------------------

# First try: use already-built yearly_epu.csv if it exists
PATH_YEARLY_EPU = REPO_ROOT / "Regression" / "Output" / "yearly_epu.csv"

ctr["epu_index"] = np.nan

if PATH_YEARLY_EPU.exists():
    epu_yearly = pd.read_csv(PATH_YEARLY_EPU)
    epu_yearly.columns = [str(c).strip() for c in epu_yearly.columns]
    # Expect columns: year, epu_yearly
    if "year" in epu_yearly.columns:
        if "epu_yearly" not in epu_yearly.columns:
            # fallback: try epu_mean
            if "epu_mean" in epu_yearly.columns:
                epu_yearly["epu_yearly"] = epu_yearly["epu_mean"]
        if "epu_yearly" in epu_yearly.columns:
            epu_yearly["year"] = pd.to_numeric(epu_yearly["year"], errors="coerce").astype("Int64")
            epu_yearly["epu_yearly"] = pd.to_numeric(epu_yearly["epu_yearly"], errors="coerce")
            epu_yearly = epu_yearly.dropna(subset=["year", "epu_yearly"]).copy()
            epu_yearly = epu_yearly[["year", "epu_yearly"]].rename(columns={"epu_yearly": "epu_index"})

            # Avoid _x/_y suffixes
            if "epu_index" in ctr.columns:
                ctr = ctr.drop(columns=["epu_index"])

            ctr = ctr.merge(epu_yearly, on="year", how="left")
else:
    # If yearly file isn't present, try to build from a monthly EPU file (CSV/XLSX)
    EPU_FILE_CANDIDATES = [
        REPO_ROOT / "Regression" / "Data" / "epu_monthly.csv",
        REPO_ROOT / "Regression" / "Data" / "EPU.csv",
        REPO_ROOT / "Regression" / "Data" / "European_News_Index.csv",
        REPO_ROOT / "Data" / "epu_monthly.csv",
        REPO_ROOT / "Regression" / "Data" / "epu_monthly.xlsx",
        REPO_ROOT / "Regression" / "Data" / "EPU_index.xlsx",
        REPO_ROOT / "Regression" / "Data" / "European_News_Index.xlsx",
        REPO_ROOT / "Data" / "epu_monthly.xlsx",
    ]

    def _find_epu_path(cands):
        for p in cands:
            if p.exists():
                return p
        return None

    epupath = _find_epu_path(EPU_FILE_CANDIDATES)
    if epupath is None:
        print("No EPU file found and yearly_epu.csv missing — skipping EPU index.")
    else:
        print("Building yearly EPU from:", epupath)
        if epupath.suffix.lower() in [".xlsx", ".xls"]:
            df_epu = pd.read_excel(epupath)
        else:
            df_epu = pd.read_csv(epupath)
        df_epu.columns = [str(c).strip() for c in df_epu.columns]

        # Build monthly date either from Year/Month columns or a single date column
        year_col = "Year" if "Year" in df_epu.columns else ("year" if "year" in df_epu.columns else None)
        month_col = "Month" if "Month" in df_epu.columns else ("month" if "month" in df_epu.columns else None)

        if year_col and month_col:
            y = pd.to_numeric(df_epu[year_col], errors="coerce")
            m = pd.to_numeric(df_epu[month_col], errors="coerce")
            dt = pd.to_datetime(
                y.astype("Int64").astype(str) + "-" + m.astype("Int64").astype(str).str.zfill(2),
                format="%Y-%m",
                errors="coerce",
            )
            if "European_News_Index" in df_epu.columns:
                value_col = "European_News_Index"
            else:
                # fallback: first numeric column not year/month
                num_cols = [c for c in df_epu.columns if pd.api.types.is_numeric_dtype(df_epu[c])]
                value_col = None
                for c in num_cols:
                    lc = str(c).lower()
                    if lc in ["year", "month"]:
                        continue
                    value_col = c
                    break
                if value_col is None:
                    value_col = df_epu.columns[-1]
        else:
            # fallback: detect a date-ish column
            date_col = None
            for c in df_epu.columns:
                lc = str(c).lower()
                if "date" in lc or "month" in lc or "time" in lc or "period" in lc:
                    date_col = c
                    break
            if date_col is None:
                date_col = df_epu.columns[0]
            dt = pd.to_datetime(df_epu[date_col], errors="coerce")
            if "European_News_Index" in df_epu.columns:
                value_col = "European_News_Index"
            else:
                # pick last column
                value_col = df_epu.columns[-1]

        epu_series = pd.to_numeric(df_epu[value_col], errors="coerce")
        tmp = pd.DataFrame({"date": dt, "epu": epu_series}).dropna(subset=["date", "epu"])
        tmp["year"] = tmp["date"].dt.year.astype(int)

        epu_yearly = (
            tmp.groupby("year")["epu"].mean().reset_index(name="epu_index")
        )

        if "epu_index" in ctr.columns:
            ctr = ctr.drop(columns=["epu_index"])

        ctr = ctr.merge(epu_yearly, on="year", how="left")

# Ensure column exists
if "epu_index" not in ctr.columns:
    ctr["epu_index"] = np.nan

# ------------------------------------------------------------
# Build summary statistics function
# ------------------------------------------------------------

def summary_stats(series):
    s = pd.to_numeric(series, errors="coerce")
    return {
        "Mean": s.mean(),
        "SD": s.std(ddof=1),
        "Min": s.min(),
        "Median": s.median(),
        "Max": s.max(),
        "# obs.": s.notna().sum(),
    }

# ------------------------------------------------------------
# Winsorization helpers (reduce outlier influence)
# ------------------------------------------------------------

def winsor_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lo, hi)

# Apply winsorization / capping to problematic variables

# Cap EBIT margin to [-200, 200]
if "ebit_margin" in ctr.columns:
    ctr["ebit_margin"] = pd.to_numeric(ctr["ebit_margin"], errors="coerce")
    ctr["ebit_margin"] = ctr["ebit_margin"].clip(-200, 200)

# Winsorize debt/assets at 1% tails
if "debt_assets" in ctr.columns:
    ctr["debt_assets"] = winsor_series(ctr["debt_assets"], p=0.01)

# ------------------------------------------------------------
# Variables for Table 3 (A and B)
# ------------------------------------------------------------

variables = {
    # A. Firm characteristics
    "ln(Total assets)": "ln_total_assets",
    "CapEx growth (5Y CAGR)": "capital_expenditures",
    "Firm age": "firm_age",

    # B. Financial structure
    "Debt / total assets": "debt_assets",
    "Cash / total assets": "cash_assets",
    "EBIT margin (%)": "ebit_margin",
    "Capital (Equity/assets)": "capital_equity_assets",
    "Stock return volatility": "stock_return_volatility",
    "VIX index (Euro Stoxx)": "vix_index",
    "EPU index": "epu_index",
    "Green revenue percentage": "green_revenue_pct",
    "ESG Score": "esg_score",
    "Avg pair covariance": "avg_pair_covariance",
    "SD pair covariance": "sd_pair_covariance",
    "Avg SD monthly returns": "sd_monthly_returns",
    "Avg SD monthly returns (fin. only)": "sd_monthly_returns_fin",
}

rows = []
for label, col in variables.items():
    stats = summary_stats(ctr[col])
    stats["Variable"] = label
    rows.append(stats)

summary_table = pd.DataFrame(rows)[
    ["Variable", "Mean", "SD", "Min", "Median", "Max", "# obs."]
]

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUT_DIR = REPO_ROOT / "Regression" / "Output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

out_path = OUT_DIR / "Table3_summary_A_B.csv"
summary_table.to_csv(out_path, index=False)

print("Saved summary table to:", out_path)
print("\nPreview:")
# Use higher precision so small covariance values are visible
print(summary_table.round(6).to_string(index=False))