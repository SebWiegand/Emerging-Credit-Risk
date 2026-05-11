import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt

# Robust path: Regression/Data/Stockdata1.xlsx (relative to this script file)
DATA_PATH = Path(__file__).resolve().parent / "Data" / "Stock_data_final.xlsx"

# Stock_data_final.xlsx should contain columns like: Ticker, Company, Date, Price Close
# Read the first sheet by default.
df = pd.read_excel(DATA_PATH, engine="openpyxl")
print("Loaded:", DATA_PATH)
print(df.head())

# Standardize column names (handles minor variations)
df.columns = [c.strip().lower() for c in df.columns]

# --------------------------------------------------
# Standardize expected column names
# Input example: Ticker, Company, Date, Price Close
# --------------------------------------------------
col_map = {
    "ticker": "ticker",
    "company": "company",
    "date": "date",
    "price close": "price_close",
    "price_close": "price_close",
    "close": "price_close",
    "adj close": "price_close",
    "adj_close": "price_close",
    "price": "price_close",
    "stock price": "price_close",
    "stock_price": "price_close",
}

# Apply mapping where possible
df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})

required = {"company", "date", "price_close"}
missing = required - set(df.columns)
if missing:
    raise ValueError(
        f"Missing required columns {sorted(missing)} in {DATA_PATH}. "
        f"Found columns: {list(df.columns)}"
    )

def to_numeric_eu(x):
    """Parse numbers that may use '.' as thousands sep and ',' as decimal sep."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "null":
        return np.nan
    s = s.replace(" ", "")
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    return pd.to_numeric(s, errors="coerce")

# Parse types
df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
df["price_close"] = df["price_close"].apply(to_numeric_eu)

df = df.dropna(subset=["company", "date", "price_close"])
df = df[df["price_close"] > 0].copy()

# --------------------------------------------------
# Restrict sample window
# --------------------------------------------------
START_DATE = pd.Timestamp("2005-01-01")
END_DATE = pd.Timestamp("2025-12-31")

# Keep only observations within the requested window (inclusive)
df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]

# --------------------------------------------------
# Step 1: Quarterly pairwise covariance of daily returns
# (computed using overlapping dates within each quarter)
# --------------------------------------------------

MIN_OVERLAP_DAYS = 5  # adjust if you want stricter/looser filtering

# 1) Clean dtypes
df["date"] = pd.to_datetime(df["date"]).dt.normalize()  # keep date only

# 2) Drop missing essentials
df = df.dropna(subset=["company", "date", "price_close"])

# 3) De-duplicate: keep one price per (company, date)
df = df.sort_values(["company", "date"])
df = (
    df.groupby(["company", "date"], as_index=False)
      .agg({"price_close": "last"})
)

# 4) Compute daily log returns per company
df = df.sort_values(["company", "date"])
df["log_ret"] = np.log(
    df["price_close"] / df.groupby("company")["price_close"].shift(1)
)
df = df.dropna(subset=["log_ret"])
# Filter extreme daily returns that are likely data errors (e.g., unadjusted splits)
# Keep within [-1.0, +1.0] in log-return space (~ -63% to +172% daily)
df = df[(df["log_ret"] >= -1.0) & (df["log_ret"] <= 1.0)].copy()

# 5) Assign calendar quarter
df["quarter"] = df["date"].dt.to_period("Q")

# Log-return diagnostics (overall)
print("\nLog-return diagnostics (overall):")
print(df["log_ret"].describe(percentiles=[0.001, 0.01, 0.05, 0.95, 0.99, 0.999]))

# 6) Compute pairwise covariance per quarter using overlapping dates
cov_frames = []
for q, qdf in df.groupby("quarter", sort=True):
    # Wide matrix of returns: rows = dates, cols = companies
    r = qdf.pivot_table(index="date", columns="company", values="log_ret", aggfunc="mean")

    # Pairwise overlap counts (how many days both i and j have returns)
    mask = r.notna().astype(int)
    overlap = mask.T @ mask  # (company x company) matrix of overlap days

    # Pairwise covariance computed using only overlapping observations
    cov = r.cov(min_periods=MIN_OVERLAP_DAYS)  # pairwise complete observations by default

    # Stack to long form (robust to axis-name collisions)
    cov_long = (
        cov.rename_axis(index="i", columns="j")
           .stack(future_stack=True)
           .rename("cov_ij_q")
           .reset_index()
    )

    # Add overlap days
    overlap_long = (
        overlap.rename_axis(index="i", columns="j")
              .stack(future_stack=True)
              .rename("n_overlap_days")
              .reset_index()
    )

    # Merge and filter
    out = cov_long.merge(overlap_long, on=["i", "j"], how="left")
    out["quarter"] = q

    # Keep only unique pairs (i < j), require minimum overlap, and drop NaN covariances
    out = out[(out["i"] < out["j"]) & (out["n_overlap_days"] >= MIN_OVERLAP_DAYS)]
    out = out.dropna(subset=["cov_ij_q"])
    # Winsorize covariances within quarter (1% / 99%) to reduce outlier influence
    lo, hi = out["cov_ij_q"].quantile([0.01, 0.99]).tolist()
    out["cov_ij_q"] = out["cov_ij_q"].clip(lower=lo, upper=hi)

    cov_frames.append(out)

cov_qtr = pd.concat(cov_frames, ignore_index=True) if cov_frames else pd.DataFrame(
    columns=["i", "j", "cov_ij_q", "n_overlap_days", "quarter"]
)

# 7) Save output to Regression/Output
OUTPUT_DIR = Path(__file__).resolve().parent / "Output"
OUTPUT_DIR.mkdir(exist_ok=True)

output_path = OUTPUT_DIR / "quarterly_pairwise_covariance_2025.csv"
cov_qtr.to_csv(output_path, index=False)

print(f"Saved quarterly covariance to: {output_path}")

print("\nQuarterly pairwise covariance (head):")
print(cov_qtr.head())

# Quick diagnostics
print("\nUnique companies (post-clean):", df["company"].nunique())
print("Companies (first 50):", sorted(df["company"].unique())[:50])
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Rows per company (top 10):")
print(df.groupby("company")["date"].count().sort_values(ascending=False).head(10))
print("\nCovariance rows:", len(cov_qtr))

# --------------------------------------------------
# Overlap-days diagnostics (average overlap)
# --------------------------------------------------
if len(cov_qtr) > 0:
    avg_overlap = cov_qtr["n_overlap_days"].mean()
    med_overlap = cov_qtr["n_overlap_days"].median()
    p10_overlap = cov_qtr["n_overlap_days"].quantile(0.10)
    p90_overlap = cov_qtr["n_overlap_days"].quantile(0.90)
    print("\nOverlap days (n_overlap_days) summary:")
    print(f"  Mean:   {avg_overlap:.2f}")
    print(f"  Median: {med_overlap:.0f}")
    print(f"  P10:    {p10_overlap:.0f}")
    print(f"  P90:    {p90_overlap:.0f}")
else:
    print("\nOverlap days (n_overlap_days) summary: cov_qtr is empty")


# --------------------------------------------------
# Visual diagnostics (saved to Regression/Output)
# --------------------------------------------------
PLOT_DIR = OUTPUT_DIR
PLOT_DIR.mkdir(exist_ok=True)

# 1) Histogram of log returns
try:
    plt.figure(figsize=(10, 4))
    plt.hist(df["log_ret"].dropna().values, bins=200)
    plt.title("Daily log returns (all firms, all dates)")
    plt.xlabel("log return")
    plt.ylabel("count")
    plt.tight_layout()
    p = PLOT_DIR / "diag_log_returns_hist.png"
    plt.savefig(p, dpi=200)
    print("Saved plot:", p)
    plt.close()
except Exception as e:
    print("WARNING: Failed to plot log returns histogram:", e)

# 2) Histogram of covariances (pooled)
try:
    if len(cov_qtr) > 0:
        plt.figure(figsize=(10, 4))
        plt.hist(cov_qtr["cov_ij_q"].dropna().values, bins=200)
        plt.title("Quarterly pairwise covariances (pooled across quarters)")
        plt.xlabel("covariance")
        plt.ylabel("count")
        plt.tight_layout()
        p = PLOT_DIR / "diag_covariances_hist.png"
        plt.savefig(p, dpi=200)
        print("Saved plot:", p)
        plt.close()
except Exception as e:
    print("WARNING: Failed to plot covariance histogram:", e)

# 3) Time series: mean covariance and mean absolute covariance per quarter
try:
    if len(cov_qtr) > 0:
        tmp = cov_qtr.copy()
        tmp["quarter_ts"] = pd.PeriodIndex(tmp["quarter"].astype(str), freq="Q").to_timestamp()
        qstats = (
            tmp.groupby("quarter_ts", as_index=False)
               .agg(mean_cov=("cov_ij_q", "mean"),
                    mean_abs_cov=("cov_ij_q", lambda s: s.abs().mean()))
               .sort_values("quarter_ts")
        )
        plt.figure(figsize=(12, 4))
        plt.plot(qstats["quarter_ts"], qstats["mean_cov"], label="Mean covariance")
        plt.plot(qstats["quarter_ts"], qstats["mean_abs_cov"], label="Mean |covariance|")
        plt.title("Covariance diagnostics over time")
        plt.xlabel("Quarter")
        plt.ylabel("Covariance")
        plt.legend()
        plt.tight_layout()
        p = PLOT_DIR / "diag_covariance_timeseries.png"
        plt.savefig(p, dpi=200)
        print("Saved plot:", p)
        plt.close()
except Exception as e:
    print("WARNING: Failed to plot covariance time series:", e)

# 4) Time series: number of pairs and mean overlap days per quarter
try:
    if len(cov_qtr) > 0:
        tmp = cov_qtr.copy()
        tmp["quarter_ts"] = pd.PeriodIndex(tmp["quarter"].astype(str), freq="Q").to_timestamp()
        qstats2 = (
            tmp.groupby("quarter_ts", as_index=False)
               .agg(n_pairs=("cov_ij_q", "size"),
                    mean_overlap=("n_overlap_days", "mean"))
               .sort_values("quarter_ts")
        )
        plt.figure(figsize=(12, 4))
        plt.plot(qstats2["quarter_ts"], qstats2["n_pairs"], label="# pairs")
        plt.title("Pairs per quarter")
        plt.xlabel("Quarter")
        plt.ylabel("# pairs")
        plt.tight_layout()
        p = PLOT_DIR / "diag_pairs_per_quarter.png"
        plt.savefig(p, dpi=200)
        print("Saved plot:", p)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(qstats2["quarter_ts"], qstats2["mean_overlap"], label="Mean overlap days")
        plt.title("Mean overlap days per quarter")
        plt.xlabel("Quarter")
        plt.ylabel("Days")
        plt.tight_layout()
        p = PLOT_DIR / "diag_overlap_days_per_quarter.png"
        plt.savefig(p, dpi=200)
        print("Saved plot:", p)
        plt.close()
except Exception as e:
    print("WARNING: Failed to plot pairs/overlap time series:", e)

