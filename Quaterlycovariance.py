import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
# Define the input and output locations used in the script.
# The stock price file is loaded from the same folder as the script, and results are saved in an Output folder.
DATA_PATH = Path(__file__).resolve().parent / "Stock_data_final.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "Output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Settings
# --------------------------------------------------
# These settings control the sample period, the minimum number of overlapping trading days,
# the filter for implausible daily returns, and the winsorization applied to quarterly covariance estimates.
START_DATE = pd.Timestamp("2005-01-01")
END_DATE = pd.Timestamp("2025-12-31")

MIN_OVERLAP_DAYS = 5                # minimum common trading days per firm pair-quarter
MIN_PRICE = 0                       # remove non-positive prices
MIN_LOG_RETURN = -1.0               # lower bound for valid daily log returns
MAX_LOG_RETURN = 1.0                # upper bound for valid daily log returns

WINSOR_LOWER_Q = 0.01               # 1st percentile winsorization within quarter
WINSOR_UPPER_Q = 0.99               # 99th percentile winsorization within quarter

# --------------------------------------------------
# Load data
# --------------------------------------------------
# Read the Excel file containing daily stock prices and standardize the column names.
# This ensures that minor differences in naming conventions do not affect the rest of the script.
df = pd.read_excel(DATA_PATH, engine="openpyxl")
df.columns = [c.strip().lower() for c in df.columns]

col_map = {
    "ticker": "ticker",
    "company": "company",
    "date": "date",
    "price close": "price_close",
}
df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})

required = {"company", "date", "price_close"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# --------------------------------------------------
# Parse European number formatting
# --------------------------------------------------
# Convert price values that may use European decimal notation, such as commas for decimals
# and periods for thousands separators, into standard numeric format.
def to_numeric_eu(x):
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

# --------------------------------------------------
# Clean and restrict the raw stock price data
# --------------------------------------------------
# Convert dates and prices to usable formats, drop incomplete observations, and restrict the sample
# to the analysis period used in the thesis.
df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.normalize()
df["price_close"] = df["price_close"].apply(to_numeric_eu)

df = df.dropna(subset=["company", "date", "price_close"])
df = df[df["price_close"] > MIN_PRICE]
df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]

# --------------------------------------------------
# Remove duplicates and keep one price per firm-date
# --------------------------------------------------
# If multiple observations exist for the same company on the same date, only the last observed closing price is kept.
# This ensures a unique daily price series for each firm.
df = (
    df.sort_values(["company", "date"])
      .groupby(["company", "date"], as_index=False)
      .agg({"price_close": "last"})
)

# --------------------------------------------------
# Compute daily log returns
# --------------------------------------------------
# Convert daily prices into daily log returns, which are used as the building blocks for quarterly covariance.
# Extreme daily returns are removed to reduce the influence of likely data errors such as unadjusted stock splits.
df["log_ret"] = np.log(df["price_close"] / df.groupby("company")["price_close"].shift(1))
df = df.dropna(subset=["log_ret"])
df = df[(df["log_ret"] >= MIN_LOG_RETURN) & (df["log_ret"] <= MAX_LOG_RETURN)].copy()

# Assign each daily return observation to a calendar quarter.
df["quarter"] = df["date"].dt.to_period("Q")

# --------------------------------------------------
# Compute quarterly pairwise covariance
# --------------------------------------------------
# For each quarter, returns are reshaped into a wide matrix where rows are dates and columns are firms.
# Pairwise covariance is then computed using only overlapping trading days between firms within the same quarter.
cov_frames = []

for q, qdf in df.groupby("quarter", sort=True):
    r = qdf.pivot_table(index="date", columns="company", values="log_ret", aggfunc="mean")
    overlap = r.notna().astype(int).T @ r.notna().astype(int)
    cov = r.cov(min_periods=MIN_OVERLAP_DAYS)

    # Convert the covariance matrix to long form so that each row corresponds to one firm pair in one quarter.
    cov_long = (
        cov.rename_axis(index="i", columns="j")
           .stack(future_stack=True)
           .rename("cov_ij_q")
           .reset_index()
    )

    # Convert the overlap matrix to long form so the number of common trading days can be merged onto each pair.
    overlap_long = (
        overlap.rename_axis(index="i", columns="j")
              .stack(future_stack=True)
              .rename("n_overlap_days")
              .reset_index()
    )

    # Keep only unique firm pairs, require a minimum number of overlapping days,
    # and winsorize covariance within quarter to reduce the influence of extreme observations.
    out = cov_long.merge(overlap_long, on=["i", "j"], how="left")
    out["quarter"] = q
    out = out[(out["i"] < out["j"]) & (out["n_overlap_days"] >= MIN_OVERLAP_DAYS)]
    out = out.dropna(subset=["cov_ij_q"])

    lo, hi = out["cov_ij_q"].quantile([WINSOR_LOWER_Q, WINSOR_UPPER_Q]).tolist()
    out["cov_ij_q"] = out["cov_ij_q"].clip(lower=lo, upper=hi)

    cov_frames.append(out)

# --------------------------------------------------
# Combine all quarterly outputs
# --------------------------------------------------
# Concatenate the quarterly firm-pair covariance results into one panel dataset.
# If no valid pairs are found, return an empty DataFrame with the expected structure.
cov_qtr = pd.concat(cov_frames, ignore_index=True) if cov_frames else pd.DataFrame(
    columns=["i", "j", "cov_ij_q", "n_overlap_days", "quarter"]
)

# --------------------------------------------------
# Save final output
# --------------------------------------------------
# Export the quarterly pairwise covariance panel to CSV so it can be used directly
# in the subsequent regression analysis.
output_path = OUTPUT_DIR / "quarterly_pairwise_covariance_2025.csv"
cov_qtr.to_csv(output_path, index=False)

print(f"Saved quarterly covariance to: {output_path}")