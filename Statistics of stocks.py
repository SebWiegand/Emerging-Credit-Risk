

# -*- coding: utf-8 -*-
"""
Summary statistics for stock data and quarterly pairwise covariance.

This script computes:
1) Descriptive statistics for quarterly pairwise covariance
2) Correlation statistics derived from covariance (if returns are available)
3) Firm-level stock return volatility statistics
4) Cross-sectional dispersion measures

Outputs are saved to Regression/Output/
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path("/Users/sebastianwiegandmoller/PycharmProjects/Emerging-Credit-Risk_1")

COV_PATH = PROJECT_ROOT / "Regression" / "Output" / "quarterly_pairwise_covariance.csv"
STOCK_PATH = PROJECT_ROOT / "Regression" / "Data" / "Stockdata1.xlsx"
OUT_DIR = PROJECT_ROOT / "Regression" / "Output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1) QUARTERLY PAIRWISE COVARIANCE STATISTICS
# ============================================================

cov_df = pd.read_csv(COV_PATH)
cov_df.columns = [c.strip() for c in cov_df.columns]

# Detect covariance column automatically
cov_col = None
for c in cov_df.columns:
    if "cov" in c.lower():
        cov_col = c
        break

if cov_col is None:
    raise ValueError("Could not detect covariance column in quarterly_pairwise_covariance.csv")

cov_stats = cov_df[cov_col].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

cov_stats.to_csv(OUT_DIR / "covariance_summary_statistics.csv")

print("\nCovariance summary statistics:")
print(cov_stats)

# ============================================================
# 2) STOCK DATA STATISTICS
# ============================================================

stock_df = pd.read_excel(STOCK_PATH)
stock_df.columns = [c.strip() for c in stock_df.columns]

# Attempt to detect date and price columns
date_col = None
price_col = None
firm_col = None

for c in stock_df.columns:
    if "date" in c.lower():
        date_col = c
    if "price" in c.lower() or "close" in c.lower():
        price_col = c
    if "firm" in c.lower() or "company" in c.lower() or "ticker" in c.lower():
        firm_col = c

if date_col is None or price_col is None:
    raise ValueError("Could not detect date or price column in Stockdata1.xlsx")

stock_df[date_col] = pd.to_datetime(stock_df[date_col], errors="coerce")
stock_df = stock_df.dropna(subset=[date_col, price_col])

# Convert price to numeric
stock_df[price_col] = pd.to_numeric(stock_df[price_col], errors="coerce")
stock_df = stock_df.dropna(subset=[price_col])

# Compute daily log returns
stock_df = stock_df.sort_values([firm_col, date_col]) if firm_col else stock_df.sort_values(date_col)

if firm_col:
    stock_df["log_return"] = stock_df.groupby(firm_col)[price_col].transform(
        lambda x: np.log(x) - np.log(x.shift(1))
    )
else:
    stock_df["log_return"] = np.log(stock_df[price_col]) - np.log(stock_df[price_col].shift(1))

stock_df = stock_df.dropna(subset=["log_return"])

# Annualized volatility per firm-year
stock_df["year"] = stock_df[date_col].dt.year

if firm_col:
    vol_df = (
        stock_df.groupby([firm_col, "year"])["log_return"]
        .std()
        .reset_index(name="daily_vol")
    )
else:
    vol_df = (
        stock_df.groupby("year")["log_return"]
        .std()
        .reset_index(name="daily_vol")
    )

vol_df["annualized_vol"] = vol_df["daily_vol"] * np.sqrt(252)

vol_stats = vol_df["annualized_vol"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
vol_stats.to_csv(OUT_DIR / "volatility_summary_statistics.csv")

print("\nVolatility summary statistics:")
print(vol_stats)

# ============================================================
# 3) CROSS-SECTIONAL DISPERSION OF RETURNS
# ============================================================

if firm_col:
    monthly_returns = (
        stock_df
        .groupby([firm_col, stock_df[date_col].dt.to_period("M")])["log_return"]
        .sum()
        .reset_index()
    )

    monthly_returns.rename(columns={date_col: "month"}, inplace=True)

    dispersion = (
        monthly_returns
        .groupby("month")["log_return"]
        .std()
        .reset_index(name="cross_sectional_sd")
    )

    dispersion_stats = dispersion["cross_sectional_sd"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    dispersion_stats.to_csv(OUT_DIR / "dispersion_summary_statistics.csv")

    print("\nCross-sectional dispersion statistics:")
    print(dispersion_stats)

print("\nAll statistics saved to Regression/Output/")