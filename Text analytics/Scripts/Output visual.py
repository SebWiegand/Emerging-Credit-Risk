import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH = "extraction_summary_ALL_V1.csv"  # change if needed
df = pd.read_csv(PATH)

# --- choose how to aggregate across banks within a year ---
AGG = "mean"   # "mean" or "median"

# --- Plot average topic loadings by year ---
TOPICS = [197]  # keep unique topics only

# Build column names
topic_cols = [f"topic_loading_{t}" for t in TOPICS]
missing = [c for c in topic_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing topic columns in df: {missing}")

# Aggregate by year (across banks)
if AGG == "mean":
    yearly = df.groupby("year")[topic_cols].mean().sort_index()
elif AGG == "median":
    yearly = df.groupby("year")[topic_cols].median().sort_index()
else:
    raise ValueError("AGG must be 'mean' or 'median'")

# --- Plot average topic loadings by year ---
fig, axes = plt.subplots(len(TOPICS), 1, figsize=(12, 2.2 * len(TOPICS)), sharex=True)
# If only one topic, axes is not a list
if len(TOPICS) == 1:
    axes = [axes]

for ax, t in zip(axes, TOPICS):
    col = f"topic_loading_{t}"
    ax.bar(yearly.index.astype(int), yearly[col].values)
    ax.set_title(f"Topic {t} – average loading by year ({AGG})")
    ax.set_ylabel("Average topic loading")
    ax.grid(True, axis="y", alpha=0.3)

axes[-1].set_xlabel("Year")
plt.tight_layout()
plt.show()