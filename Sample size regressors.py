

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# ------------------------------------------------------------
# Sample size diagnostic for quarterly pairwise regressions
# ------------------------------------------------------------
# The thresholds are rules of thumb based on observations per regressor.
# Model size: 8 controls + 30 textual themes = 38 regressors.
# Thresholds:
#   10 x 38 = 380  absolute minimum
#   20 x 38 = 760  good minimum
#   30 x 38 = 1140 conservative benchmark


# -----------------------------
# Data
# -----------------------------
years = list(range(2005, 2025))

sample = [
    27, 29, 31, 33, 32, 35, 35, 37, 40, 41,
    45, 46, 46, 46, 48, 50, 48, 50, 51, 49
]

firm_pairs = [
    351, 406, 465, 528, 496, 595, 595, 666, 780, 820,
    990, 1035, 1035, 1035, 1128, 1225, 1128, 1225, 1275, 1176
]

absolute_minimum = [380] * len(years)
good_minimum = [760] * len(years)
conservative_benchmark = [1140] * len(years)

# Create dataframe for transparency and possible export
sample_size_df = pd.DataFrame({
    "Year": years,
    "Sample": sample,
    "Firm pairs": firm_pairs,
    "Minimum threshold (10xK)": absolute_minimum,
    "Recommended threshold (20xK)": good_minimum,
    "Conservative threshold (30xK)": conservative_benchmark,
})


# -----------------------------
# Plot
# -----------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

# Firm pairs and thresholds on left axis
ax1.bar(
    sample_size_df["Year"],
    sample_size_df["Firm pairs"],
    label="Firm pairs",
    color="#1f77b4",
    edgecolor="black",
    linewidth=0.5,
    width=0.70,
)

ax1.plot(
    sample_size_df["Year"],
    sample_size_df["Minimum threshold (10xK)"],
    label="Minimum threshold (10xK)",
    color="#2ca02c",
    linewidth=2.5,
)

ax1.plot(
    sample_size_df["Year"],
    sample_size_df["Recommended threshold (20xK)"],
    label="Recommended threshold (20xK)",
    color="#0b6fa4",
    linewidth=2.5,
)

ax1.plot(
    sample_size_df["Year"],
    sample_size_df["Conservative threshold (30xK)"],
    label="Conservative threshold (30xK)",
    color="#006d5b",
    linewidth=2.5,
)

ax1.set_xlabel("Year")
ax1.set_ylabel("Number of sample firm pairs")
ax1.set_ylim(0, 1400)
ax1.set_xticks(years)
ax1.tick_params(axis="x", rotation=0)

# Sample firms on right axis
ax2 = ax1.twinx()
ax2.plot(
    sample_size_df["Year"],
    sample_size_df["Sample"],
    label="Sample firms (right axis)",
    color="#004b6e",
    linewidth=3.0,
    marker="o",
    markersize=6,
)
ax2.set_ylabel("Number of firms")
ax2.set_ylim(0, 60)


# Grid
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# Combine legends from both axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
legend = ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    ncol=1,
    frameon=True,
    fontsize=10,
)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_alpha(1)

# Add borders around plot area
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_edgecolor("black")

for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_edgecolor("black")

plt.tight_layout()


# -----------------------------
# Save output
# -----------------------------
output_dir = Path("/Users/sebastianwiegandmoller/PycharmProjects/Speciale_final/Regression/Output")
output_dir.mkdir(parents=True, exist_ok=True)

plot_path = output_dir / "sample_size_regressors.png"
csv_path = output_dir / "sample_size_regressors.csv"

plt.savefig(plot_path, dpi=300, bbox_inches="tight")
sample_size_df.to_csv(csv_path, index=False)

print("Saved plot to:", plot_path)
print("Saved data to:", csv_path)

plt.show()