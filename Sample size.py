from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Resolve project root
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]   # go back to project root

# Load extraction summary
path = ROOT / "Text analytics/outputs_textual_factors/extraction_summary_ALL_Final_V1.csv"
df = pd.read_csv(path)

# 2) Pick the company identifier column (use what you actually have)
# Common ones in your files: "bank" or "firm_id"
ID_COL = "bank" if "bank" in df.columns else "firm_id"


# 4) Count total annual reports per year (number of rows per year)
sample_reports = (
    df.groupby("year")
      .size()
      .reset_index(name="n_reports")
      .sort_values("year")
)

# Plot number of annual reports per year (single graph, no title)
plt.figure(figsize=(12, 4))

bars = plt.bar(sample_reports["year"].astype(str), sample_reports["n_reports"])

plt.xlabel("Year")
plt.ylabel("Number of annual reports")

for b in bars:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width()/2, h, f"{int(h)}",
             ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()

# Optional: save
# plt.savefig("sample_size_companies_by_year.png", dpi=300, bbox_inches="tight")