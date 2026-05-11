from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path("/Users/sebastianwiegandmoller/PycharmProjects/Speciale_final")

TF_CANDIDATES = [
    REPO_ROOT / "Text analytics" / "Final_TF_Output" / "extraction_summary_ALL_Final_V1.csv",
    REPO_ROOT / "Text analytics" / "Scripts" / "Final_TF_Output" / "extraction_summary_ALL_Final_V1.csv",
]

OUT_DIR = REPO_ROOT / "Text analytics" / "Final_TF_Output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Selected topics
# ============================================================

TOPIC_INCLUDE = [
    "topic_loading_46",
    "topic_loading_66",
    "topic_loading_76",
    "topic_loading_398",
    "topic_loading_112",
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
    "topic_loading_730",
    "topic_loading_230",
    "topic_loading_349",
    "topic_loading_416",
    "topic_loading_294",
    "topic_loading_454",
    "topic_loading_515",
    # "topic_loading_518",
    "topic_loading_535",
    "topic_loading_627",
    "topic_loading_828",
    "topic_loading_871",
    "topic_loading_872",
    "topic_loading_1201",
]

# ============================================================
# Load data
# ============================================================

tf_path = next((p for p in TF_CANDIDATES if p.exists()), None)

if tf_path is None:
    raise FileNotFoundError(
        "Could not find extraction_summary_ALL_Final_V1.csv in any candidate path:\n"
        + "\n".join(str(p) for p in TF_CANDIDATES)
    )

print(f"Loading TF data from: {tf_path}")

df = pd.read_csv(tf_path)

if "year" not in df.columns:
    raise ValueError("CSV must contain a 'year' column.")

topic_cols = [c for c in TOPIC_INCLUDE if c in df.columns]
missing_topics = [c for c in TOPIC_INCLUDE if c not in df.columns]

if missing_topics:
    print("Missing selected topics:", missing_topics)

if not topic_cols:
    raise ValueError("None of the selected topic columns were found.")

print(f"Using {len(topic_cols)} selected topics.")

# ============================================================
# Compute average absolute loading per year
# ============================================================

df["avg_abs_loading"] = df[topic_cols].abs().mean(axis=1)

yearly = (
    df.groupby("year", as_index=False)
      .agg(
          avg_abs_loading=("avg_abs_loading", "mean"),
          n_firms=("file", "nunique")
      )
      .sort_values("year")
)

out_csv = OUT_DIR / "average_selected_topic_loading_by_year.csv"
yearly.to_csv(out_csv, index=False)
print(f"Saved data to: {out_csv}")

# ============================================================
# Plot: same style as z-score bar plots
# ============================================================

all_years = np.arange(int(yearly["year"].min()), int(yearly["year"].max()) + 1)

plot_series = (
    yearly.set_index("year")["avg_abs_loading"]
          .reindex(all_years)
          .fillna(0.0)
)

plt.figure(figsize=(12, 4))
x = np.arange(len(all_years))

plt.bar(x, plot_series.values)

plt.xlabel("Year")
plt.ylabel("Average absolute loading")

plt.xticks(x, all_years.astype(str), rotation=0)

plt.tight_layout()

out_png = OUT_DIR / "average_selected_topic_loading_by_year_bar.png"
plt.savefig(out_png, dpi=200)
print("Saved plot:", out_png)

plt.close()