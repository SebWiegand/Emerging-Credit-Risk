import os
import pandas as pd

# =====================================================
# USER INPUT (only things you may need to change)
# =====================================================

#
# Path to the folder with the PDF reports.
# Uses the project directory (the directory containing this script) so you don't
# have to hard-code absolute paths.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "Reports")

# Topic model output (resolved relative to this script)
FIRST_DOC_TOPICS = os.path.join(BASE_DIR, "first_doc_topics.csv")

# Market panel (resolved relative to this script)
PANEL_FILE = os.path.join(BASE_DIR, "panel_bank_quarter.csv")

# Output (written next to this script)
OUTPUT_FILE = os.path.join(BASE_DIR, "panel_with_topics.csv")

firm_to_gvkey = {
    "Danske": 15552,
    "DeutscheBank": 15575,
    "ING": 15617,
    "UBS": 144496
}

# =====================================================
# 1. Build document → firm → year mapping
# =====================================================

if not os.path.isdir(REPORT_DIR):
    raise FileNotFoundError(
        f"Report directory not found: {REPORT_DIR}\n"
        f"Expected a folder named 'Reports' next to this script.\n"
        f"If your folder is elsewhere, set REPORT_DIR to the correct absolute path."
    )

files = sorted([f for f in os.listdir(REPORT_DIR) if f.endswith(".pdf")])

rows = []
for doc_id, fname in enumerate(files):
    try:
        year, firm, _ = fname.split("_", 2)
    except ValueError:
        raise ValueError(f"Filename does not match YEAR_FIRM_*.pdf format: {fname}")

    if firm not in firm_to_gvkey:
        raise KeyError(f"Firm '{firm}' not in firm_to_gvkey mapping")

    rows.append({
        "document": doc_id,
        "year": int(year),
        "firm": firm,
        "gvkey": firm_to_gvkey[firm]
    })

doc_map = pd.DataFrame(rows)
print("Document map:")
print(doc_map.head())

# =====================================================
# 2. Attach metadata to topic loadings
# =====================================================

if not os.path.isfile(FIRST_DOC_TOPICS):
    raise FileNotFoundError(
        f"Topic file not found: {FIRST_DOC_TOPICS}\n"
        f"Put 'first_doc_topics.csv' next to this script (in {BASE_DIR}) or update FIRST_DOC_TOPICS to its full path."
    )

if not os.path.isfile(PANEL_FILE):
    raise FileNotFoundError(
        f"Panel file not found: {PANEL_FILE}\n"
        f"Put 'panel_bank_quarter.csv' next to this script (in {BASE_DIR}) or update PANEL_FILE to its full path."
    )

topics = pd.read_csv(FIRST_DOC_TOPICS)
topics = topics.merge(doc_map, on="document", how="left")

if topics[['gvkey','year']].isna().any().any():
    raise ValueError("Some documents could not be mapped to gvkey/year")

topic_cols = [c for c in topics.columns if c.startswith("topic")]

firm_year_topics = (
    topics
    .groupby(["gvkey","year"])[topic_cols]
    .mean()
    .reset_index()
)

print("\nFirm–year topics:")
print(firm_year_topics.head())

# =====================================================
# 3. Expand yearly topics to quarters
# =====================================================

firm_year_topics["quarter"] = pd.PeriodIndex(
    firm_year_topics["year"].astype(str) + "Q1",
    freq="Q"
)

firm_year_topics = firm_year_topics.sort_values(["gvkey","quarter"])

# =====================================================
# 4. Merge into bank–quarter panel
# =====================================================

panel = pd.read_csv(PANEL_FILE)
panel["quarter"] = pd.PeriodIndex(panel["quarter"], freq="Q")

panel = panel.sort_values(["gvkey","quarter"])

panel = pd.merge_asof(
    panel,
    firm_year_topics.drop(columns="year"),
    by="gvkey",
    on="quarter",
    direction="backward"
)

panel.to_csv(OUTPUT_FILE, index=False)

print(f"\nDONE ✔ Output written to: {OUTPUT_FILE}")
print(panel.head())