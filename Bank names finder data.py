from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "New_Data.csv"

print("Looking for data at:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
print(df.columns)

# Keep unique company name – gvkey pairs
name_gvkey = (
    df[["conm", "gvkey"]]
    .dropna()
    .drop_duplicates()
    .sort_values("conm")
)

print("\nCompany names with gvkey:\n")
for _, row in name_gvkey.iterrows():
    print(f"{row['conm']}  |  gvkey={row['gvkey']}")

print(f"\nNumber of unique companies: {name_gvkey['conm'].nunique()}")
print(f"Number of unique gvkeys: {name_gvkey['gvkey'].nunique()}")

# Optional: save lookup table for later merges / inspection
name_gvkey.to_csv(
    BASE_DIR / "company_name_gvkey_lookup.csv",
    index=False
)

# --------------------------------------------------
# Inspect unique banks in extraction_summary_ALL.csv
# --------------------------------------------------

EXTRACTION_PATH = (
    BASE_DIR
    / "Text analytics"
    / "Scripts"
    / "outputs_textual_factors"
    / "extraction_summary_ALL.csv"
)

print("\nLooking for extraction summary at:", EXTRACTION_PATH)
print("Extraction file exists:", EXTRACTION_PATH.exists())

ext = pd.read_csv(EXTRACTION_PATH)

# Ensure bank column exists
if "bank" not in ext.columns:
    raise ValueError("Column 'bank' not found in extraction_summary_ALL.csv")

# Clean bank names for consistent grouping
ext["bank_clean"] = (
    ext["bank"]
    .astype(str)
    .str.upper()
    .str.strip()
)

# Unique banks
unique_banks = (
    ext[["bank_clean"]]
    .drop_duplicates()
    .sort_values("bank_clean")
    .reset_index(drop=True)
)

print("\nUnique banks found in extraction_summary_ALL.csv:\n")
for b in unique_banks["bank_clean"]:
    print(b)

print(f"\nNumber of unique banks in extraction summary: {len(unique_banks)}")

# Save for inspection / manual mapping
unique_banks.to_csv(
    BASE_DIR / "unique_banks_in_extraction_summary.csv",
    index=False
)

BBVA  |  gvkey=15181
SANTANDER  |  gvkey=14140
BARCLAYS  |  gvkey=12673
COMMERZBANK  |  gvkey=15575
CREDITAGRICOLE  |  gvkey=24563
DANSKEBANK  |  gvkey=15552
DEUTSCHE  |  gvkey=15576
DNB  |  gvkey=15538
ERSTE  |  gvkey=214659
KBC  |  gvkey=15703
RAIFFEISEN  |  gvkey=272817
SEB  |  gvkey=15671
HANDELSBANKEN  |  gvkey=15654
SWEDBANK  |  gvkey=24578
UBS  |  gvkey=144496
UNICREDIT  |  gvkey=15549