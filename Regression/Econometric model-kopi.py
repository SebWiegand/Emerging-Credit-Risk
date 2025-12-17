import pandas as pd
import numpy as np

# ----------------------------------------------------------
# 1. LOAD DATA (DATA I BAD BTW, BUT USED AS TEST DATA))
# ----------------------------------------------------------
# Import daily TRFD data (with columns like: country, gvkey, date, iid, trfd)
df = pd.read_csv("daily_returns.csv")

# Convert datadate to date
df['date'] = df['datadate']

# Ensure correct types
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# ----------------------------------------------------------
# 2. COMPUTE DAILY RETURNS PER ISSUE
# ----------------------------------------------------------
# Sort by issue (iid) and date
df = df.sort_values(['iid', 'date'])

# Compute daily returns within each issue (TRFD is cumulative index)
df['ret'] = df.groupby('iid')['trfd'].pct_change()

# Keep only what you need (for now)
df = df[['gvkey','date', 'ret']]

# ----------------------------------------------------------
# 3. AGGREGATE TO FIRM-LEVEL RETURNS
# ----------------------------------------------------------
# Some firms (gvkey) have multiple issues per day, so average across them (Discussion to be had? Maybe specific iid?)
df = df.groupby(['gvkey', 'date'], as_index=False)['ret'].mean()

# Drop missing values
df = df.dropna(subset=['ret'])

# ----------------------------------------------------------
# 4. ADD QUARTER IDENTIFIER
# ----------------------------------------------------------
df['quarter'] = df['date'].dt.to_period('Q')

# ----------------------------------------------------------
# 5. BUILD QUARTERLY COVARIANCE MATRICES
# ----------------------------------------------------------
quarterly_covs = {}

for q, data_q in df.groupby('quarter'):
    # Pivot into matrix: rows = daily dates, columns = gvkey (firms)
    pivot = data_q.pivot(index='date', columns='gvkey', values='ret')
    pivot = pivot.dropna(axis=1, how='any')  # drop banks missing data that quarter

    # Skip quarters with too few banks (Maybe not? idk)
    if pivot.shape[1] < 2:
        continue

    # Compute covariance matrix (daily returns → quarterly covariance)
    cov = pivot.cov()
    quarterly_covs[q] = cov

# ----------------------------------------------------------
# 6. FLATTEN COVARIANCE MATRICES INTO LONG DATAFRAME
# ----------------------------------------------------------
cov_list = []

for q, cov in quarterly_covs.items():
    cov.index.name = 'gvkey_i'
    cov.columns.name = 'gvkey_j'

    cov_stacked = cov.stack().reset_index()
    cov_stacked.columns = ['gvkey_i', 'gvkey_j', 'cov']
    cov_stacked['quarter'] = q

    cov_list.append(cov_stacked)

cov_df = pd.concat(cov_list, ignore_index=True)

# ----------------------------------------------------------
# 7. DISPLAY RESULTS
# ----------------------------------------------------------
print("Example covariance matrix for one quarter:")
select_q = list(quarterly_covs.keys())[0]
print(quarterly_covs[select_q].round(5))

print("\nFlattened covariance DataFrame:")
print(cov_df.head())

# ----------------------------------------------------------
# 8. CHECK RETURN DISTRIBUTION (JUST TO CHECK THE DATA, AS IT'S NO GOOD)
# ----------------------------------------------------------
print("\nDaily return summary statistics:")
print(df['ret'].describe())