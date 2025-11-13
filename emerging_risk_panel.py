# emerging_risk_panel.py
# ----------------------------------------------------------
# Builds quarterly bank-pair panel, constructs controls, adds
# placeholder (dummy) risk exposures, runs per-quarter regressions,
# and outputs adjusted R² series ("emerging risk index"), and plot of z-score comapred to baseline (93-03).
# ----------------------------------------------------------

import pandas as pd #read and manipulate the data
import numpy as np #numerical operations (log returns, covariances)
from itertools import combinations #create all pairs of banks (i,j)
import statsmodels.api as sm #run regressions (OLS, get adjusted R²)
import matplotlib.pyplot as plt #make the “emerging risk index” plot
import os, glob #help find CSV file automatically, if not defined already

# ---------------------------
# CONFIGURATION
# ---------------------------
CSV_PATH = "data2.csv"  # If None, auto-detects largest .csv in current directive
RANDOM_SEED = 42 #ensures random dummy exposures are reproducible
N_DUMMY_EXPOSURES = 3 #number of placeholder risk exposures to create
MIN_DAYS_PER_QUARTER = 20 #minimum trading days per quarter to include bank in analysis

np.random.seed(RANDOM_SEED) #Ensures all random numbers (e.g. dummy exposures) are consistent if the script is rerun.

# below function auto-detects CSV file if none specified
def detect_csv():
    files = glob.glob("*.csv")
    if not files:
        raise FileNotFoundError("No CSV files found. Place your CSV in this folder.")
    files = sorted(files, key=lambda p: os.path.getsize(p), reverse=True)
    print(f"Auto-detected CSV: {files[0]}")
    return files[0]

if not CSV_PATH:
    CSV_PATH = detect_csv()

# ---------------------------
# 1) LOAD DATA
# ---------------------------
df = pd.read_csv(CSV_PATH) #load data
df['datadate'] = pd.to_datetime(df['datadate']) #ensure date is datetime
for col in ['cshoc','cshtrd','prccd','prchd','prcld']: #ensure numeric types
    if col in df.columns: #check column exists
        df[col] = pd.to_numeric(df[col], errors='coerce') #convert to numeric, coerce errors to NaN

# ---------------------------
# 2) COLLAPSE TO FIRM-LEVEL DAILY PRICE  (prirow is the IID, e.g., "04W")
# ---------------------------
def collapse_firm_daily(group):
    # Try to match prirow (a string like "04W") to iid
    primary_iid = None
    if 'prirow' in group.columns and group['prirow'].notna().any():
        # most common non-null prirow for this gvkey-date (usually one)
        s = group['prirow'].dropna().astype(str).str.strip() #For the gvkey–datadate group, it reads the primary issue code (e.g., "04W") from prirow. If multiple rows disagree (rare), it takes the mode (most frequent).
        if not s.empty:
            primary_iid = s.mode().iloc[0]

    # If we have a primary iid, select the matching issue by iid
    if primary_iid is not None and 'iid' in group.columns:
        g = group[group['iid'].astype(str).str.strip() == str(primary_iid).strip()] #It then filters the group to only include rows where the iid matches this primary issue code.
        if not g.empty:
            # if multiple rows, prefer the largest cshoc (largest number of shares outstanding)
            g = g.sort_values('cshoc', ascending=False).head(1)
            return pd.Series({
                'prccd_firm': g['prccd'].iloc[0],
                'cshoc_firm': g['cshoc'].iloc[0],
                'cshtrd_firm': g['cshtrd'].iloc[0],
                'gsector': g['gsector'].iloc[0],
                'loc': g['loc'].iloc[0],
                'curcdd': g['curcdd'].iloc[0]
            })

    # Fallback (if no cshoc): market-cap-weighted average across issues for that firm-day. i.e., weighted average price by cshoc (number of shares outstanding)
    g = group.copy()
    w = g['cshoc'].replace(0, np.nan)
    if w.notna().any() and w.sum() > 0:
        price = (g['prccd'] * w).sum() / w.sum()
    else:
        price = g['prccd'].mean()

    # mode helpers for categorical columns
    def mode_or_first(s): #returns the mode (most common value) of a Series, or first value if no mode
        s = s.dropna() #drop NaNs
        return s.mode().iloc[0] if not s.mode().empty else (s.iloc[0] if not s.empty else np.nan) #return mode or first value

    return pd.Series({ #returns a Series with firm-level daily data
        'prccd_firm': price, #weighted average price
        'cshoc_firm': g['cshoc'].mean(), #average shares outstanding
        'cshtrd_firm': g['cshtrd'].mean(), #average shares traded
        'gsector': mode_or_first(g['gsector']), #most common sector
        'loc': mode_or_first(g['loc']), #most common location
        'curcdd': mode_or_first(g['curcdd']) #most common currency
    })

#Yields one row per firm per date
collapsed = (
    df.groupby(['gvkey','datadate'], group_keys=False, sort=False) #group by firm and date
      .apply(collapse_firm_daily)    #apply collapse_firm_daily function
      .reset_index()                 #keep gvkey/datadate as normal columns
)
#Sanity check 

m = df['prirow'].notna() & (df['iid'].astype(str).str.strip() == df['prirow'].astype(str).str.strip()) #boolean Series where prirow matches iid
mismatch_rate = 1 - m.mean() #compute share of rows where prirow matches iid
print("Share of rows where prirow != iid:", mismatch_rate) #print mismatch rate

# ---------------------------
# 3) COMPUTE RETURNS & QUARTERS
# ---------------------------
collapsed = collapsed.sort_values(['gvkey','datadate']).reset_index(drop=True) #sort by gvkey and date

# Make sure ids are simple types
collapsed['gvkey'] = collapsed['gvkey'].astype(str) #ensure gvkey is string

# Sort first
collapsed = collapsed.sort_values(['gvkey','datadate']).reset_index(drop=True) #sort by gvkey and date

# Compute returns in an index-preserving way
collapsed['log_price'] = np.log(collapsed['prccd_firm']) #compute log price
collapsed['ret'] = collapsed.groupby('gvkey', sort=False)['log_price'].diff() #compute log returns per gvkey

# (optional) drop helper
collapsed = collapsed.drop(columns='log_price') #drop log_price column

collapsed['quarter'] = collapsed['datadate'].dt.to_period('Q') #compute quarter identifier
collapsed['mktcap'] = collapsed['prccd_firm'] * collapsed['cshoc_firm'] #compute market cap (used later for size calculation)

# ---------------------------
# 4) QUARTERLY BANK CONTROLS
# ---------------------------
bank_quarter = (
    collapsed.groupby(['gvkey','quarter']).agg( #aggregate to quarterly level
        avg_price=('prccd_firm','mean'), #average daily price in the quarter
        avg_mktcap=('mktcap','mean'), #average daily market cap in the quarter
        turnover=('cshtrd_firm','mean'), #average shares traded
        shares=('cshoc_firm','mean'), #average shares outstanding
        vol=('ret', lambda x: np.nanstd(x, ddof=1)), #volatility of daily returns
        gsector=('gsector', lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]), #most common sector
        loc=('loc', lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]), #most common HQ country
        curcdd=('curcdd', lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]), #most common currency
        n_days=('ret','count') #number of trading days with returns
    )
    .reset_index()
)
bank_quarter = bank_quarter[bank_quarter['n_days'] >= MIN_DAYS_PER_QUARTER].copy() #keep only quarters with enough trading days
bank_quarter['size'] = np.log(bank_quarter['avg_mktcap'].replace({0: np.nan})) #log size (market cap).
bank_quarter['turnover'] = (bank_quarter['turnover'] / bank_quarter['shares']).replace([np.inf,-np.inf], np.nan) #turnover ratio

# ---------------------------
# 5) PLACEHOLDER EXPOSURES
# ---------------------------
def stable_randoms(keys, n_cols, seed=42): #generates stable random numbers based on keys
    uniques = pd.unique(keys) #unique keys
    mat = {} #dictionary to hold random numbers per unique key
    for k in uniques: #for each unique key
        local_seed = abs(hash((str(k), seed))) % (2**32 - 1) #create a local seed based on the key and global seed
        mat[k] = np.random.default_rng(local_seed).random(n_cols) #generate n_cols random numbers
    arr = np.vstack([mat[k] for k in keys]) #stack random numbers in the order of the original keys
    return arr #return as numpy array

keys = bank_quarter['gvkey'].astype(str) + "_" + bank_quarter['quarter'].astype(str) #unique identifier per bank-quarter
exposures = stable_randoms(keys, N_DUMMY_EXPOSURES, seed=RANDOM_SEED) #generate stable random exposures
for k in range(N_DUMMY_EXPOSURES): #add exposures to bank_quarter DataFrame
    bank_quarter[f'exp{k+1}'] = exposures[:,k] #add exposure column

# Lag by one quarter
def lag_by_quarter(df_bq): #lag all specified columns by one quarter per gvkey
    df_bq = df_bq.sort_values(['gvkey','quarter']).copy() #sort by gvkey and quarter
    lag_cols = ['size','turnover','vol'] + [f'exp{i+1}' for i in range(N_DUMMY_EXPOSURES)] #columns to lag
    for c in lag_cols: #lag each column
        df_bq[c+'_lag1'] = df_bq.groupby('gvkey')[c].shift(1) #lag by one quarter
    return df_bq #return lagged DataFrame

bank_quarter = lag_by_quarter(bank_quarter) #apply lagging function
bank_quarter.to_csv("panel_bank_quarter.csv", index=False) #save bank-quarter panel

# ---------------------------
# 6) PAIRWISE COVARIANCES
# ---------------------------
daily_q = collapsed[['gvkey','datadate','quarter','ret']].dropna().copy() #keep only relevant columns
pair_rows = [] #list to hold pairwise covariance rows
for q, qdf in daily_q.groupby('quarter'): #for each quarter
    counts = qdf.groupby('gvkey')['ret'].count() #count trading days per bank
    valid_g = counts[counts >= MIN_DAYS_PER_QUARTER].index.tolist() #banks with enough trading days
    qdf = qdf[qdf['gvkey'].isin(valid_g)] #filter to valid banks
    pivot = qdf.pivot_table(index='datadate', columns='gvkey', values='ret', aggfunc='first') #pivot to wide format
    pivot = pivot.dropna(axis=1, thresh=MIN_DAYS_PER_QUARTER) #drop banks with insufficient data
    gvkeys = list(pivot.columns) #list of valid gvkeys
    if len(gvkeys) < 2: 
        continue #skip quarters with less than 2 banks
    for i,j in combinations(gvkeys,2): #for each bank pair
        rij = pivot[[i,j]].dropna() #drop rows with NaNs for either bank
        if len(rij) < MIN_DAYS_PER_QUARTER: continue #skip pairs with insufficient overlapping data
        cov_ij = np.cov(rij[i], rij[j], ddof=1)[0,1] #compute covariance
        pair_rows.append({'quarter':q, 'gvkey_i':i, 'gvkey_j':j, 'cov_ij':cov_ij}) #store result
pairs = pd.DataFrame(pair_rows) #create DataFrame from pairwise covariance rows

# ---------------------------
# 7) MERGE CONTROLS, BUILD REGR DATA
# ---------------------------
lag_cols = [c for c in bank_quarter.columns if c.endswith('_lag1')] #columns with lagged controls
id_cols = ['gvkey','quarter','gsector','loc','curcdd'] #identifier columns
bq_lag = bank_quarter[id_cols + lag_cols].copy() #bank-quarter lagged controls
bq_i = bq_lag.rename(columns={'gvkey':'gvkey_i'}) #rename for merging
bq_j = bq_lag.rename(columns={'gvkey':'gvkey_j'}) #rename for merging
pairs = pairs.merge(bq_i, on=['gvkey_i','quarter'], how='left') #merge bank i controls
pairs = pairs.merge(bq_j, on=['gvkey_j','quarter'], how='left', suffixes=('_i','_j')) #merge bank j controls

for base in ['size','turnover','vol']: #compute product of controls for bank pairs
    pairs[f'{base}_prod_lag1'] = pairs[f'{base}_lag1_i'] * pairs[f'{base}_lag1_j'] #product of lagged controls
for k in range(N_DUMMY_EXPOSURES): #compute product of dummy exposures for bank pairs
    base = f'exp{k+1}' #exposure base name
    pairs[f'{base}_prod_lag1'] = pairs[f'{base}_lag1_i'] * pairs[f'{base}_lag1_j'] #product of lagged exposures
pairs['same_sector'] = (pairs['gsector_i'] == pairs['gsector_j']).astype(int) #indicator if same sector
pairs['same_country'] = (pairs['loc_i'] == pairs['loc_j']).astype(int) #indicator if same country
pairs['same_currency'] = (pairs['curcdd_i'] == pairs['curcdd_j']).astype(int) #indicator if same currency
pairs = pairs.dropna(subset=['cov_ij']+[c for c in pairs.columns if c.endswith('_lag1')]) #drop rows with missing data
pairs.to_csv("panel_pairs_quarter.csv", index=False) #save pairs-quarter panel

# ---------------------------
# 8) PER-QUARTER REGRESSIONS
# ---------------------------
results = [] #list to hold regression results
control_vars = ['size_prod_lag1','turnover_prod_lag1','vol_prod_lag1',
                'same_sector','same_country','same_currency'] #control variables
exposure_vars = [f'exp{k+1}_prod_lag1' for k in range(N_DUMMY_EXPOSURES)] #exposure variables

for q, qdf in pairs.groupby('quarter'): #for each quarter
    Xc = sm.add_constant(qdf[control_vars], has_constant='add') #design matrix for controls only
    yc = qdf['cov_ij'] #response variable
    try:   
        model_c = sm.OLS(yc, Xc).fit() #fit OLS model with controls only
        adjR2_c = model_c.rsquared_adj #get adjusted R²
    except Exception: 
        adjR2_c = np.nan #if error, set adjusted R² to NaN
    Xf = sm.add_constant(qdf[control_vars+exposure_vars], has_constant='add') #design matrix for full model (controls + exposures)
    yf = qdf['cov_ij'] #response variable
    try:
        model_f = sm.OLS(yf, Xf).fit() #fit OLS model with controls + exposures
        adjR2_f = model_f.rsquared_adj #get adjusted R²
    except Exception:
        adjR2_f = np.nan #if error, set adjusted R² to NaN
    results.append({'quarter':str(q),'adjR2_controls':adjR2_c,'adjR2_full':adjR2_f,
                    'delta_adjR2':(adjR2_f - adjR2_c) if (pd.notna(adjR2_f) and pd.notna(adjR2_c)) else np.nan}) #store results

res_df = pd.DataFrame(results).sort_values('quarter') #create results DataFrame
res_df.to_csv("regression_results.csv", index=False) #save regression results

# ---------------------------
# 9) PLOT EMERGING RISK INDEX (z-score of ΔAdj. R²)
# ---------------------------
plt.figure(figsize=(9,4.5)) #create figure
plt.plot(pd.PeriodIndex(res_df['quarter'], freq='Q').to_timestamp(), res_df['adjR2_controls'], label='Controls only') #plot adjusted R² for controls only
plt.plot(pd.PeriodIndex(res_df['quarter'], freq='Q').to_timestamp(), res_df['adjR2_full'], label='Controls + exposures') #plot adjusted R² for full model
plt.title("Quarterly Adjusted R² (Controls vs +Exposures)") #set title
plt.xlabel("Quarter") #set x-axis label
plt.ylabel("Adjusted R²") #set y-axis label
plt.legend() #add legend
plt.tight_layout() #adjust layout
plt.savefig("emerging_risk_index.png") #save figure
print("All outputs saved: panel_bank_quarter.csv, panel_pairs_quarter.csv, regression_results.csv, emerging_risk_index.png") #notify user of saved outputs

# Choose baseline period (paper uses 1998–2003)
baseline_mask = (res_df['quarter'] >= '1998Q1') & (res_df['quarter'] <= '2003Q4')
baseline = res_df.loc[baseline_mask].copy()

# If baseline has missing delta values, drop them for mean/std
base_vals = baseline['delta_adjR2'].dropna()

# Guard against too few baseline points or zero std
if len(base_vals) >= 3 and base_vals.std() not in (0, None) and not np.isnan(base_vals.std()): #ensure enough data for baseline
    mu = base_vals.mean() #baseline mean
    sigma = base_vals.std() #baseline std
else:
    # Fallback: use entire sample as baseline
    sample_vals = res_df['delta_adjR2'].dropna() #all available delta values
    mu = sample_vals.mean() if len(sample_vals) else np.nan #compute mean
    sigma = sample_vals.std() if len(sample_vals) else np.nan #compute std

# Compute z-score
res_df['z_score'] = (res_df['delta_adjR2'] - mu) / sigma #standardize delta adjusted R²

# Save an extended results file (keeps the original file unchanged)
res_df.to_csv("regression_results_with_z.csv", index=False) #save extended results with z-score

# Plot the standardized Emerging Risk Index (histogram-style time series, like in the article)
quarters = pd.PeriodIndex(res_df['quarter'], freq='Q').to_timestamp()
z_vals = res_df['z_score']

plt.figure(figsize=(10, 5))
plt.bar(
    quarters,
    z_vals,
    width=70,  # width of bars in days
    color=np.where(z_vals >= 0, 'steelblue', 'indianred'),  # blue for positive, red for negative
    edgecolor='black',
    alpha=0.85
)
plt.axhline(0, color='black', linewidth=1)
plt.title("Emerging Risk Index (Standardized ΔAdj. R²)", fontsize=13)
plt.xlabel("Quarter")
plt.ylabel("Z-score of ΔAdj. R²")
plt.tight_layout()
plt.savefig("emerging_risk_index_timeseries_histogram.png")
plt.close()

print("Saved: regression_results_with_z.csv and emerging_risk_index_timeseries_histogram.png")