import pandas as pd

# Load datasets
cluster = pd.read_csv("cluster.csv")
job = pd.read_csv("Job.csv")
map_df = pd.read_csv("Map.csv")
reduce_df = pd.read_csv("Reduce.csv")

# -------------------------------
# STEP 1: Aggregate Map & Reduce
# -------------------------------
map_agg = map_df.groupby("jobId").mean(numeric_only=True).reset_index()
reduce_agg = reduce_df.groupby("jobId").mean(numeric_only=True).reset_index()

# -------------------------------
# STEP 2: Merge datasets
# -------------------------------
# Merge using LEFT JOIN instead of INNER
df = pd.merge(cluster, job, on="jobId", how="left")
df = pd.merge(df, map_agg, on="jobId", how="left")
df = pd.merge(df, reduce_agg, on="jobId", how="left")
# Aggregate per jobId (VERY IMPORTANT)
df = df.groupby("jobId").mean(numeric_only=True).reset_index()

df['jobDuration'] = df['finishTime_y'] - df['startedTime']
df = df.drop(columns=['startedTime', 'finishTime_y'], errors='ignore')
# -------------------------------
# STEP 3: Clean data
# -------------------------------
# Drop non-useful columns
drop_cols = [col for col in df.columns if "timestamp" in col.lower()]
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df.fillna(0,inplace=True)
# -------------------------------
# STEP 4: Feature selection
# -------------------------------
features = [col for col in df.columns if (
    'cpu' in col.lower() or
    'mem' in col.lower() or
    'time' in col.lower() or
    'duration' in col.lower()
)]

df = df[features]

# -------------------------------
# STEP 5: Create label (Straggler)
# -------------------------------
threshold = df.mean().mean()

df['label'] = df.mean(axis=1).apply(lambda x: 1 if x > threshold else 0)

# -------------------------------
# STEP 6: Save final dataset
# -------------------------------
df.to_csv("final_merged_data.csv", index=False)

print("✅ FINAL DATA READY")
print(df.head())
