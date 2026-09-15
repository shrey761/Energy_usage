import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\energy_data.csv')
df = df[df['timestamp'] != 'TIME_ERROR'].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print("Unique dates present in dataset:")
print(df['timestamp'].dt.date.value_counts().sort_index())

print("\nDistribution of hours present per date:")
ct = pd.crosstab(df['timestamp'].dt.date, df['timestamp'].dt.hour)
print(ct)
