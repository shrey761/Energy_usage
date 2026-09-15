import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\energy_data.csv')
df = df[df['timestamp'] != 'TIME_ERROR'].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print("Room 1 Power Value Quantiles / Value Counts (binned):")
print(pd.cut(df['power1'], bins=[-1, 50, 500, 1000, 1500, 2000]).value_counts().sort_index())

print("\nRoom 2 Power Value Quantiles / Value Counts (binned):")
print(pd.cut(df['power2'], bins=[-1, 50, 500, 1000, 1500, 2000]).value_counts().sort_index())
