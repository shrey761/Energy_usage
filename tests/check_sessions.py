import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\energy_data.csv')
df = df[df['timestamp'] != 'TIME_ERROR'].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print("Dates and row counts:")
print(df.groupby(df['timestamp'].dt.date).agg(
    count=('timestamp', 'count'),
    min_time=('timestamp', 'min'),
    max_time=('timestamp', 'max'),
    hours=('timestamp', lambda s: sorted(list(s.dt.hour.unique())))
))
