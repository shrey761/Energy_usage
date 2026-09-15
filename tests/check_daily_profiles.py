import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\energy_data.csv')
df = df[df['timestamp'] != 'TIME_ERROR'].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Build a lookup table of maximum power per (date, hour)
hourly_summary = df.groupby([df['timestamp'].dt.date.rename('date'), df['timestamp'].dt.hour.rename('hour')]).agg({
    'power1': 'max',
    'power2': 'max'
}).reset_index()

hourly_summary['high_r1'] = (hourly_summary['power1'] >= 1200.0).astype(int)
hourly_summary['high_r2'] = (hourly_summary['power2'] >= 1200.0).astype(int)

# Create a next-day lookup: for a given (date, hour), the tomorrow key is (date + 1 day, hour)
hourly_summary['next_date'] = hourly_summary['date'] + pd.Timedelta(days=1)
# Some recordings are 2 days apart (e.g. 2026-08-18 to 2026-08-20)
# Let's check matching next available recorded day at same hour
print("Hourly summary:")
print(hourly_summary)
