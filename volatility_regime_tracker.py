import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

# Load the CSV file
df = pd.read_csv('automated_trades.csv')

# Convert numeric columns and handle missing values
numeric_columns = ['Outcome %', 'MAE', 'MFE']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values in the numeric columns
df = df.dropna(subset=numeric_columns)

# Calculate Efficiency
df['Efficiency'] = df['Outcome %'] / df['MFE'].replace(0, np.nan)

# Drop rows where Efficiency is NaN (due to MFE being 0)
df = df.dropna(subset=['Efficiency'])

# Normalize MAE, MFE, and Efficiency
scaler = MinMaxScaler()
df['MAE_scaled'] = scaler.fit_transform(df[['MAE']])
df['MFE_scaled'] = scaler.fit_transform(df[['MFE']])
df['Efficiency_scaled'] = scaler.fit_transform(df[['Efficiency']])

# Calculate Trade Quality Score (TQS)
df['TQS'] = 0.5 * df['MFE_scaled'] - 0.3 * df['MAE_scaled'] + 0.2 * df['Efficiency_scaled']

# Group by Volatility Level and calculate statistics
summary = df.groupby('Volatility Level').agg({
    'No.': 'count',  # Total Trades
    'Outcome %': lambda x: (x > 0).mean() * 100,  # Win Rate
    'TQS': ['mean', 'max', 'min', 'std']  # Added std for TQS standard deviation
}).round(2)

# Flatten the multi-level columns
summary.columns = ['Trades', 'Win_Rate', 'Avg_TQS', 'Max_TQS', 'Min_TQS', 'TQS_StdDev']

# Print the summary table
print("\nVolatility Regime Analysis Summary:")
print(tabulate(summary, headers='keys', tablefmt='grid', floatfmt='.2f'))

# Print interpretation
print("\nLower TQS variance indicates consistent trade quality. Higher variance suggests structural inconsistency.")

# Save the summary to CSV
summary.to_csv('volatility_regime_summary.csv')
print("\nSummary saved to 'volatility_regime_summary.csv'") 