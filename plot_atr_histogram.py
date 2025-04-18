import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('automated_trades.csv')

# Create a figure with a larger size
plt.figure(figsize=(12, 6))

# Create the histogram using seaborn for better aesthetics
sns.histplot(data=df, x='ATR %', bins=20, kde=True)

# Customize the plot
plt.title('Distribution of ATR Percentage in Trades', fontsize=14, pad=15)
plt.xlabel('ATR Percentage', fontsize=12)
plt.ylabel('Number of Trades', fontsize=12)

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Add some statistics as text
mean_atr = df['ATR %'].mean()
median_atr = df['ATR %'].median()
stats_text = f'Mean: {mean_atr:.3f}\nMedian: {median_atr:.3f}'
plt.text(0.95, 0.95, stats_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adjust layout to prevent text cutoff
plt.tight_layout()

# Save the plot
plt.savefig('atr_percentage_histogram.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some basic statistics
print("\nATR Percentage Statistics:")
print(f"Mean: {mean_atr:.3f}")
print(f"Median: {median_atr:.3f}")
print(f"Min: {df['ATR %'].min():.3f}")
print(f"Max: {df['ATR %'].max():.3f}")
print(f"Standard Deviation: {df['ATR %'].std():.3f}") 