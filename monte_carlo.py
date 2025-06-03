import pandas as pd, numpy as np, matplotlib.pyplot as plt, random, io, itertools

# Read the CSV file directly
df = pd.read_csv('automated_trades.csv')

# 2)  Convert Outcome % to "R"
df['R'] = df['Outcome %'] / 3.5          # +2.14 R winners, –1 R losers
outcomes = df['R'].to_numpy()

def sim_block(outcomes, n_trades=50, risk_mult=1.0):
    """ Return equity curve (PnL in R) for one bootstrapped block """
    sample = np.random.choice(outcomes, size=n_trades, replace=True)
    return np.cumsum(sample * risk_mult)

def worst_dd(curve):
    peak = curve[0]
    dd   = 0
    for x in curve:
        peak = max(peak, x)
        dd   = min(dd, x-peak)
    return dd

N = 10_000
metrics_full = []
metrics_half = []

for _ in range(N):
    eq_full = sim_block(outcomes, risk_mult=1.0)
    eq_half = sim_block(outcomes, risk_mult=0.5)
    metrics_full.append( (eq_full[-1], worst_dd(eq_full),
                          max(len(list(g)) for k, g in
                              itertools.groupby(pd.Series(eq_full).diff(), lambda z: z<0 and abs(z)==1.0)) ) )
    metrics_half.append( (eq_half[-1], worst_dd(eq_half), 0) )

# 3)  Summaries
pnls_full, dds_full, streaks_full = map(np.array, zip(*metrics_full))
pnls_half, dds_half, _           = map(np.array, zip(*metrics_half))

print("FULL-risk:  mean PnL %.1f R  |  σ %.1f R" % (pnls_full.mean(), pnls_full.std()))
print("           95-pct worst DD  %.1f R" % np.percentile(dds_full, 95))
print("           Prob( ≥8 straight losses ) = %.1f %%"
      % ( (streaks_full>=8).mean()*100 ) )

print("\nHALF-risk: mean PnL %.1f R  |  σ %.1f R" % (pnls_half.mean(), pnls_half.std()))
print("           95-pct worst DD  %.1f R" % np.percentile(dds_half, 95))

# 4)  Visual drill – 10 random equity curves at full risk
plt.figure()
for _ in range(10):
    plt.plot(sim_block(outcomes, risk_mult=1.0))
plt.title("10 Monte-Carlo equity paths (full risk, 50 trades)")
plt.xlabel("Trades"); plt.ylabel("PnL in R")
plt.show()
