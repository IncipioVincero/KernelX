import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load benchmark results
df = pd.read_csv("benchmark_results.csv")

# Roofline parameters (adjust to your GPU's specs)
peak_gflops = 7000    # e.g., A100: ~7000 GFLOPs FP32
peak_bandwidth = 1550 # GB/s

# Derived roofline lines
oi = np.logspace(-1, 4, 100)
gflops_bound = np.minimum(peak_bandwidth * oi, peak_gflops)

# Plot
plt.figure(figsize=(10, 7))
plt.loglog(oi, gflops_bound, label='Roofline', color='gray', linestyle='--')
plt.xlabel('Operational Intensity [FLOPs/Byte]')
plt.ylabel('Performance [GFLOP/s]')
plt.title('Roofline Model')
plt.grid(True, which='both', ls='--', lw=0.5)

# Plot data points
for _, row in df.iterrows():
    plt.scatter(row['OperationalIntensity'], row['GFLOPs'], label=row['Program'], s=100)

plt.legend()
plt.tight_layout()
plt.savefig("roofline_plot.png")
plt.show()

