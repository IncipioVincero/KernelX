import matplotlib.pyplot as plt
import numpy as np

# Define kernel names for comparison
kernels = ['Vanilla CUDA Kernel', 'Inline PTX Kernel']

# Example arithmetic intensity (FLOPs/byte) — Replace with measured values
arithmetic_intensity = [4.0, 6.5]  # e.g., based on estimated memory access and computation

# Example achieved performance in GFLOPs/s — Replace with profiled numbers
achieved_performance = [900, 1450]

# GPU-specific theoretical limits — Update for your actual GPU
peak_performance = 17000  # Theoretical peak in GFLOPs/s (e.g., NVIDIA A100)
memory_bandwidth = 1555   # Memory bandwidth in GB/s (e.g., A100 HBM2)

# X-axis: Arithmetic intensity range
intensity_range = np.logspace(-1, 2, 500)  # From 0.1 to 100 FLOPs/byte

# Roofline curve: min(compute_bound, memory_bound)
roofline = np.minimum(peak_performance, intensity_range * memory_bandwidth)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(intensity_range, roofline, label="Theoretical Roofline", color='black', linewidth=2)
plt.scatter(arithmetic_intensity, achieved_performance, color=['blue', 'red'], s=100)

# Annotate kernel points
for i, kernel in enumerate(kernels):
    plt.annotate(kernel,
                 (arithmetic_intensity[i], achieved_performance[i]),
                 textcoords="offset points",
                 xytext=(10, 10),
                 ha='left',
                 fontsize=10,
                 color='darkblue' if i == 0 else 'darkred')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
plt.ylabel('Performance (GFLOPs/s)', fontsize=12)
plt.title('Roofline Model: Vanilla vs Inline PTX CUDA Kernel', fontsize=14)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

