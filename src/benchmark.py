import torch
import torch.utils.benchmark as benchmark
from Model_Integration import VanillaConv2D, PTXConv2D


def benchmark_model(model, input_tensor, label, warmup=10, runs=50):
    for _ in range(warmup):
        _ = model(input_tensor)

    torch.cuda.synchronize()
    timer = benchmark.Timer(
        stmt='model(x)',
        setup='from __main__ import model, x',
        globals={'model': model, 'x': input_tensor},
        num_threads=1,
        label=label
    )
    results = timer.blocked_autorange(min_run_time=1.0)
    print(results)
    return results.median, results.stddev

def run_benchmarks():
    device = 'cuda'
    B, C, H, W = 4, 3, 256, 256

    input_tensor = torch.rand(B, C, H, W, device=device)
    kernel = torch.ones(5, 5, device=device) / 25.0

    model_vanilla = VanillaConv2D(kernel).to(device)
    model_ptx = PTXConv2D(kernel).to(device)

    # Run benchmarks
    vanilla_time, vanilla_std = benchmark_model(model_vanilla, input_tensor, "Vanilla CUDA")
    ptx_time, ptx_std = benchmark_model(model_ptx, input_tensor, "Inline PTX CUDA")

    # Print results
    print(f"Vanilla CUDA: {vanilla_time:.3f} ms ± {vanilla_std:.3f}")
    print(f"Inline  PTX: {ptx_time:.3f} ms ± {ptx_std:.3f}")
    print(f"Speedup (PTX over Vanilla): {vanilla_time / ptx_time:.2f}x")

    # Plot
    labels = ['Vanilla CUDA', 'Inline PTX']
    times = [vanilla_time, ptx_time]
    stds = [vanilla_std, ptx_std]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, times, yerr=stds, capsize=8, color=['skyblue', 'salmon'])
    plt.ylabel('Execution Time (ms)')
    plt.title('Convolution Kernel Benchmark (B=4, C=3, H=W=256)')
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2.0, t + 0.5, f"{t:.2f} ms", ha='center', va='bottom')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_benchmarks()
