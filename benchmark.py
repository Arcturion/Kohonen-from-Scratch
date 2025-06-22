import torch
from pytorch_som import PyTorchSOM # Keep for baseline comparison
from kohonen_triton import SOM as TritonSOM # Use the new library
import time

# --- Configuration ---
NUM_SAMPLES = 10000
NUM_FEATURES = 256
MAP_SIZE = (6, 6)
NUM_NEURONS = MAP_SIZE[0] * MAP_SIZE[1]
LEARNING_RATE = 0.5
SIGMA = float(max(MAP_SIZE) / 2.0) # Neighborhood radius
DEVICE = 'cuda'
DTYPE = torch.float32

# --- Benchmarking Parameters ---
WARMUP_ITER = 5
BENCH_ITER = 20

def benchmark_som(som_class, name: str, data: torch.Tensor):
    """
    A generic benchmarking function for any SOM implementation.
    """
    print(f"\n--- Benchmarking {name} ---")
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return None, None

    # Instantiate the SOM and move data to GPU
    som_instance = som_class(MAP_SIZE, NUM_FEATURES, DEVICE)
    data_gpu = data.to(DEVICE)

    # Warm-up iterations
    print(f"Running {WARMUP_ITER} warm-up iterations...")
    for _ in range(WARMUP_ITER):
        som_instance.train_epoch(data_gpu, LEARNING_RATE, SIGMA)
    torch.cuda.synchronize()

    # Timed benchmark iterations
    print(f"Running {BENCH_ITER} timed iterations...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(BENCH_ITER):
        som_instance.train_epoch(data_gpu, LEARNING_RATE, SIGMA)
    end_event.record()
    
    torch.cuda.synchronize()

    # Calculate metrics
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / BENCH_ITER
    
    # FLOPS Calculation (simplified for BMU search, the dominant operation)
    # For each sample, for each neuron, we compute distance over features.
    # Dist(a,b)^2 = sum((a_i - b_i)^2). This is 1 sub, 1 mul per feature.
    # So, 2 FLOPs per feature.
    flops_per_epoch = 2 * NUM_SAMPLES * NUM_NEURONS * NUM_FEATURES
    
    # Note: The weight update step adds more FLOPs, but BMU search is a
    # standard and consistent proxy for comparing the bulk of the work.
    
    tflops = (flops_per_epoch / (avg_latency_ms / 1000)) / 1e12

    print(f"  > Average Epoch Latency: {avg_latency_ms:.2f} ms")
    print(f"  > Effective TFLOPS: {tflops:.2f}")
    
    return avg_latency_ms, tflops

def main():
    """Main execution function."""
    print("--- Setting up SOM Benchmark ---")
    
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "N/A (CUDA not found)"

    print(f"GPU: {gpu_name}")
    print(f"Data Shape: ({NUM_SAMPLES}, {NUM_FEATURES})")
    print(f"SOM Grid: {MAP_SIZE[0]}x{MAP_SIZE[1]} ({NUM_NEURONS} neurons)")
    print(f"Data Type: {DTYPE}")
    
    # Generate dummy data on CPU first
    dummy_data = torch.rand(NUM_SAMPLES, NUM_FEATURES, dtype=DTYPE)

    # Run benchmarks
    pytorch_latency, _ = benchmark_som(PyTorchSOM, "PyTorch Vectorized", dummy_data)
    triton_latency, _ = benchmark_som(TritonSOM, "Triton Custom Kernel", dummy_data)
    
    # Final comparison
    if pytorch_latency is not None and triton_latency is not None:
        speedup = pytorch_latency / triton_latency
        print("\n--- Comparison Summary ---")
        print(f"Speedup (Triton vs PyTorch): {speedup:.2f}x")

if __name__ == "__main__":
    main()
