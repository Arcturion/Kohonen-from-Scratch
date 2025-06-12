# High-Performance SOM

This project provides a comparative benchmark for a Self-Organizing Map (SOM) implementation, pitting a standard, vectorized PyTorch version against a high-performance custom kernel written in Triton.

The goal is to demonstrate the performance difference in terms of latency and TFLOPS on modern NVIDIA GPUs.


## Requirements

-   Python 3.10+
-   PyTorch 2.0+ (with CUDA support)
-   Triton

You can install the necessary packages using pip:

```bash
pip install torch triton
```

**Note:** This project is designed to run on an NVIDIA GPU with CUDA capabilities.

## How to Run

To run the benchmark and see the performance comparison, simply execute the `benchmark.py` script from your terminal:

```bash
python benchmark.py
```

## Expected Output

The script will run warm-up iterations followed by timed benchmarks for both implementations. The final output will look similar to this:

```
--- Setting up SOM Benchmark ---
GPU: NVIDIA RTX A6000
Data Shape: (10000, 256)
SOM Grid: 6x6 (36 neurons)
Data Type: torch.float32

--- Benchmarking PyTorch Vectorized ---
Running 5 warm-up iterations...
Running 20 timed iterations...
  > Average Epoch Latency: 15.31 ms
  > Effective TFLOPS: 1.21

--- Benchmarking Triton Custom Kernel ---
Running 5 warm-up iterations...
Running 20 timed iterations...
  > Average Epoch Latency: 4.52 ms
  > Effective TFLOPS: 4.09

--- Comparison Summary ---
Speedup (Triton vs PyTorch): 3.39x
```
