# Kohonen Triton SOM Library

**A PyTorch-based Self-Organizing Map (SOM) library accelerated with custom Triton kernels for high-performance training on NVIDIA GPUs.**

This library provides a `SOM` class that allows you to define, train, and utilize Kohonen Self-Organizing Maps. It leverages the power of Triton to execute performance-critical parts of the SOM algorithm (Best Matching Unit search and weight updates) directly on the GPU, offering significant speedups compared to pure PyTorch implementations for large datasets.

## Features

-   **Triton Accelerated:** High-performance training leveraging custom Triton kernels.
-   **Easy-to-use API:** Simple interface for SOM creation, training, and mapping.
-   **Customizable:** Configure map dimensions, input data dimensionality, and training parameters.
-   **PyTorch Compatible:** Built on PyTorch, integrates seamlessly with PyTorch tensors and ecosystem.
-   **Utilities:** Includes functions for quantization error calculation and mapping data to BMU locations.

## Requirements

-   Python 3.8+
-   PyTorch 2.0+ (with CUDA support)
-   Triton (typically installed with recent PyTorch versions that support it, or `pip install triton`)
-   An NVIDIA GPU with CUDA capabilities is required for Triton acceleration.

## Installation

Currently, this library is not packaged for PyPI. To use it, you can include the `kohonen_triton` directory directly in your project, or install it locally if a `setup.py` is provided in the future.

Assuming the `kohonen_triton` directory is in your Python path:

```python
from kohonen_triton import SOM
import torch
```

## Quick Start

Here's a basic example of how to use the `SOM` class:

```python
import torch
from kohonen_triton import SOM

# Check for CUDA availability
if not torch.cuda.is_available():
    print("CUDA not available. This library requires a CUDA-enabled GPU.")
    exit()

device = 'cuda'

# 1. Configuration
num_samples = 1000
input_features = 64
som_map_size = (8, 8)  # 8x8 grid

# 2. Dummy Data
data = torch.rand(num_samples, input_features, device=device, dtype=torch.float32)

# 3. Initialize SOM
som_model = SOM(
    map_size=som_map_size,
    input_dim=input_features,
    device=device,
    random_seed=42 # For reproducible results
)

print(f"SOM initialized: {som_model.map_rows}x{som_model.map_cols} grid, {som_model.num_neurons} neurons.")

# 4. Train SOM
num_epochs = 20
learning_rate = 0.5
sigma = float(max(som_map_size) / 2.0)

print(f"Training for {num_epochs} epochs...")
som_model.train(data, num_epochs, learning_rate, sigma)
print("Training complete.")

# 5. Map new data
sample_data = data[:5]
bmu_indices = som_model.map_to_bmu_indices(sample_data)
bmu_locations = som_model.map_to_bmu_locations(sample_data)

print("\nBMU indices for first 5 samples:", bmu_indices.tolist())
print("BMU grid locations for first 5 samples:\n", bmu_locations.tolist())

# 6. Calculate Quantization Error
q_error = som_model.quantization_error(data)
print(f"\nQuantization Error: {q_error:.4f}")
```

## API Overview

### `SOM(map_size, input_dim, device='cuda', random_seed=None)`
-   `map_size: tuple[int, int]`: (rows, cols) of the SOM grid.
-   `input_dim: int`: Dimensionality of input vectors.
-   `device: str | torch.device`: Computation device (must be CUDA-compatible for Triton).
-   `random_seed: int | None`: Seed for weight initialization.

### Training
-   `som_model.train(data, num_epochs, initial_learning_rate, initial_sigma)`: Trains the SOM.
-   `som_model.train_epoch(data_batch, learning_rate, sigma)`: Trains for a single epoch (useful for custom training loops or benchmarking).

### Mapping & Data
-   `som_model.get_weights() -> torch.Tensor`: Returns the SOM's weight vectors.
-   `som_model.get_neuron_locations() -> torch.Tensor`: Returns the 2D grid locations of neurons.
-   `som_model.map_to_bmu_indices(data) -> torch.Tensor`: Returns BMU indices for input data.
-   `som_model.map_to_bmu_locations(data) -> torch.Tensor`: Returns BMU grid coordinates for input data.

### Evaluation
-   `som_model.quantization_error(data) -> float`: Calculates the average distance between data points and their BMU weights.

## Future Enhancements
-   More sophisticated learning rate and sigma decay schedules.
-   Additional evaluation metrics (e.g., topographic error).
-   CPU backend (PyTorch-based) as a fallback.
-   Packaging for PyPI via `setup.py` or `pyproject.toml`.
-   More comprehensive examples and tutorials.

## Contributing
(Details to be added if the project becomes open source)

## License
(To be determined - likely MIT or Apache 2.0 if open-sourced)
```
