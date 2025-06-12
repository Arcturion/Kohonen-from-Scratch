import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_F': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_F': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_F': 256}, num_warps=8),
    ],
    key=['NUM_FEATURES', 'NUM_NEURONS'],
)
@triton.jit
def find_bmu_kernel(
    input_ptr,       # Pointer to input data tensor
    weights_ptr,     # Pointer to SOM weights tensor
    bmu_indices_ptr, # Pointer to output BMU indices
    NUM_SAMPLES: tl.constexpr,
    NUM_NEURONS: tl.constexpr,
    NUM_FEATURES: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr,
):
    """
    Triton kernel to find the Best Matching Unit (BMU) for each input sample.
    Grid is 1D, with one program instance per input sample.
    """
    # Get the index for the current input sample this program is responsible for
    pid = tl.program_id(axis=0)

    # Initialize minimum distance to infinity and BMU index to -1
    min_dist_sq = float('inf')
    bmu_index = -1

    # This program computes the BMU for a single input vector.
    # We iterate through all neurons to find the one with the minimum distance.
    for n_idx in range(NUM_NEURONS):
        dist_sq = 0.0
        # Iterate over the feature dimension in blocks for efficient memory access
        for f_offset in range(0, NUM_FEATURES, BLOCK_SIZE_F):
            # Create masks to handle feature dimensions that are not a multiple of BLOCK_SIZE_F
            f_offsets = f_offset + tl.arange(0, BLOCK_SIZE_F)
            f_mask = f_offsets < NUM_FEATURES

            # Load a block of the input vector and a block of the current neuron's weights
            # This access pattern is coalesced as all threads in a warp access contiguous memory
            input_chunk = tl.load(input_ptr + pid * NUM_FEATURES + f_offsets, mask=f_mask, other=0.0)
            weight_chunk = tl.load(weights_ptr + n_idx * NUM_FEATURES + f_offsets, mask=f_mask, other=0.0)

            # Calculate squared Euclidean distance for the chunk and accumulate
            diff = input_chunk - weight_chunk
            dist_sq += tl.sum(diff * diff)

        # Update the minimum distance and BMU index if a closer neuron is found
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            bmu_index = n_idx

    # Write the final BMU index to the output tensor
    tl.store(bmu_indices_ptr + pid, bmu_index)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_F': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_F': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_F': 256}, num_warps=8),
    ],
    key=['NUM_FEATURES', 'NUM_NEURONS'],
)
@triton.jit
def update_weights_kernel(
    input_ptr,         # Pointer to input data tensor
    weights_ptr,       # Pointer to SOM weights tensor to be updated
    bmu_indices_ptr,   # Pointer to BMU indices from the first kernel
    neuron_locs_ptr,   # Pointer to the 2D grid locations of neurons
    learning_rate,
    sigma,
    NUM_SAMPLES: tl.constexpr,
    NUM_NEURONS: tl.constexpr,
    NUM_FEATURES: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr,
):
    """
    Triton kernel to update SOM weights.
    Grid is 1D, with one program instance per input sample.
    Each program updates ALL neurons based on its assigned input sample.
    This requires atomic adds to prevent race conditions when multiple programs
    update the same weight vector.
    """
    # Get the index for the current input sample
    pid = tl.program_id(axis=0)

    # Load the BMU index for this input sample
    bmu_idx = tl.load(bmu_indices_ptr + pid)

    # Load the 2D coordinates of this sample's BMU
    bmu_loc_x = tl.load(neuron_locs_ptr + bmu_idx * 2)
    bmu_loc_y = tl.load(neuron_locs_ptr + bmu_idx * 2 + 1)

    # This program iterates over all neurons and updates them based on their
    # distance to the BMU of the current input sample (pid).
    for n_idx in range(NUM_NEURONS):
        # Load the 2D coordinates of the current neuron (n_idx)
        neuron_loc_x = tl.load(neuron_locs_ptr + n_idx * 2)
        neuron_loc_y = tl.load(neuron_locs_ptr + n_idx * 2 + 1)

        # Calculate squared Euclidean distance on the 2D grid
        grid_dist_sq = (bmu_loc_x - neuron_loc_x)**2 + (bmu_loc_y - neuron_loc_y)**2

        # Calculate Gaussian influence
        # Note: tl.exp is available in Triton
        influence = tl.exp(-grid_dist_sq / (2.0 * sigma * sigma))
        
        # Calculate the final update factor
        update_factor = learning_rate * influence

        # Iterate over the feature dimension in blocks to apply the update
        for f_offset in range(0, NUM_FEATURES, BLOCK_SIZE_F):
            f_offsets = f_offset + tl.arange(0, BLOCK_SIZE_F)
            f_mask = f_offsets < NUM_FEATURES

            # Calculate the pointer to the current chunk of weights for neuron n_idx
            weight_chunk_ptr = weights_ptr + n_idx * NUM_FEATURES + f_offsets

            # Load the chunks of data
            input_chunk = tl.load(input_ptr + pid * NUM_FEATURES + f_offsets, mask=f_mask, other=0.0)
            weight_chunk = tl.load(weight_chunk_ptr, mask=f_mask, other=0.0)

            # Calculate the delta and the final update value
            delta = input_chunk - weight_chunk
            update_val = update_factor * delta

            # Atomically add the update to the global weights tensor.
            # This is critical to avoid race conditions as multiple input samples
            # (and thus multiple program instances) will try to update the same
            # neuron weights simultaneously.
            tl.atomic_add(weight_chunk_ptr, update_val, mask=f_mask)


class TritonSOM:
    """
    Triton-based implementation of a Self-Organizing Map.
    """
    def __init__(self, map_size: tuple[int, int], input_dim: int, device: str | torch.device):
        self.map_rows, self.map_cols = map_size
        self.num_neurons = self.map_rows * self.map_cols
        self.input_dim = input_dim
        self.device = device

        self.weights = torch.randn(self.num_neurons, self.input_dim, device=self.device)
        self.weights = self.weights / torch.linalg.norm(self.weights, dim=1, keepdim=True)
        
        self.neuron_locations = torch.stack(
            torch.meshgrid(
                torch.arange(self.map_rows, device=self.device),
                torch.arange(self.map_cols, device=self.device),
                indexing='ij'
            )
        ).float().reshape(2, -1).T

    def train_epoch(self, data: torch.Tensor, learning_rate: float, sigma: float):
        num_samples = data.shape[0]
        
        # 1. Find BMUs using the first Triton kernel
        bmu_indices = torch.empty(num_samples, dtype=torch.long, device=self.device)
        grid = (num_samples,) # One program instance per sample
        find_bmu_kernel[grid](
            data, self.weights, bmu_indices,
            NUM_SAMPLES=num_samples,
            NUM_NEURONS=self.num_neurons,
            NUM_FEATURES=self.input_dim,
        )

        # 2. Update weights using the second Triton kernel
        grid = (num_samples,) # One program instance per sample
        update_weights_kernel[grid](
            data, self.weights, bmu_indices, self.neuron_locations,
            learning_rate, sigma,
            NUM_SAMPLES=num_samples,
            NUM_NEURONS=self.num_neurons,
            NUM_FEATURES=self.input_dim
        )
