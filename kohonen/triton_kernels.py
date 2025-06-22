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
    pid = tl.program_id(axis=0)
    min_dist_sq = float('inf')
    bmu_index = -1

    for n_idx in range(NUM_NEURONS):
        dist_sq = 0.0
        for f_offset in range(0, NUM_FEATURES, BLOCK_SIZE_F):
            f_offsets = f_offset + tl.arange(0, BLOCK_SIZE_F)
            f_mask = f_offsets < NUM_FEATURES
            input_chunk = tl.load(input_ptr + pid * NUM_FEATURES + f_offsets, mask=f_mask, other=0.0)
            weight_chunk = tl.load(weights_ptr + n_idx * NUM_FEATURES + f_offsets, mask=f_mask, other=0.0)
            diff = input_chunk - weight_chunk
            dist_sq += tl.sum(diff * diff)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            bmu_index = n_idx
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
    pid = tl.program_id(axis=0)
    bmu_idx = tl.load(bmu_indices_ptr + pid)
    bmu_loc_x = tl.load(neuron_locs_ptr + bmu_idx * 2)
    bmu_loc_y = tl.load(neuron_locs_ptr + bmu_idx * 2 + 1)

    for n_idx in range(NUM_NEURONS):
        neuron_loc_x = tl.load(neuron_locs_ptr + n_idx * 2)
        neuron_loc_y = tl.load(neuron_locs_ptr + n_idx * 2 + 1)
        grid_dist_sq = (bmu_loc_x - neuron_loc_x)**2 + (bmu_loc_y - neuron_loc_y)**2
        influence = tl.exp(-grid_dist_sq / (2.0 * sigma * sigma))
        update_factor = learning_rate * influence

        for f_offset in range(0, NUM_FEATURES, BLOCK_SIZE_F):
            f_offsets = f_offset + tl.arange(0, BLOCK_SIZE_F)
            f_mask = f_offsets < NUM_FEATURES
            weight_chunk_ptr = weights_ptr + n_idx * NUM_FEATURES + f_offsets
            input_chunk = tl.load(input_ptr + pid * NUM_FEATURES + f_offsets, mask=f_mask, other=0.0)
            weight_chunk = tl.load(weight_chunk_ptr, mask=f_mask, other=0.0)
            delta = input_chunk - weight_chunk
            update_val = update_factor * delta
            tl.atomic_add(weight_chunk_ptr, update_val, mask=f_mask)
