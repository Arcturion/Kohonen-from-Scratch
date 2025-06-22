import torch
from .triton_kernels import find_bmu_kernel, update_weights_kernel

class SOM:
    """
    A Self-Organizing Map (SOM) implementation accelerated with Triton kernels.

    This class provides functionality to create, train, and utilize a SOM
    for dimensionality reduction and clustering. The performance-critical
    operations (BMU search and weight updates) are executed on NVIDIA GPUs
    using custom Triton kernels.
    """
    def __init__(self,
                 map_size: tuple[int, int],
                 input_dim: int,
                 device: str | torch.device = 'cuda',
                 random_seed: int | None = None
                ):
        """
        Initializes the Self-Organizing Map.

        Args:
            map_size (tuple[int, int]): Dimensions (rows, cols) of the SOM grid.
            input_dim (int): Dimensionality of the input data.
            device (str | torch.device): Device for computation ('cuda', 'cpu').
                                        Triton backend requires 'cuda'.
            random_seed (int | None): Seed for random weight initialization.
        """
        self.map_rows, self.map_cols = map_size
        self.num_neurons = self.map_rows * self.map_cols
        self.input_dim = input_dim
        self.device = torch.device(device)

        if self.device.type != 'cuda':
            raise ValueError("Triton-accelerated SOM currently requires a CUDA-enabled device.")

        if random_seed is not None:
            torch.manual_seed(random_seed)
            if self.device.type == 'cuda': # Redundant check given above, but good for clarity
                torch.cuda.manual_seed_all(random_seed)

        # Initialize weights
        self.weights = torch.randn(self.num_neurons, self.input_dim, device=self.device, dtype=torch.float32)
        self.weights = self.weights / torch.linalg.norm(self.weights, dim=1, keepdim=True)

        # Pre-calculate neuron locations
        self.neuron_locations = self._calculate_neuron_locations() # Already created on self.device

    def _calculate_neuron_locations(self) -> torch.Tensor:
        """Helper to compute neuron grid locations. Tensors are created on self.device."""
        # Using torch.meshgrid as in the original implementations
        neuron_locs_x = torch.arange(self.map_rows, device=self.device, dtype=torch.float32)
        neuron_locs_y = torch.arange(self.map_cols, device=self.device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(neuron_locs_x, neuron_locs_y, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)


    def train(self,
              data: torch.Tensor,
              num_epochs: int,
              initial_learning_rate: float,
              initial_sigma: float,
              # TODO: Consider adding learning rate and sigma decay schedules
             ):
        """
        Trains the SOM for a specified number of epochs.

        Args:
            data (torch.Tensor): Input data of shape (num_samples, input_dim).
                                 Data will be moved to the SOM's device.
            num_epochs (int): Total number of training epochs.
            initial_learning_rate (float): Starting learning rate.
            initial_sigma (float): Starting neighborhood radius (sigma).
        """
        if data.dim() != 2 or data.shape[1] != self.input_dim:
            raise ValueError(f"Input data must be 2D with {self.input_dim} features. Got shape {data.shape}")

        data_on_device = data.to(self.device)
        if data_on_device.dtype != torch.float32:
            data_on_device = data_on_device.float()


        current_learning_rate = float(initial_learning_rate)
        current_sigma = float(initial_sigma)

        for epoch in range(num_epochs):
            # Basic linear decay for learning rate and sigma as a starting point
            # More sophisticated decay functions can be added later
            lr_decay_factor = 1.0 - (epoch / num_epochs)
            effective_learning_rate = current_learning_rate * lr_decay_factor
            effective_sigma = current_sigma * lr_decay_factor
            
            # Ensure sigma doesn't become too small or zero, which can cause issues
            effective_sigma = max(0.1, effective_sigma)


            self._train_epoch(data_on_device, effective_learning_rate, effective_sigma)
            # print(f"Epoch {epoch+1}/{num_epochs}, LR: {effective_learning_rate:.4f}, Sigma: {effective_sigma:.4f}") # For debugging

    # Make _train_epoch public as train_epoch for benchmark compatibility and general single-epoch training
    def train_epoch(self,
                     data_batch: torch.Tensor, # Should be on the correct device and float32
                     learning_rate: float,
                     sigma: float):
        """
        Performs a single training epoch using the Triton backend.
        This method is also used by the benchmark script.
        """
        # Ensure data is on the correct device and float32, as this might be called directly
        data_on_device = data_batch.to(self.device)
        if data_on_device.dtype != torch.float32:
            data_on_device = data_on_device.float()

        if data_on_device.dim() != 2 or data_on_device.shape[1] != self.input_dim:
            raise ValueError(f"Input data for train_epoch must be 2D with {self.input_dim} features. Got shape {data_on_device.shape}")

        self._train_epoch(data_on_device, learning_rate, sigma)


    def _train_epoch(self,
                     data_batch: torch.Tensor, # Assumed to be on correct device and float32 by this internal method
                     learning_rate: float,
                     sigma: float):
        """
        Performs a single training epoch using the Triton backend.
        """
        num_samples = data_batch.shape[0]

        # 1. Find BMUs using Triton kernel
        bmu_indices = torch.empty(num_samples, dtype=torch.long, device=self.device)
        
        # Grid for BMU kernel: one program per sample
        grid_bmu = (num_samples,)
        find_bmu_kernel[grid_bmu](
            data_batch, self.weights, bmu_indices,
            NUM_SAMPLES=num_samples,
            NUM_NEURONS=self.num_neurons,
            NUM_FEATURES=self.input_dim,
            # BLOCK_SIZE_F is autotuned
        )
        
        # 2. Update weights using Triton kernel
        # Grid for update kernel: one program per sample (each sample updates all neurons with atomics)
        grid_update = (num_samples,)
        update_weights_kernel[grid_update](
            data_batch, self.weights, bmu_indices, self.neuron_locations,
            learning_rate, sigma,
            NUM_SAMPLES=num_samples,
            NUM_NEURONS=self.num_neurons,
            NUM_FEATURES=self.input_dim,
            # BLOCK_SIZE_F is autotuned
        )

    def get_weights(self) -> torch.Tensor:
        """Returns a copy of the current SOM weights."""
        return self.weights.clone().detach()

    def get_neuron_locations(self) -> torch.Tensor:
        """Returns a copy of the 2D locations of neurons on the grid."""
        return self.neuron_locations.clone().detach()

    def map_to_bmu_indices(self, data: torch.Tensor) -> torch.Tensor:
        """
        Maps input data points to their Best Matching Unit (BMU) indices.

        Args:
            data (torch.Tensor): Input data of shape (num_samples, input_dim).
                                 Data will be moved to the SOM's device.

        Returns:
            torch.Tensor: A 1D tensor containing the BMU index for each input data point.
        """
        if data.dim() != 2 or data.shape[1] != self.input_dim:
            raise ValueError(f"Input data must be 2D with {self.input_dim} features. Got shape {data.shape}")

        data_on_device = data.to(self.device)
        if data_on_device.dtype != torch.float32:
            data_on_device = data_on_device.float()

        num_samples = data_on_device.shape[0]
        bmu_indices = torch.empty(num_samples, dtype=torch.long, device=self.device)
        
        grid_bmu = (num_samples,)
        find_bmu_kernel[grid_bmu](
            data_on_device, self.weights, bmu_indices,
            NUM_SAMPLES=num_samples,
            NUM_NEURONS=self.num_neurons,
            NUM_FEATURES=self.input_dim,
        )
        return bmu_indices.detach()

    def map_to_bmu_locations(self, data: torch.Tensor) -> torch.Tensor:
        """
        Maps input data points to the 2D grid locations of their Best Matching Units (BMUs).

        Args:
            data (torch.Tensor): Input data of shape (num_samples, input_dim).

        Returns:
            torch.Tensor: A 2D tensor of shape (num_samples, 2) containing the
                          (row, col) grid coordinates for each input data point's BMU.
        """
        bmu_indices = self.map_to_bmu_indices(data)
        return self.neuron_locations[bmu_indices].detach()

    # TODO: Add save/load methods
    # def save(self, filepath: str):
    #     torch.save({
    #         'map_size': (self.map_rows, self.map_cols),
    #         'input_dim': self.input_dim,
    #         'weights': self.weights,
    #         'neuron_locations': self.neuron_locations,
    #         # Potentially other relevant attributes
    #     }, filepath)

    # @classmethod
    # def load(cls, filepath: str, device: str | torch.device = 'cuda'):
    #     checkpoint = torch.load(filepath, map_location=device)
    #     som = cls(map_size=checkpoint['map_size'],
    #               input_dim=checkpoint['input_dim'],
    #               device=device)
    #     som.weights = checkpoint['weights'].to(device)
    #     som.neuron_locations = checkpoint['neuron_locations'].to(device)
    #     return som

    def quantization_error(self, data: torch.Tensor) -> float:
        """
        Calculates the quantization error for the given data.
        Quantization error is the average distance between each data vector and its BMU.

        Args:
            data (torch.Tensor): Input data of shape (num_samples, input_dim).

        Returns:
            float: The quantization error.
        """
        if data.dim() != 2 or data.shape[1] != self.input_dim:
            raise ValueError(f"Input data must be 2D with {self.input_dim} features. Got shape {data.shape}")

        data_on_device = data.to(self.device).float()
        bmu_indices = self.map_to_bmu_indices(data_on_device)
        bmu_weights = self.weights[bmu_indices]
        
        error = torch.linalg.norm(data_on_device - bmu_weights, dim=1).mean().item()
        return error

    # Topographic error might be more involved if we need to find second BMU as well.
    # For now, keeping it simple.
    
    # def __str__(self) -> str:
    #     return f"SOM(map_size=({self.map_rows}, {self.map_cols}), input_dim={self.input_dim}, device='{self.device.type}')"

    # def __repr__(self) -> str:
    #     return self.__str__()

# Example of how to use (for testing purposes, will be removed or moved to examples)
if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is available. Running example.")
        
        # Configuration
        N_SAMPLES = 1000
        N_FEATURES = 64
        MAP_ROWS, MAP_COLS = 5, 5
        DEVICE = 'cuda'
        
        # Dummy data
        dummy_data_tensor = torch.rand(N_SAMPLES, N_FEATURES, dtype=torch.float32)
        
        # Instantiate SOM
        som_instance = SOM(map_size=(MAP_ROWS, MAP_COLS), input_dim=N_FEATURES, device=DEVICE, random_seed=42)
        
        print(f"SOM initialized with map size: ({som_instance.map_rows}, {som_instance.map_cols}), "
              f"input dim: {som_instance.input_dim}, device: {som_instance.device}")
        print(f"Weights shape: {som_instance.weights.shape}")
        print(f"Neuron locations shape: {som_instance.neuron_locations.shape}")

        # Test training
        print("\nStarting training...")
        som_instance.train(dummy_data_tensor, num_epochs=10, initial_learning_rate=0.5, initial_sigma=float(max(MAP_ROWS, MAP_COLS) / 2))
        print("Training finished.")

        # Test mapping
        print("\nMapping data to BMU indices...")
        bmu_indices_output = som_instance.map_to_bmu_indices(dummy_data_tensor[:10])
        print(f"BMU indices for first 10 samples: {bmu_indices_output}")

        print("\nMapping data to BMU locations...")
        bmu_locations_output = som_instance.map_to_bmu_locations(dummy_data_tensor[:10])
        print(f"BMU locations for first 10 samples:\n{bmu_locations_output}")

        # Test quantization error
        q_error = som_instance.quantization_error(dummy_data_tensor)
        print(f"\nQuantization Error: {q_error:.4f}")

        print("\nTesting get_weights():")
        weights_copy = som_instance.get_weights()
        print(f"Shape of weights_copy: {weights_copy.shape}")
        # Ensure it's a copy
        weights_copy[0,0] = 999.0 
        assert som_instance.weights[0,0] != 999.0, "get_weights() did not return a copy!"
        print("get_weights() returns a copy as expected.")

    else:
        print("CUDA not available. Skipping SOM example.")

# Ensure this file can be imported as part of a package
# For example, from the parent directory:
# import kohonen_triton
# model = kohonen_triton.SOM(...)
# Or:
# from kohonen_triton import SOM
# model = SOM(...)
