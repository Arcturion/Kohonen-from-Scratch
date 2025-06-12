import torch

class PyTorchSOM:
    """
    A fully vectorized, idiomatic PyTorch implementation of a Self-Organizing Map.
    """
    def __init__(self, map_size: tuple[int, int], input_dim: int, device: str | torch.device):
        """
        Initializes the SOM.

        Args:
            map_size (tuple[int, int]): The dimensions (rows, cols) of the SOM grid.
            input_dim (int): The dimensionality of the input vectors.
            device (str or torch.device): The device to run the computations on.
        """
        self.map_rows, self.map_cols = map_size
        self.num_neurons = self.map_rows * self.map_cols
        self.input_dim = input_dim
        self.device = device

        # Initialize weights randomly and normalize
        self.weights = torch.randn(self.num_neurons, self.input_dim, device=self.device)
        self.weights = self.weights / torch.linalg.norm(self.weights, dim=1, keepdim=True)

        # Pre-calculate neuron locations on the 2D grid for neighborhood function
        # This is a crucial optimization to avoid re-computing it in every iteration
        self.neuron_locations = torch.stack(
            torch.meshgrid(
                torch.arange(self.map_rows, device=self.device),
                torch.arange(self.map_cols, device=self.device),
                indexing='ij'
            )
        ).float().reshape(2, -1).T

    def train_epoch(self, data: torch.Tensor, learning_rate: float, sigma: float):
        """
        Performs one training epoch.

        Args:
            data (torch.Tensor): The input data batch of shape (num_samples, input_dim).
            learning_rate (float): The learning rate for the weight update.
            sigma (float): The radius of the neighborhood function (Gaussian).
        """
        # 1. Find the Best Matching Unit (BMU) for each input vector
        # `torch.cdist` is highly optimized for this pairwise distance calculation.
        dists = torch.cdist(data, self.weights)  # Shape: (num_samples, num_neurons)
        bmu_indices = torch.argmin(dists, dim=1) # Shape: (num_samples)

        # 2. Calculate the influence of the neighborhood function (Gaussian)
        # Get the 2D grid coordinates of the BMUs for each input sample
        bmu_locations = self.neuron_locations[bmu_indices] # Shape: (num_samples, 2)

        # Calculate the squared distance from each BMU to all other neurons on the grid
        # This uses broadcasting: (num_samples, 1, 2) - (1, num_neurons, 2)
        distance_to_bmus_sq = torch.sum(
            (bmu_locations.unsqueeze(1) - self.neuron_locations.unsqueeze(0)) ** 2,
            dim=2
        ) # Shape: (num_samples, num_neurons)

        # Apply the Gaussian neighborhood function
        influence = torch.exp(-distance_to_bmus_sq / (2 * (sigma ** 2))) # Shape: (num_samples, num_neurons)

        # 3. Update the weights
        # Calculate the required change for each weight vector (delta)
        # This uses broadcasting: (num_samples, 1, input_dim) - (1, num_neurons, input_dim)
        delta = data.unsqueeze(1) - self.weights.unsqueeze(0) # Shape: (num_samples, num_neurons, input_dim)

        # Weight the deltas by the learning rate and the neighborhood influence
        # Broadcasting: (1.0) * (num_samples, num_neurons, 1) * (num_samples, num_neurons, input_dim)
        update_term = learning_rate * influence.unsqueeze(2) * delta

        # Sum the updates for each neuron across the entire batch and apply them
        self.weights += torch.sum(update_term, dim=0)
