import torch
from kohonen import SOM

def run_som_example():
    """
    Demonstrates the basic usage of the kohonen_triton.SOM library.
    """
    print("--- Running SOM Library Example ---")

    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a CUDA-enabled GPU to run the Triton-accelerated SOM.")
        return

    # 1. Configuration
    device = 'cuda'
    num_samples = 2000
    input_features = 128
    som_map_size = (10, 10) # 10x10 grid, 100 neurons
    random_seed = 42 # For reproducibility

    print(f"Configuration: Device={device}, Samples={num_samples}, Features={input_features}, Map Size={som_map_size}")

    # 2. Generate Dummy Data
    # Data should be somewhat clustered to see meaningful mapping
    # Creating two clusters for demonstration
    data1 = torch.randn(num_samples // 2, input_features, device=device) + 2.0
    data2 = torch.randn(num_samples // 2, input_features, device=device) - 2.0
    data = torch.cat([data1, data2], dim=0)
    data = data[torch.randperm(data.shape[0])] # Shuffle the data

    print(f"Generated data of shape: {data.shape}")

    # 3. Initialize the SOM
    print("\nInitializing SOM...")
    som_model = SOM(
        map_size=som_map_size,
        input_dim=input_features,
        device=device,
        random_seed=random_seed
    )
    print(f"SOM initialized. Number of neurons: {som_model.num_neurons}")
    initial_weights = som_model.get_weights().clone() # Save for comparison later

    # 4. Train the SOM
    num_epochs = 50
    learning_rate = 0.6
    sigma = float(max(som_map_size) / 2.0) # Initial neighborhood radius

    print(f"\nTraining SOM for {num_epochs} epochs...")
    print(f"Initial LR: {learning_rate}, Initial Sigma: {sigma}")
    
    # You can track training progress or errors if needed
    # For this example, we'll just run the training.
    # The `train` method has a simple linear decay for LR and Sigma.
    som_model.train(
        data,
        num_epochs=num_epochs,
        initial_learning_rate=learning_rate,
        initial_sigma=sigma
    )
    print("Training complete.")

    trained_weights = som_model.get_weights()
    # Check if weights have changed
    assert not torch.equal(initial_weights, trained_weights), "Weights did not change after training!"
    print("Weights have been updated during training.")

    # 5. Map Data to the SOM
    print("\nMapping data to BMU locations on the SOM grid...")
    # Get BMU grid locations for the first 10 samples
    sample_data_to_map = data[:10]
    bmu_locations = som_model.map_to_bmu_locations(sample_data_to_map)
    
    print("BMU (row, col) locations for the first 10 samples:")
    for i in range(sample_data_to_map.shape[0]):
        print(f"  Sample {i}: Data starts with [{sample_data_to_map[i,0]:.2f}, {sample_data_to_map[i,1]:.2f}, ...] -> BMU @ ({int(bmu_locations[i,0].item())}, {int(bmu_locations[i,1].item())})")

    # 6. Calculate Quantization Error
    q_error = som_model.quantization_error(data)
    print(f"\nQuantization Error on the full dataset: {q_error:.4f}")

    # 7. Get Neuron Information
    neuron_grid_locations = som_model.get_neuron_locations()
    print(f"\nTotal neurons on grid: {neuron_grid_locations.shape[0]}")
    print(f"Example neuron location (neuron 0): ({neuron_grid_locations[0,0].item()}, {neuron_grid_locations[0,1].item()})")


    print("\n--- SOM Library Example Finished ---")

if __name__ == "__main__":
    run_som_example()
