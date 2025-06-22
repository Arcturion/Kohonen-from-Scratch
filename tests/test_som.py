import unittest
import torch

# Attempt to import SOM, but handle import error if Triton issues prevent it
try:
    from kohonen_triton import SOM
    TRITON_AVAILABLE = True
except RuntimeError as e:
    print(f"Warning: Could not import SOM due to Triton runtime error: {e}")
    print("Skipping tests that require SOM instantiation with Triton.")
    TRITON_AVAILABLE = False
except ImportError as e:
    print(f"Warning: Could not import SOM due to ImportError: {e}")
    print("Skipping tests that require SOM instantiation.")
    TRITON_AVAILABLE = False


class TestSOMInitialization(unittest.TestCase):

    def test_dummy_true(self):
        """ A dummy test to ensure the test runner is working. """
        self.assertTrue(True)

    @unittest.skipIf(not TRITON_AVAILABLE or not torch.cuda.is_available(), "SOM/Triton or CUDA not available")
    def test_som_initialization_values(self):
        map_size = (5, 5)
        input_dim = 10
        som = SOM(map_size=map_size, input_dim=input_dim, device='cuda')
        self.assertEqual(som.map_rows, map_size[0])
        self.assertEqual(som.map_cols, map_size[1])
        self.assertEqual(som.input_dim, input_dim)
        self.assertEqual(som.num_neurons, map_size[0] * map_size[1])
        self.assertEqual(som.weights.shape, (map_size[0] * map_size[1], input_dim))
        self.assertEqual(som.neuron_locations.shape, (map_size[0] * map_size[1], 2))
        self.assertTrue(som.weights.is_cuda)
        self.assertTrue(som.neuron_locations.is_cuda)

    @unittest.skipIf(not TRITON_AVAILABLE or not torch.cuda.is_available(), "SOM/Triton or CUDA not available")
    def test_get_weights_returns_copy(self):
        som = SOM(map_size=(2,2), input_dim=3, device='cuda')
        weights_original_ptr = som.weights.data_ptr()
        weights_copy = som.get_weights()
        self.assertNotEqual(weights_copy.data_ptr(), weights_original_ptr)
        weights_copy[0,0] = 999.0
        self.assertNotEqual(som.weights[0,0].item(), 999.0, "Modifying copy changed original weights.")

    @unittest.skipIf(not TRITON_AVAILABLE or not torch.cuda.is_available(), "SOM/Triton or CUDA not available")
    def test_get_neuron_locations_returns_copy(self):
        som = SOM(map_size=(2,2), input_dim=3, device='cuda')
        locs_original_ptr = som.neuron_locations.data_ptr()
        locs_copy = som.get_neuron_locations()
        self.assertNotEqual(locs_copy.data_ptr(), locs_original_ptr)
        locs_copy[0,0] = 999.0
        self.assertNotEqual(som.neuron_locations[0,0].item(), 999.0, "Modifying copy changed original locations.")

    @unittest.skipIf(not TRITON_AVAILABLE or not torch.cuda.is_available(), "SOM/Triton or CUDA not available")
    def test_input_validation_train(self):
        som = SOM(map_size=(2,2), input_dim=3, device='cuda')
        # Correct data
        correct_data = torch.rand(10, 3, device='cuda')
        try:
            # Run for a single epoch to test data validation path in train -> _train_epoch
            som.train(correct_data, num_epochs=1, initial_learning_rate=0.1, initial_sigma=1.0)
        except ValueError:
            self.fail("train() raised ValueError unexpectedly for correct data shape.")

        # Incorrect feature dimension
        incorrect_data_feat = torch.rand(10, 4, device='cuda')
        with self.assertRaisesRegex(ValueError, "Input data must be 2D with 3 features"):
            som.train(incorrect_data_feat, num_epochs=1, initial_learning_rate=0.1, initial_sigma=1.0)
        
        # Incorrect data dimension (3D)
        incorrect_data_dim = torch.rand(10, 3, 1, device='cuda')
        with self.assertRaisesRegex(ValueError, "Input data must be 2D with 3 features"):
            som.train(incorrect_data_dim, num_epochs=1, initial_learning_rate=0.1, initial_sigma=1.0)

    @unittest.skipIf(not TRITON_AVAILABLE or not torch.cuda.is_available(), "SOM/Triton or CUDA not available")
    def test_input_validation_map_to_bmu_indices(self):
        som = SOM(map_size=(2,2), input_dim=3, device='cuda')
        # Correct data
        correct_data = torch.rand(10, 3, device='cuda')
        try:
            som.map_to_bmu_indices(correct_data)
        except ValueError:
            self.fail("map_to_bmu_indices() raised ValueError unexpectedly for correct data shape.")

        # Incorrect feature dimension
        incorrect_data_feat = torch.rand(10, 4, device='cuda')
        with self.assertRaisesRegex(ValueError, "Input data must be 2D with 3 features"):
            som.map_to_bmu_indices(incorrect_data_feat)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Some tests will be skipped.")
    
    # Need to ensure kohonen_triton is in PYTHONPATH if running directly
    # For example, by running as: python -m tests.test_som from the root directory
    unittest.main()
