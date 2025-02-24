import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

# Define the PredictionCNN model
class PredictionCNN(nn.Module):
    def __init__(self):
        super(PredictionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # (1, 8, 8) -> (16, 8, 8)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # (16, 8, 8) -> (32, 8, 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 256)  # 32 channels * 8 * 8 grid
        self.fc2 = nn.Linear(256, 2049)  # Output size is 2049
        
    def forward(self, x):
        # Apply convolutions with ReLU activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Flatten the output of convolution layers
        x = x.view(x.size(0), -1)  # Flatten (N, 32, 8, 8) -> (N, 32*8*8)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Function to perform inference
def perform_inference(model, input_data, device):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        input_tensor = input_tensor.to(device)
        
        # Perform the forward pass
        output = model(input_tensor)
        return output.cpu().numpy()  # Convert output back to CPU numpy array

def load_model(model_path, device):
    model = PredictionCNN().to(device)
    # Load the model, mapping the storage to CPU if CUDA is not available
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Inference for the trained CNN model")
    parser.add_argument("--input-data", type=str, required=True, help="Path to the input data (.npy file)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model (.pth file)")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the output prediction (.npy file)")
    args = parser.parse_args()
    
    # Load input data
    input_data = np.load(args.input_data)  # Shape (N, 8, 8)
    
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    model = load_model(args.model_path, device)

    # Perform inference
    predictions = perform_inference(model, input_data, device)

    # Save predictions to the output path
    np.save(args.output_path, predictions)
    print(f"Inference complete. Predictions saved to {args.output_path}")

if __name__ == "__main__":
    main()
