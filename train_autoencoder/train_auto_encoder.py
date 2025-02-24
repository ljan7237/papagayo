import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 64),  # Flatten and compress to a 64D latent space
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 64*8*8),  # From latent space back to 8x8 feature map
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Kalman Filter Class
class KalmanFilter:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-2, initial_estimate=0):
        self.process_noise = process_noise  # Process noise
        self.measurement_noise = measurement_noise  # Measurement noise
        self.estimate = initial_estimate  # Initial state estimate
        self.error_covariance = 1.0  # Initial error covariance

    def predict(self):
        # Predict next state (no actual control model, so using previous estimate)
        self.estimate = self.estimate
        self.error_covariance += self.process_noise  # Increase error covariance

    def update(self, measurement):
        # Kalman Gain
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_noise)
        
        # Update estimate
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        
        # Update error covariance
        self.error_covariance = (1 - kalman_gain) * self.error_covariance
        return self.estimate

# Function to save the trained model and Kalman filter state
def save_model(autoencoder, kalman_filter, path="data/models/butcherbird/autoencoder/butcherbird_autoencoder_kalman.pth"):
    checkpoint = {
        "autoencoder_state_dict": autoencoder.state_dict(),
        "kalman_filter_state": {
            "estimate": kalman_filter.estimate,
            "error_covariance": kalman_filter.error_covariance,
            "process_noise": kalman_filter.process_noise,
            "measurement_noise": kalman_filter.measurement_noise,
        },
    }
    torch.save(checkpoint, path)
    print(f"Model and Kalman filter saved to {path}")

# Training the model
def train_autoencoder_with_kalman_filter(autoencoder, data, num_epochs=10, learning_rate=0.001, save_path="autoencoder_kalman.pth"):
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    kalman_filter = KalmanFilter()

    for epoch in range(num_epochs):
        for i, input_data in enumerate(data):
            input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension
            
            # Forward pass through the autoencoder
            output = autoencoder(input_data)
            
            # Compute the reconstruction loss
            loss = criterion(output, input_data)
            
            # Kalman filter update
            kalman_filter.predict()  # Predict the next state
            kalman_filter.update(output.detach().numpy())  # Update with the new output
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100==0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data)}], Loss: {loss.item():.4f}")

    # Save model after training
    save_model(autoencoder, kalman_filter, save_path)

# Example usage
autoencoder = Autoencoder()

# Load training data (N, 8, 8)
data = np.load("data/models/butcherbird/training_data/embeddings/butcherbird_embeddings.npy")

# Train and save model
train_autoencoder_with_kalman_filter(autoencoder, data)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64*8*8, 64),  # Flatten and compress to a 64D latent space
#         )
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(64, 64*8*8),  # From latent space back to 8x8 feature map
#             nn.ReLU(),
#             nn.Unflatten(1, (64, 8, 8)),
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# # Kalman Filter Class
# class KalmanFilter:
#     def __init__(self, process_noise=1e-5, measurement_noise=1e-2, initial_estimate=0):
#         self.process_noise = process_noise  # Process noise
#         self.measurement_noise = measurement_noise  # Measurement noise
#         self.estimate = initial_estimate  # Initial state estimate
#         self.error_covariance = 1.0  # Initial error covariance

#     def predict(self):
#         # Predict next state (no actual control model, so using previous estimate)
#         self.estimate = self.estimate
#         self.error_covariance += self.process_noise  # Increase error covariance

#     def update(self, measurement):
#         # Kalman Gain
#         kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_noise)
        
#         # Update estimate
#         self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        
#         # Update error covariance
#         self.error_covariance = (1 - kalman_gain) * self.error_covariance
#         return self.estimate

# # Training the model
# def train_autoencoder_with_kalman_filter(autoencoder, data, num_epochs=10, learning_rate=0.001):
#     criterion = nn.MSELoss()  # Mean Squared Error Loss
#     optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

#     kalman_filter = KalmanFilter()

#     for epoch in range(num_epochs):
#         for i, input_data in enumerate(data):
#             input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension
            
#             # Forward pass through the autoencoder
#             output = autoencoder(input_data)
            
#             # Compute the reconstruction loss
#             loss = criterion(output, input_data)
            
#             # Kalman filter update
#             kalman_filter.predict()  # Predict the next state
#             kalman_filter.update(output.detach().numpy())  # Update with the new output
            
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data)}], Loss: {loss.item():.4f}")

# # Example usage
# autoencoder = Autoencoder()

# # Dummy training data (N, 8, 8)
# data = np.load("data/models/kookaburra/training_data/embeddings/kookaburra_embeddings.npy")#np.random.rand(100, 8, 8)  # 100 samples of 8x8 inputs (replace with actual data)

# train_autoencoder_with_kalman_filter(autoencoder, data)