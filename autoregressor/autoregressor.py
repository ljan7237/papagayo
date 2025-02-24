import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import soundfile as sf
from pydub import AudioSegment

# Load pre-trained models (assuming you have these trained already)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class KalmanFilter:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-2, initial_estimate=0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate = initial_estimate
        self.error_covariance = 1.0

    def predict(self):
        self.estimate = self.estimate
        self.error_covariance += self.process_noise

    def update(self, measurement):
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_noise)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.error_covariance = (1 - kalman_gain) * self.error_covariance
        return self.estimate

    def load_params(self, process_noise, measurement_noise, initial_estimate):
        """Manually load Kalman filter parameters"""
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate = initial_estimate
        self.error_covariance = 1.0

class UpsamplingCNN(nn.Module):
    def __init__(self):
        super(UpsamplingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2049)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to generate autoregressive sequence
def generate_sequence(autoencoder, kalman_filter, upsampler, initial_input, num_steps):
    autoencoder.eval()
    upsampler.eval()
    
    generated_data = []
    current_input = initial_input.clone()

    for _ in range(num_steps):
        with torch.no_grad():
            # Predict next 8x8 using autoencoder
            predicted_8x8 = autoencoder(current_input)

            # Apply Kalman filter
            kalman_filter.predict()
            kalman_output = kalman_filter.update(predicted_8x8.numpy())

            # Convert back to tensor
            predicted_8x8_filtered = torch.tensor(kalman_output, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Assuming predicted_8x8_filtered is the output of the Kalman filter
            predicted_8x8_filtered = predicted_8x8_filtered.squeeze()  # Remove unnecessary dimensions
            if predicted_8x8_filtered.dim() == 2:  # If it's 2D, add batch and channel dimensions
                predicted_8x8_filtered = predicted_8x8_filtered.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 8, 8]
            else:
                # Add any necessary reshaping for other cases
                raise Exception("Error reshaping input.")
            # Upsample to 2049
            upsampled_output = upsampler(predicted_8x8_filtered).squeeze().numpy()
            
            generated_data.append(upsampled_output)

            # Feed the predicted 8x8 back for the next step
            current_input = predicted_8x8_filtered.clone()

    return np.array(generated_data)  # Shape (N, 2049)

def npy_to_audio(npy_file, output_wav, output_mp3, sr=22050, n_fft=4096, hop_length=256):
    # Load magnitude spectrogram
    magnitude = np.load(npy_file)  # Shape: (N, 2049)

    # Transpose to match librosa's expected shape (freq_bins, time_frames)
    magnitude = magnitude.T  # Shape: (2049, N)

    # Estimate phase using Griffin-Lim
    reconstructed_waveform = librosa.griffinlim(magnitude, n_fft=n_fft, hop_length=hop_length)

    # Save as WAV
    sf.write(output_wav, reconstructed_waveform, sr)

    # Convert to MP3
    audio = AudioSegment.from_wav(output_wav)
    audio.export(output_mp3, format="mp3")

    print(f"Saved: {output_wav} and {output_mp3}")

@click.command()
@click.option("--initial-input", type=str, required=True, help="Path to the initial 8x8 .npy file.")
@click.option("--num-steps", type=int, default=10, help="Number of autoregressive steps.")
@click.option("--output-file", type=str, default="generated_slices.npy", help="Output .npy file.")
@click.option("--to-wav", is_flag=True, help="Convert output .npy to .wav.")
@click.option("--to-mp3", is_flag=True, help="Convert output .wav to .mp3.")
def main(initial_input, num_steps, output_file, to_wav, to_mp3):
    # Load initial input
    initial_data = np.load(initial_input)
    initial_data = torch.tensor(initial_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Initialize models
    autoencoder = Autoencoder()  
    kalman_filter = KalmanFilter()  # Initialize the Kalman filter
    upsampler = UpsamplingCNN()  
    
    # Load the checkpoint
    checkpoint = torch.load("data/models/butcherbird/autoencoder/autoencoder_kalman.pth", map_location='cpu', weights_only=False)

    # Extract and load autoencoder state_dict
    autoencoder_state_dict = checkpoint['autoencoder_state_dict']
    autoencoder.load_state_dict(autoencoder_state_dict)

    # Extract and load Kalman filter parameters
    kalman_filter_params = checkpoint.get('kalman_filter_params')
    if kalman_filter_params:
        process_noise = kalman_filter_params.get('process_noise', 1e-5)
        measurement_noise = kalman_filter_params.get('measurement_noise', 1e-2)
        initial_estimate = kalman_filter_params.get('initial_estimate', 0)
        kalman_filter.load_params(process_noise, measurement_noise, initial_estimate)

    # Initialize upsampler model
    upsampler.load_state_dict(torch.load("data/models/butcherbird/upsampler/butcherbird_upsampler.pth", map_location='cpu'))
    upsampler.eval()

    # Generate sequence
    generated_slices = generate_sequence(autoencoder, kalman_filter, upsampler, initial_data, num_steps)

    # Save to .npy
    np.save(output_file, generated_slices)
    click.echo(f"Saved generated slices to: {output_file}")

    if to_wav:
        wav_file = output_file.replace(".npy", ".wav")
        output_mp3 = output_file.replace(".npy", ".mp3")
        npy_to_audio(output_file, wav_file, output_mp3, sr=22050, n_fft=4096, hop_length=256)


if __name__ == "__main__":
    main()

# import click
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import scipy.io.wavfile as wav
# from pydub import AudioSegment

# # Load pre-trained models (assuming you have these trained already)
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * 8 * 8, 64),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(64, 64 * 8 * 8),
#             nn.ReLU(),
#             nn.Unflatten(1, (64, 8, 8)),
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
    

# class KalmanFilter:
#     def __init__(self, process_noise=1e-5, measurement_noise=1e-2, initial_estimate=0):
#         self.process_noise = process_noise
#         self.measurement_noise = measurement_noise
#         self.estimate = initial_estimate
#         self.error_covariance = 1.0

#     def predict(self):
#         self.estimate = self.estimate
#         self.error_covariance += self.process_noise

#     def update(self, measurement):
#         kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_noise)
#         self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
#         self.error_covariance = (1 - kalman_gain) * self.error_covariance
#         return self.estimate

#     def load_params(self, process_noise, measurement_noise, initial_estimate):
#         """Manually load Kalman filter parameters"""
#         self.process_noise = process_noise
#         self.measurement_noise = measurement_noise
#         self.estimate = initial_estimate
#         self.error_covariance = 1.0
# # class KalmanFilter:
# #     def __init__(self, process_noise=1e-5, measurement_noise=1e-2, initial_estimate=0):
# #         self.process_noise = process_noise
# #         self.measurement_noise = measurement_noise
# #         self.estimate = initial_estimate
# #         self.error_covariance = 1.0

# #     def predict(self):
# #         self.estimate = self.estimate
# #         self.error_covariance += self.process_noise

# #     def update(self, measurement):
# #         kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_noise)
# #         self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
# #         self.error_covariance = (1 - kalman_gain) * self.error_covariance
# #         return self.estimate

# class UpsamplingCNN(nn.Module):
#     def __init__(self):
#         super(UpsamplingCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(32 * 8 * 8, 256)
#         self.fc2 = nn.Linear(256, 2049)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Function to generate autoregressive sequence
# def generate_sequence(autoencoder, kalman_filter, upsampler, initial_input, num_steps):
#     autoencoder.eval()
#     upsampler.eval()
    
#     generated_data = []
#     current_input = initial_input.clone()

#     for _ in range(num_steps):
#         with torch.no_grad():
#             # Predict next 8x8 using autoencoder
#             predicted_8x8 = autoencoder(current_input)

#             # Apply Kalman filter
#             kalman_filter.predict()
#             kalman_output = kalman_filter.update(predicted_8x8.numpy())

#             # Convert back to tensor
#             predicted_8x8_filtered = torch.tensor(kalman_output, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
#             # Assuming predicted_8x8_filtered is the output of the Kalman filter
#             predicted_8x8_filtered = predicted_8x8_filtered.squeeze()  # Remove unnecessary dimensions
#             if predicted_8x8_filtered.dim() == 2:  # If it's 2D, add batch and channel dimensions
#                 predicted_8x8_filtered = predicted_8x8_filtered.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 8, 8]
#             else:
#                 # Add any necessary reshaping for other cases
#                 raise Exception("Error reshaping input.")
#             # Upsample to 2049
#             upsampled_output = upsampler(predicted_8x8_filtered).squeeze().numpy()
            
#             generated_data.append(upsampled_output)

#             # Feed the predicted 8x8 back for the next step
#             current_input = predicted_8x8_filtered.clone()

#     return np.array(generated_data)  # Shape (N, 2049)

# # Convert .npy file to WAV
# def npy_to_wav(npy_file, wav_file, sample_rate=22050):
#     data = np.load(npy_file)

#     # Normalize
#     data = data / np.max(np.abs(data))

#     # Convert to 16-bit PCM
#     waveform = np.int16(data * 32767)
#     wav.write(wav_file, sample_rate, waveform)

#     click.echo(f"Saved WAV file: {wav_file}")

# # Convert WAV to MP3
# def wav_to_mp3(wav_file, mp3_file):
#     audio = AudioSegment.from_wav(wav_file)
#     audio.export(mp3_file, format="mp3")
#     click.echo(f"Saved MP3 file: {mp3_file}")


# @click.command()
# @click.option("--initial-input", type=str, required=True, help="Path to the initial 8x8 .npy file.")
# @click.option("--num-steps", type=int, default=10, help="Number of autoregressive steps.")
# @click.option("--output-file", type=str, default="generated_slices.npy", help="Output .npy file.")
# @click.option("--to-wav", is_flag=True, help="Convert output .npy to .wav.")
# @click.option("--to-mp3", is_flag=True, help="Convert output .wav to .mp3.")
# def main(initial_input, num_steps, output_file, to_wav, to_mp3):
#     # Load initial input
#     initial_data = np.load(initial_input)
#     initial_data = torch.tensor(initial_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

#     # Initialize models
#     autoencoder = Autoencoder()  
#     kalman_filter = KalmanFilter()  # Initialize the Kalman filter
#     upsampler = UpsamplingCNN()  
    
#     # Load the checkpoint
#     checkpoint = torch.load("data/models/butcherbird/autoencoder/autoencoder_kalman.pth", map_location='cpu', weights_only=False)

#     # Extract and load autoencoder state_dict
#     autoencoder_state_dict = checkpoint['autoencoder_state_dict']
#     autoencoder.load_state_dict(autoencoder_state_dict)

#     # Extract and load Kalman filter parameters
#     kalman_filter_params = checkpoint.get('kalman_filter_params')
#     if kalman_filter_params:
#         process_noise = kalman_filter_params.get('process_noise', 1e-5)
#         measurement_noise = kalman_filter_params.get('measurement_noise', 1e-2)
#         initial_estimate = kalman_filter_params.get('initial_estimate', 0)
#         kalman_filter.load_params(process_noise, measurement_noise, initial_estimate)

#     # Initialize upsampler model
#     upsampler.load_state_dict(torch.load("data/models/butcherbird/upsampler/butcherbird_upsampler.pth", map_location='cpu'))
#     upsampler.eval()

#     # Generate sequence
#     generated_slices = generate_sequence(autoencoder, kalman_filter, upsampler, initial_data, num_steps)

#     # Save to .npy
#     np.save(output_file, generated_slices)
#     click.echo(f"Saved generated slices to: {output_file}")

#     if to_wav:
#         wav_file = output_file.replace(".npy", ".wav")
#         npy_to_wav(output_file, wav_file)

#         if to_mp3:
#             mp3_file = output_file.replace(".npy", ".mp3")
#             wav_to_mp3(wav_file, mp3_file)

# if __name__ == "__main__":
#     main()
