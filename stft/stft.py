import os
import torch
import librosa
import numpy as np
import click
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Check for Apple M3 GPU support
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def process_audio(file_path, output_dir, n_fft, hop_length, save_phase):
    filename = Path(file_path).name

    # Load audio (Librosa runs on CPU)
    audio, sr = librosa.load(file_path, sr=None)

    # Convert to PyTorch Tensor & move to GPU
    audio_tensor = torch.from_numpy(audio).to(dtype=torch.float32, device=device)

    # Create the window function (Hann)
    window = torch.hann_window(n_fft, device=device)

    # Apply STFT on GPU
    stft_output = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

    # Convert to magnitude and phase spectrograms
    magnitude = stft_output.abs().cpu().numpy().T  # Magnitude spectrogram
    phase = np.angle(stft_output.cpu().numpy()).T if save_phase else None  # Phase spectrogram

    # Save magnitude and phase as .npy files
    magnitude_output_path = Path(output_dir) / filename.replace(".mp3", "_magnitude.npy").replace(".wav", "_magnitude.npy")
    np.save(magnitude_output_path, magnitude.astype(np.float32))

    if save_phase:
        phase_output_path = Path(output_dir) / filename.replace(".mp3", "_phase.npy").replace(".wav", "_phase.npy")
        np.save(phase_output_path, phase.astype(np.float32))

    return f"Processed {filename}"

def process_audio_wrapper(args):
    return process_audio(*args)

@click.command()
@click.option("--audio-dir", required=True, type=click.Path(exists=True), help="Directory containing audio files")
@click.option("--output-dir", required=True, type=click.Path(), help="Directory to save processed spectrograms")
@click.option("--n-fft", default=4096, show_default=True, help="FFT window size")
@click.option("--hop-length", default=256, show_default=True, help="Hop length for STFT")
@click.option("--save-phase", is_flag=True, help="Save phase spectrogram along with magnitude")
@click.option("--num-workers", default=cpu_count(), show_default=True, help="Number of parallel workers")
def main(audio_dir, output_dir, n_fft, hop_length, save_phase, num_workers):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all valid audio files
    audio_files = [str(Path(audio_dir) / f) for f in os.listdir(audio_dir) if f.endswith((".mp3", ".wav", ".WAV", ".MP3"))]
    
    # Prepare arguments for parallel processing
    args_list = [(file, output_dir, n_fft, hop_length, save_phase) for file in audio_files]

    # Parallel processing with tqdm progress bar
    with Pool(num_workers) as pool, tqdm(total=len(audio_files), desc="Processing", unit="file") as pbar:
        for _ in pool.imap_unordered(process_audio_wrapper, args_list):
            pbar.update()

if __name__ == "__main__":
    main()
