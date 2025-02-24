import click
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import io

@click.command()
@click.option('--audio-path', help="Path to the original audio file.")
@click.option('--spectrogram-path', help="Path to the retrieved slices spectrogram (B).")
@click.option('--phase-path', help="Path to the phase spectrogram (A_Phase).")
@click.option('--output-wav', help="Output WAV file path.")
@click.option('--output-mp3', help="Output MP3 file path.")
@click.option('--n-fft', default=4096, type=int, help="FFT window size for ISTFT.")
@click.option('--hop-length', default=None, type=int, help="Hop length for ISTFT. If not provided, it will be computed as n_fft // 16.")
def reconstruct_audio(audio_path, spectrogram_path, phase_path, output_wav, output_mp3, n_fft, hop_length):
    """Reconstruct audio from magnitude spectrogram B and phase A_Phase."""
    
    # Load original audio to get the sample rate
    audio, sr = librosa.load(audio_path, sr=None)  # Keep original sample rate

    # Load magnitude spectrogram B and phase A_Phase
    B = np.load(spectrogram_path)
    A_Phase = np.load(phase_path)

    # Ensure shapes match
    assert B.shape == A_Phase.shape, "B and A_Phase must have the same shape!"

    # Correct hop length based on what worked with Griffin-Lim
    if hop_length is None:
        hop_length = n_fft // 16  # Ensure this matches original settings

    # Construct complex spectrogram using B and A_Phase
    B_complex = B * np.exp(1j * A_Phase)

    # ISTFT to reconstruct audio
    reconstructed_audio = librosa.istft(B_complex.T, hop_length=hop_length, n_fft=n_fft)

    # Save as WAV
    sf.write(output_wav, reconstructed_audio, sr)
    print(f"Reconstructed audio (with phase) saved as {output_wav}")

    # Convert to MP3 using pydub
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, reconstructed_audio, sr, format='WAV')
    audio_buffer.seek(0)  # Rewind buffer

    # Export as MP3
    audio_segment = AudioSegment.from_wav(audio_buffer)
    audio_segment.export(output_mp3, format="mp3")

    print(f"Reconstructed MP3 (with phase) saved as {output_mp3}")

if __name__ == '__main__':
    reconstruct_audio()