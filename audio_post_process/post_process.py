import click
import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
from pydub.effects import high_pass_filter, low_pass_filter

@click.command()
@click.option('--input-mp3', default="sugar_plum_lyrebird_reconstructed_audio_with_phase.mp3", help="Input MP3 file path.")
@click.option('--output-wav', default="output_denoised_compressed_smoothed.wav", help="Output WAV file path.")
@click.option('--output-mp3', default="output_denoised_compressed_smoothed.mp3", help="Output MP3 file path.")
@click.option('--prop-decrease', default=0.8, type=float, help="Noise reduction strength (prop_decrease).")
@click.option('--compress', is_flag=True, help="Apply dynamic range compression.")
@click.option('--low-pass', default=3000, type=int, help="Low pass filter cutoff frequency (Hz).")
@click.option('--high-pass', default=100, type=int, help="High pass filter cutoff frequency (Hz).")
def post_process(input_mp3, output_wav, output_mp3, prop_decrease, compress, low_pass, high_pass):
    """Apply denoising, compression, and smoothing to an audio file."""
    
    # Load MP3 file
    audio, sr = librosa.load(input_mp3, sr=None)  # Keep original sample rate

    # Denoising
    print("Applying noise reduction...")
    denoised_audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease)
    
    # Save denoised audio as WAV
    sf.write(output_wav, denoised_audio, sr)
    print(f"Denoised audio saved as {output_wav}")
    
    # Load the WAV file into pydub for further processing
    audio_segment = AudioSegment.from_wav(output_wav)

    # Apply compression if flag is set
    if compress:
        print("Applying dynamic range compression...")
        compressed_audio = effects.compress_dynamic_range(audio_segment)
        audio_segment = compressed_audio
    
    # Apply smoothing (low-pass and high-pass filters)
    print(f"Applying low-pass filter with cutoff {low_pass} Hz...")
    smoothed_audio = low_pass_filter(audio_segment, cutoff=low_pass)
    print(f"Applying high-pass filter with cutoff {high_pass} Hz...")
    smoothed_audio = high_pass_filter(smoothed_audio, cutoff=high_pass)

    # Save the smoothed audio as MP3
    smoothed_audio.export(output_mp3, format="mp3")
    print(f"Smoothed audio saved as {output_mp3}")

if __name__ == '__main__':
    post_process()