import os
import pytest
import numpy as np
import soundfile as sf
from click.testing import CliRunner
from stft.stft import main  # Ensure correct import

@pytest.fixture
def tmp_audio_dir(tmp_path):
    """Create a temporary directory with a valid test audio file"""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Generate a test sine wave
    sample_rate = 22050
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    # Save as a valid WAV file
    test_audio_path = audio_dir / "test.wav"
    sf.write(test_audio_path, sine_wave, sample_rate)

    return str(audio_dir)

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create a temporary output directory"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)

def test_stft_cli(tmp_audio_dir, tmp_output_dir):
    """Test the functionality of the STFT CLI tool"""
    runner = CliRunner()
    
    # Run CLI
    result = runner.invoke(
        main,
        [
            "--audio-dir", tmp_audio_dir,
            "--output-dir", tmp_output_dir,
            "--n-fft", "2048",
            "--hop-length", "512",
            "--save-phase",
            "--num-workers", "2",
        ]
    )
    
    # Assert CLI success
    assert result.exit_code == 0, f"CLI tool failed: {result.output}"

    # Check output files exist
    output_files = os.listdir(tmp_output_dir)
    assert any(f.endswith("_magnitude.npy") for f in output_files), "Magnitude file not generated"
    assert any(f.endswith("_phase.npy") for f in output_files), "Phase file not generated"

# import os
# import tempfile
# import pytest
# from click.testing import CliRunner
# from stft.stft_cli import main  # assuming stft_cli.py is the name of your script

# @pytest.fixture
# def tmp_audio_dir():
#     """Fixture to create a temporary directory with test audio files"""
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         # Create a temporary .mp3 file
#         tmp_file = os.path.join(tmp_dir, "test_audio.mp3")
#         with open(tmp_file, "wb") as f:
#             f.write(b"fake_audio_data")
#         yield tmp_dir  # Return the directory path
#         # Cleanup happens automatically when exiting the context

# @pytest.fixture
# def tmp_output_dir():
#     """Fixture to create a temporary output directory for results"""
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         yield tmp_dir  # Return the directory path
#         # Cleanup happens automatically when exiting the context

# def test_stft_cli(tmp_audio_dir, tmp_output_dir):
#     """Test the functionality of the STFT CLI tool"""
    
#     # Initialize the CLI runner
#     runner = CliRunner()

#     # Call the CLI tool with test options
#     result = runner.invoke(
#         main,
#         [
#             "--audio-dir", tmp_audio_dir,
#             "--output-dir", tmp_output_dir,
#             "--n-fft", "2048",  # Testing with a custom n_fft value
#             "--hop-length", "512",  # Testing with a custom hop_length value
#             "--save-phase",  # Testing phase saving flag
#             "--num-workers", "2",  # Test with multiple workers
#         ]
#     )

#     # Check if the CLI ran successfully
#     assert result.exit_code == 0, f"CLI tool failed: {result.output}"

#     # Check if the output directory contains the expected files
#     assert any(f.endswith("_magnitude.npy") for f in os.listdir(tmp_output_dir)), "Magnitude file not found"
#     assert any(f.endswith("_phase.npy") for f in os.listdir(tmp_output_dir)), "Phase file not found"
    
#     # Additional checks based on your needs
#     assert "Processed test_audio.mp3" in result.output, "Processing message not found in output"