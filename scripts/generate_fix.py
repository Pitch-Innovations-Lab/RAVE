import os
import torch
import librosa as li
import soundfile as sf

import os
import glob

def get_audio_files(directory):
    """
    Get filenames of all WAV, MP3, and FLAC files in the specified directory.
    Allows batch processing of audio files in a directory.
    """

    audio_files = []
    audio_formats = ['wav', 'mp3', 'flac']

    for format in audio_formats:
        # Match WAV or MP3 or FLAC files
        pattern = os.path.join(directory, f'*.{format}')  # Match WAV files
        audio_files.extend(glob.glob(pattern))

    return audio_files


import librosa as li
import soundfile as sf
import torch

def generate(input_path, model_path, sample_rate=44100, output_path="/home/output.wav"):
    """
    Generate audio using a trained model.

    Args:
        input_path (str): Path to the input audio file.
        model_path (str): Path to the model file.
        sample_rate (int, optional): Sampling rate of the input audio and output audio. Defaults to 44100.
        output_path (str, optional): Path to save the generated audio file. Defaults to "/home/output.wav".

    Returns:
        None: The generated audio is saved to the specified output path.
    """
    # Load input audio
    x, sr = li.load(input_path, sr=sample_rate)

    # Load pre-trained model
    model = torch.jit.load(model_path).eval()

    # Encode input audio and generate new audio
    x_tensor = torch.from_numpy(x).reshape(1, 1, -1)
    z = model.encode(x_tensor)
    x_hat = model.decode(z).detach().numpy().reshape(-1)

    # Save generated audio
    sf.write(output_path, x_hat, sr)
    print("Generated audio saved to:", output_path)

"""
#Example usage of script:
directory = '/home/RAVE/inference'
audio_files = get_audio_files(directory)
for file in audio_files:
    output_path = "".join([file[:-4], "_output", file[-4:]])
    generate(file, "model.ts", output_path=output_path)
"""
