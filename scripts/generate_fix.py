import torch
import librosa as li
import soundfile as sf

x, sr = li.load("input_audio.wav", sr=44100)

model = torch.jit.load("model_filename.ts").eval()
x = torch.from_numpy(x).reshape(1,1,-1)
z = model.encode(x)
x_hat = model.decode(z).detach().numpy().reshape(-1)

sf.write("output_audio.wav", x_hat, sr)
