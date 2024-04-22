import torch
import librosa as li
import soundfile as sf

x, sr = li.load(“path-to-input-audio.wav", sr=44100)

model = torch.jit.load(“path-to-torchscript-model.ts").eval()
x = torch.from_numpy(x).reshape(1,1,-1)
z = model.encode(x)
x_hat = model.decode(z).detach().numpy().reshape(-1)

sf.write("model_output.wav", x_hat, sr)
