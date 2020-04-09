import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt

plt.figure()
x1,sr = librosa.load("102_Vox_Lead.wav",sr=None)
x2,sr = librosa.load("MUSIC STEM.wav",sr=None)
plt.subplot(2, 3, 1)
plt.subplot(2, 3, 1).set_title("vocal")
librosa.display.waveplot(x1, sr)
plt.subplot(2, 3, 2)
plt.subplot(2, 3, 2).set_title("vocal off")
librosa.display.waveplot(x2, sr)
plt.subplot(2, 3, 4)
plt.subplot(2, 3, 4).set_title("mix1")
y=0.25*x1+0.75*x2
librosa.display.waveplot(y, sr)
librosa.output.write_wav("mix1.wav",y,sr)
plt.subplot(2, 3, 5)
plt.subplot(2, 3, 5).set_title("mix2")
y=0.5*x1+0.5*x2
librosa.display.waveplot(y, sr)
librosa.output.write_wav("mix2.wav",y,sr)
plt.subplot(2, 3, 6)
plt.subplot(2, 3, 6).set_title("mix3")
y=0.6*x1+0.4*x2
librosa.display.waveplot(y, sr)
librosa.output.write_wav("mix3.wav",y,sr)
plt.show()