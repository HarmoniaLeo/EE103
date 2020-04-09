import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt

plt.figure()
x1,sr = librosa.load("102_Vox_Lead.wav",sr=None)
x2,sr = librosa.load("MUSIC STEM.wav",sr=None)
y1=0.4*x1+0.6*x2
y2=0.6*x1+0.4*x2
y=np.stack((y1.T,y2.T),axis=0)
y=np.asfortranarray(y)
plt.subplot(2, 1, 1)
plt.subplot(2, 1, 1).set_title("left")
librosa.display.waveplot(np.asfortranarray(y[0]), sr)
plt.subplot(2, 1, 2)
plt.subplot(2, 1, 2).set_title("right")
librosa.display.waveplot(np.asfortranarray(y[1]), sr)
librosa.output.write_wav("mix-stereo.wav",y,sr)
plt.show()