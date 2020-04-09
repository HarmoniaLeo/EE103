import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt

x=np.empty((0,))
for i in range(0,13):
    x1=np.linspace(0,1, num=44100, endpoint=True, dtype=float)
    x1=5*np.sin(2*np.pi*440*np.power(2,i/12)*x1)+5*np.sin(2*np.pi*440*x1)
    x2=np.zeros((44100,))
    x=np.concatenate((x,x1,x2),axis=0)
x=np.asfortranarray(x)
librosa.output.write_wav("intervals.wav",x,44100)