import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt

plt.figure()
y,sr = librosa.load("MUSIC STEM.wav",sr=None)	#y为长度等于采样率sr*时间的音频向量
y1=y*2
librosa.output.write_wav("scaling2.wav",y1,sr)
plt.subplot(2, 2, 1)
plt.subplot(2, 2, 1).set_title("y=2y")
librosa.display.waveplot(y1, sr)	#显示波形图
y2=y*0.5
librosa.output.write_wav("scaling05.wav",y2,sr)
plt.subplot(2, 2, 2)
plt.subplot(2, 2, 2).set_title("y=0.5y")
librosa.display.waveplot(y2, sr)	#显示波形图
y3=-y
librosa.output.write_wav("scaling-1.wav",y3,sr)
plt.subplot(2, 2, 3)
plt.subplot(2, 2, 3).set_title("y=-y")
librosa.display.waveplot(y3, sr)	#显示波形图
y4=y*10
librosa.output.write_wav("scaling10.wav",y4,sr)
plt.subplot(2, 2, 4)
plt.subplot(2, 2, 4).set_title("y=10y")
librosa.display.waveplot(y4, sr)	#显示波形图
plt.show()