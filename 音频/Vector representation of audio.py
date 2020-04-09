import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt

y,sr = librosa.load("MUSIC STEM.wav",sr=None)	#y为长度等于采样率sr*时间的音频向量
plt.figure()
librosa.display.waveplot(y, sr)	#创建波形图
plt.show()	#显示波形图