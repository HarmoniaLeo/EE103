import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt

plt.figure()
m=np.empty((0,))
count=1
for i in (1,5,10,20,50):
    f=np.linspace(220,11000,num=i,endpoint=True).reshape(1,i)
    a=np.random.uniform(0,1,(1,i))
    b=np.random.uniform(0,1,(1,i))
    x=np.linspace(0,1, num=44100, endpoint=True, dtype=float).reshape(44100,1)
    res=a*np.sin(2*np.pi*f*x)+b*np.cos(2*np.pi*f*x)
    res=np.sum(res,axis=1)
    plt.subplot(2,3,count)
    plt.subplot(2,3,count).set_title("mix of {0}".format(i))
    count+=1
    librosa.display.waveplot(res[:200], 44100)
    x2=np.zeros((44100,))
    m=np.concatenate((m,res,x2),axis=0)
m=np.asfortranarray(m)
plt.show()
librosa.output.write_wav("timbres.wav",m,44100)