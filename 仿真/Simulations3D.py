import numpy as np
from lagb import *
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

colors=["b","r","y"]
m=5
T=100
h=0.1
eta=0.05
omegas=[[0,0,0],[-10,-10,0],[-10,-10,-10]]
vs=[50,75,100]
thetas=[[45,30],[45,45],[45,80]]
g=-9.8
fig=plt.figure()
count=1
for theta in thetas:
    for v in vs:
        count2=0
        ax = fig.add_subplot(3, 3, count, projection='3d')
        for omega in omegas:
            x=np.array([0,0,0,v*math.cos(math.radians(theta[0]))*math.sin(math.radians(theta[1])),\
                v*math.sin(math.radians(theta[0]))*math.sin(math.radians(theta[1])),\
                    v*math.cos(math.radians(theta[1]))])
            A=np.array([[1,0,0,h,0,0],[0,1,0,0,h,0],[0,0,1,0,0,h],\
                [0,0,0,1-h*eta/m,0,0],[0,0,0,0,1-h*eta/m,0],[0,0,0,0,0,1-h*eta/m]])
            b=np.array([0,0,0,h*eta*omega[0]/m,h*eta*omega[1]/m,h*g+h*eta*omega[2]/m])
            xs=[x[0]]
            ys=[x[1]]
            zs=[x[2]]
            for i in range(0,T):
                x=dot(A,x)+b
                xs.append(x[0])
                ys.append(x[1])
                zs.append(x[2])
            ax.plot(xs,ys,zs,color=colors[count2],label="ω={0}".format(omega))
            count2+=1
        ax.set_title("θ={0}° v={1}".format(theta,v))
        ax.legend()
        count+=1
plt.show()