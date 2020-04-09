import numpy as np
from Function import Function	#定义法求导工具
from lagb import *	#线性代数工具库
from scipy import linalg
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D

n=3 #待定系数数
K=10
omegas=np.random.randint(5,15,(K,3))
etas=np.random.uniform(0,0.1,K)
m=5
T=10
h=1
g=-9.8
p0=np.array([0,0,0])
pt=np.array([200,200,0])

def myFunc(x):
    res=0
    for l in range(0,K):
        A=np.array([[1,0,0,h,0,0],[0,1,0,0,h,0],[0,0,1,0,0,h],\
                [0,0,0,1-h*etas[l]/m,0,0],[0,0,0,0,1-h*etas[l]/m,0],[0,0,0,0,0,1-h*etas[l]/m]])
        b=np.array([0,0,0,h*etas[l]*omegas[l][0]/m,h*etas[l]*omegas[l][1]/m,h*g+h*etas[l]*omegas[l][2]/m])
        F=A
        for i in range(1,T):
            F=dot(F,A)
        F11=F[:3,:3]
        F12=F[:3,3:]
        j=np.zeros(6)
        for i in range(0,T):
            jtmp=b
            for k in range(0,i):
                jtmp=dot(A,jtmp)
            j+=jtmp
        j1=j[:3]
        C=F12
        d=dot(F11,p0)+j1
        res+=(dot(C,np.array(x))+d-pt)[0]**2+(dot(C,np.array(x))+d-pt)[1]**2+(dot(C,np.array(x))+d-pt)[2]**2
    return res/K

x=np.zeros(n)
e=0.001
beta1=1
sigma=0.4
rho=0.55
tar=Function(myFunc)
k=0
d=-tar.grad(x)
while tar.norm(x)>e:
    a=1
    if not (tar.value(x+a*d)<=tar.value(x)+rho*a*dot(turn(tar.grad(x)),d) and \
        np.abs(dot(turn(tar.grad(x+a*d)),d))>=sigma*dot(turn(tar.grad(x)),d)):
        a=beta1
        while tar.value(x+a*d)>tar.value(x)+rho*a*dot(turn(tar.grad(x)),d):
            a*=rho
        while np.abs(dot(turn(tar.grad(x+a*d)),d))<sigma*dot(turn(tar.grad(x)),d):
            a1=a/rho
            da=a1-a
            while tar.value(x+(a+da)*d)>tar.value(x)+rho*(a+da)*dot(turn(tar.grad(x)),d):
                da*=rho
            a+=da
    lx=x
    x=x+a*d
    g0=tar.grad(x)
    beta=np.max((dot(turn(g0),g0-tar.grad(lx))/(tar.norm(lx)**2),0))	#PRP+
    d=-g0+beta*d
    k+=1
    print(k)
v0=x
fig = plt.figure()
axes3d = Axes3D(fig)
for eta in etas:
    for omega in omegas:
        x=np.array([0,0,0,v0[0],v0[1],v0[2]])
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
        axes3d.plot(xs,ys,zs)
print(v0)
plt.show()