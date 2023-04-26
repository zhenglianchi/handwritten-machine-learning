import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

row_data = loadmat('支持向量机/data1.mat')
data = pd.DataFrame(row_data['X'], columns=['x1', 'x2'])
data['y']=row_data['y']
data['y'] = data['y'].map(lambda x: -1 if x == 0 else 1)
data.insert(0, 'ones', 1)

print(data)

cols = data.shape[1]
x=np.matrix(data.iloc[:,0:cols-1])
y=np.matrix(data.iloc[:,cols-1:cols])
w_1=np.matrix(np.zeros(3))
w_2=np.matrix(np.zeros(3))

def h(x,w):
    return x*w.T

def cost(x,y,w,C):
    left=np.sum(np.power(w[:,1:],2))/2
    temp=1-np.multiply(y,h(x,w))
    temp[temp<0]=0
    right=C*np.sum(temp)
    return left+right

def graddecent(x,y,w,C,alpha,count):
    Loss=np.zeros(count)
    for i in range(1,count+1):
        for j in range(len(y)):
            if 1-y[j]*h(x[j,:],w)<0:
                w[:,1:]-=alpha*(2*w[:,1:]*(1/i))
            else:
                w[:,0]-=alpha*(C*(-y[j,:]*x[j,0]))
                w[:,1:]-=alpha*(2*w[:,1:]*(1/i)+C*(-y[j,:]*x[j,1:]))
        Loss[i-1]=cost(x,y,w,C)
        if i%100==0:
            print("第%d次,Loss为%.6f"%(i,Loss[i-1]))
    return w,Loss

alpha=0.003
count=5000
C1=1
C2=100
w1,loss1=graddecent(x,y,w_1,C1,alpha,count)
w2,loss2=graddecent(x,y,w_2,C2,alpha,count)

#查看损失函数
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(np.arange(1,count+1), loss1, 'r')
ax.set_xlabel('count')
ax.set_ylabel('loss')
ax.set_title('C1')
plt.show()

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(np.arange(1,count+1), loss2, 'r')
ax.set_xlabel('count')
ax.set_ylabel('loss')
ax.set_title('C2')
plt.show()


print(w1,'\n',w2)

pos=data[data['y'].isin([1])]
neg=data[data['y'].isin([-1])]
plt.plot(pos['x1'],pos['x2'],'o',c='b')
plt.plot(neg['x1'],neg['x2'],'x',c='r')
x_=np.linspace(x[:,1:].min(),x[:,1:].max(),51)
y_= -(w1[0,0] + x_*w1[0,1]) / w1[0,2]
plt.plot(x_,y_,'r')
plt.title("C=1")
plt.show()

plt.plot(pos['x1'],pos['x2'],'o',c='b')
plt.plot(neg['x1'],neg['x2'],'x',c='r')
x_=np.linspace(x[:,1:].min(),x[:,1:].max(),51)
y_= -(w2[0,0] + x_*w2[0,1]) / w2[0,2]
plt.plot(x_,y_,'r')
plt.title("C=100")
plt.show()