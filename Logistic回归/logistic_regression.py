import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'Logistic回归\data.txt'
data = pd.read_csv(path, names=['x1', 'x2', 'y'])
data.insert(0, 'Ones', 1)
cols = data.shape[1]

x=np.matrix(data.iloc[:,0:cols-1])
y=np.matrix(data.iloc[:,cols-1:cols])
theta=np.matrix(np.zeros(3))
#看一下x,y,theta的形状
#print(x.shape,y.shape,theta.shape)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(x,y,theta):
    left=np.multiply(-y,np.log(sigmoid(x*theta.T)))
    right=np.multiply(1-y,np.log(1-sigmoid(x*theta.T)))
    return np.sum(left-right)/len(x)

def graddecent(x,y,theta,alpha,count):
    J=np.matrix(np.zeros(count))
    for i in range(count):
        error=sigmoid(x*theta.T)-y
        theta=theta-alpha*(x.T*error).T
        J[:,i]=cost(x,y,theta)
    return theta,J

alpha=0.00001
count=100000
theta,J=graddecent(x,y,theta,alpha,count)

def predict(x,theta):
    res=sigmoid(x*theta.T)
    prediction=[]
    for i in res:
        if i>=0.5:
            prediction.append(1)
        else:
            prediction.append(0)
    accuracy=0
    for i in range(len(prediction)):
        if prediction[i]==y[i]:
            accuracy+=1
    accuracy_rating=accuracy/len(prediction)
    return accuracy_rating

print(theta,J[:,-1])
print('准确率:%.2f'%predict(x,theta))

pos=data[data['y'].isin([1])]
neg=data[data['y'].isin([0])]
x1=np.arange(15,110,step=0.1)
x2 = -(theta[0,0] + x1*theta[0,1]) / theta[0,2]
plt.plot(pos['x1'],pos['x2'],'o',c='b')
plt.plot(neg['x1'],neg['x2'],'x',c='r')
plt.plot(x1,x2,c='g')
plt.show()