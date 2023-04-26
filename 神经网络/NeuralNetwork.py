from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris=load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target
data=data.iloc[:100,:]
#这里我进行二分类，只将0,1拿出来进行分类
#四个输入神经元 一个隐藏层 四个隐藏神经元 （这里不包括偏置项，实际计算包括偏置项） 一个输出神经元
#设置两个theta  每一个表示一层的权重  一个theta有四行五列 一个theta有一行五列 每一行表示后一个神经元的输入权重(随机初始化)
theta0=np.matrix(np.random.random((4,5)))
theta1=np.matrix(np.random.random((1,5)))
train=[]
test=[]
num=1
for item in np.array(data):
    if num==5:
        test.append(item)
        num=1
    else:
        train.append(item)
        num+=1
train=pd.DataFrame(train)
test=pd.DataFrame(test)
train.insert(0,'Bias',1)
test.insert(0,'Bias',1)
cols = train.shape[1]
x_train=np.matrix(train.iloc[:,0:cols-1])
y_train=np.matrix(train.iloc[:,cols-1:cols])
cols = test.shape[1]
x_test=np.matrix(test.iloc[:,0:cols-1])
y_test=np.matrix(test.iloc[:,cols-1:cols])

def sigmoid(z):
    return 1/(1+np.exp(-z))

def grad_sigmoid(z):
    f=sigmoid(z)
    return np.multiply(f,(1-f))

def loss(y_true,y_pred):
    m = y_pred.shape[0]
    logprobs = np.multiply(np.log(y_pred), y_true) + np.multiply((1 - y_true), np.log(1 - y_pred))
    cost = - np.sum(logprobs) / m
    return cost

def forward_propagation(x,theta0,theta1):
    h=sigmoid(x*theta0.T)
    h=np.insert(h,obj=1,values=0,axis=1)
    o=sigmoid(h*theta1.T)
    return o

def back_propagation(x,y,theta0,theta1,alpha,count):
    l=np.zeros(count)
    for i in range(count):
        for j in range(x.shape[0]):
            h0=1
            sum_h1=float(x[j,:]*theta0[0,:].T)
            sum_h2=float(x[j,:]*theta0[1,:].T)
            sum_h3=float(x[j,:]*theta0[2,:].T)
            sum_h4=float(x[j,:]*theta0[3,:].T)
            h1=sigmoid(sum_h1)
            h2=sigmoid(sum_h2)
            h3=sigmoid(sum_h3)
            h4=sigmoid(sum_h4)
            h=np.matrix([h0,h1,h2,h3,h4])
            sum_h=np.matrix([h0,sum_h1,sum_h2,sum_h3,sum_h4])

            sum_o1=float(h*theta1.T)
            o1=sigmoid(float(sum_o1))

            y_pred=o1
            y_true=y[j]

            delta0=y_pred-y_true
            delta1=delta0*grad_sigmoid(sum_o1)
            
            delta2=float(delta1)*np.multiply(theta1,grad_sigmoid(sum_h))
            
            theta1-=alpha*delta1*h
            theta0-=alpha*np.multiply(x[j,:],delta2)

        y_pred=forward_propagation(x,theta0,theta1)
        l[i]=loss(y,y_pred)
        print("第 %d 次的loss: %.5f"%(i,l[i]))
    return theta0,theta1,l

alpha=0.05
count=300
theta0_bcak,theta1_back,loss_history=back_propagation(x_train,y_train,theta0,theta1,alpha,count)


pred=forward_propagation(x_test,theta0_bcak,theta1_back)

for i in range(len(pred)):
    if pred[i]>=0.5:
        pred[i]=1
    else:
        pred[i]=0

def prediction(predict,y_test):
    correct=0
    for i in range(len(predict)):
        if predict[i] == y_test[i]:
            correct+=1
    return correct/len(y_test)
pred_rate=prediction(pred,y_test)
print("鸢尾花前两种数据正确率:%.2f"%pred_rate)
print(theta0_bcak,theta1_back)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(np.arange(count), loss_history, 'r')
ax.set_xlabel('count')
ax.set_ylabel('loss')
ax.set_title('loss_graph')
plt.show()
