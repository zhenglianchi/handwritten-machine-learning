#导入需要的python库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#将数据从txt中导出
data=pd.read_csv("线性回归\data.txt",names=['x1','y'])
#在第一列插入一列1，方便用于矩阵运算
data.insert(0,'x0',1)
#X为前两列的所有行,y为最后一列,参数theta设定为两个元素全为1的行向量分别对应w和b
X=np.matrix(data.iloc[:,0:2])
y=np.matrix(data.iloc[:,2:])
theta=np.matrix(np.ones(X.shape[1]))
#各个矩阵的形状为X(6,2) y(6,1) theta(1,2)


#利用例子中最小二乘法,分别求偏导后使其等于0的W和b来计算
def example(X,y,theta):
    #这里使用X第二列的元素,因为第一列是我们添加的1
    x=X[:,1:2]
    ##使用矩阵运算减少循环的次数,np.average()为矩阵所有元素的平均值
    x_=np.average(x)
    #np.sum()为矩阵所有元素和,np.power(x,y)为x的y次方
    theta[0,1]=(y.T*(x-x_))/(np.sum(np.power(x,2))-np.power(np.sum(x),2)/len(x))
    theta[0,0]=np.sum(y-theta[0,1]*x)/len(x)
    return theta
theta1=example(X,y,theta)

#以X第一列中从最小值到最大值为x轴,以新求得的theta最为参数求出y作为y轴画图
x1=np.linspace(X[:,1:].min(),X[:,1:].max(),100)
#用求出的theta来计算y
y1=x1*theta1[0,1]+theta1[0,0]
#画出拟合直线
plt.plot(x1,y1,'r',label='test1')
#画出所有的点
plt.plot(X[:,1:],y,'+')
plt.title("test1")
plt.show()


#利用梯度下降法来求解
#这里的梯度下降是看B站吴恩达老师讲自己手写的
def h(X,theta):
    #利用矩阵运算直接求各个x所预测的y  (6,2)*(2,1)为(6,1)每一行为x对应的y
    return X*theta.T

def cost(X,y,theta):
    #损失函数为(预测值-真实值)的平方的平均值的二分之一,同样使用矩阵运算
    temp=np.power(h(X,theta)-y,2)
    return np.sum(temp)/(2*X.shape[0])

def graddescent(X,y,theta,alpha,count):
    #J用来记录每一次迭代损失函数的值
    J=np.matrix(np.ones(count))
    for i in range(count):
        #求出每一个y对应的差值
        error=h(X,theta)-y
        #进行迭代,X.T为(2.6),error为(6,1)结果为(2,1)转置后可以直接用于迭代
        #(X.T*error).T中的每个元素也为对应的偏导数,减少循环次数
        theta=theta-alpha*(X.T*error).T
        #记录损失函数的值
        J[0,i]=cost(X,y,theta)
    return theta,J

#设定参数alpha和迭代次数count的值
alpha=0.00001
count=1000
J1=cost(X,y,theta1)
theta2,J2=graddescent(X,y,theta,alpha,count)

#同上画图
x2=np.linspace(X[:,1:].min(),X[:,1:].max(),100)
y2=x1*theta1[0,1]+theta1[0,0]
plt.plot(x2,y2,'r',label='test2')
plt.plot(X[:,1:],y,'+')
plt.title("test2")
plt.show()

#查看结果，两组结果一致
print(theta1,J1)
print(theta2,J2[0,-1:])