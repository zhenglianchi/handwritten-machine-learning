import numpy as np
import pandas as pd

path = 'K近邻\data.txt'
data = pd.read_csv(path, names=['x1', 'x2', 'y'])

minValue=data.min()
maxValue=data.max()
ranges=maxValue-minValue
data=(data-np.tile(minValue,(np.matrix(data).shape[0],1)))/np.tile(ranges,(np.matrix(data).shape[0],1))


train=[]
test=[]
num=1
for item in np.array(data):
    if num==4:
        test.append(item)
        num=1
    else:
        train.append(item)
        num+=1


cols = pd.DataFrame(train).shape[1]
x_train=np.array(pd.DataFrame(train).iloc[:,0:cols-1])
y_train=np.array(pd.DataFrame(train).iloc[:,cols-1:cols])
cols = pd.DataFrame(test).shape[1]
x_test=np.array(pd.DataFrame(test).iloc[:,0:cols-1])
y_test=np.array(pd.DataFrame(test).iloc[:,cols-1:cols])


def distance(p1, p2):
    return np.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

def KNN(K,x_train,y_train,x_test,y_test):
    predict=[]
    for item_test in x_test:
        dist=[]
        for item_train in x_train:
            dist.append(distance(item_test,item_train))
        #对其索引进行升序排列
        nn_index=np.argsort(dist)
        #求出K个最近点的索引值
        nn_y=[]
        for i in nn_index[:K]:
            nn_y.append(y_train[i])
        if list(nn_y).count(0)>list(nn_y).count(1):
            predict.append(0)
        else:
            predict.append(1)
    return predict

def prediction(predict,y_test):
    correct=0
    for i in range(len(predict)):
        if predict[i] == y_test[i]:
            correct+=1
    return correct/len(y_test)


predict=KNN(5,x_train,y_train,x_test,y_test)
correct_rate=prediction(predict,y_test)
print("自主测试数据准确率为：%.2f"%correct_rate)


