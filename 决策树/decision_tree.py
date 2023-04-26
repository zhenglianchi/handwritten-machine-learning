import numpy as np
import pandas as pd
from math import log
import operator

path = '决策树\data.txt'
data = np.array(pd.read_csv(path,names=['年龄', '有工作', '有自己的房子', '信贷情况', '是否给贷款']))
label=['年龄', '有工作', '有自己的房子', '信贷情况', '是否给贷款']
#信息熵求解
def get_empirical_entropy(data):
    num=len(data)
    Labelcount={}
    for item in data:
        cur=item[-1]
        if cur not in Labelcount:
            Labelcount[cur]=0
        Labelcount[cur]+=1

    empirical_entropy=0
    for item in Labelcount:
        p=float(Labelcount[item])/num
        empirical_entropy-=p*log(p,2)
    return empirical_entropy

#求解信息增益，并选取最佳特征
def choose_best_node(data):
    num=len(data[0])-1
    empirical_entropy=get_empirical_entropy(data)
    IG=0
    bestIG=0
    bestfeature=-1
    for i in range(num):
        feaList=[item[i] for item in data]
        Feach_value=set(feaList)
        conditional_entropy=0
        for item in Feach_value:
            subdata=split_data(data,i,item)
            p=len(subdata)/float(len(data))
            conditional_entropy+=p*get_empirical_entropy(subdata)
        IG=empirical_entropy-conditional_entropy
        if IG>bestIG:
            bestIG=IG
            bestfeature=i
    return bestfeature


def split_data(data,Findex,value):
    ret=[]
    for line in list(data):
        if line[Findex]==value:
            line=list(line)
            temp=line[:Findex]
            temp.extend(line[Findex+1:])
            ret.append(temp)
    return ret

def majorityCnt(classList):
    classCount={}
    #统计classList中每个元素出现的次数
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    #根据字典的值降序排列
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createTree(data,label,fealabel):
    classList=[item[-1] for item in data]

    if classList.count(classList[0])==len(classList):
        return classList[0]
    #遍历完所有特征时返回出现次数最多的类标签
    if len(data[0])==1:
        return majorityCnt(classList)

    bestFeat=choose_best_node(data)

    bestFeatLabel=label[bestFeat]
    fealabel.append(bestFeatLabel)
    myTree={bestFeatLabel:{}}
    del(label[bestFeat])
    #得到训练集中所有最优特征的属性值
    featValues=[example[bestFeat] for example in data]
    #去掉重复的属性值
    uniqueVls=set(featValues)
    #递归建立决策树
    for value in uniqueVls:
        myTree[bestFeatLabel][value]=createTree(split_data(data,bestFeat,value),label,fealabel)
    return myTree

fealabel=[]
mytree=createTree(data,label,fealabel)
print(mytree)

