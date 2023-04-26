#判断一个具有特征{色泽=青绿,根蒂=蜷缩,敲声=浊响,纹理=清晰,脐部=凹陷,触感=硬滑,密度=0.697,含糖率=0.460}的测试样例瓜是否为好瓜。

import math
import numpy as np
import pandas as pd
def Continuous_p(x,ave,std):
    left=1/(np.sqrt(2*math.pi)*std)
    right=np.exp((-np.power(x-ave,2))/(2*np.power(std,2)))
    return round(left*right,3)


#测试样例
test=['青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,0.460]

data=pd.read_csv("朴素贝叶斯分类器\data.txt",names=['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率','好瓜'])
def Bayes(data,test):
    d={'好瓜':0,'坏瓜':0,'好瓜'+test[0]:0,'坏瓜'+test[0]:0,'好瓜'+test[1]:0,'坏瓜'+test[1]:0,'好瓜'+test[2]:0,'坏瓜'+test[2]:0,
    '好瓜'+test[3]:0,'坏瓜'+test[3]:0,'好瓜'+test[4]:0,'坏瓜'+test[4]:0,'好瓜'+test[5]:0,'坏瓜'+test[5]:0,
    '好瓜密度ave':0,'好瓜密度var':0,'好瓜含糖率ave':0,'好瓜含糖率var':0,'坏瓜密度ave':0,'坏瓜密度var':0,'坏瓜含糖率ave':0,'坏瓜含糖率var':0}
    good_density=[]
    bad_density=[]
    good_sugar=[]
    bad_sugar=[]
    for item in np.array(data):
        if item[8]=='是':
            d['好瓜']+=1
            good_density.append(item[6])
            good_sugar.append(item[7])
            if item[0]==test[0]:
                d['好瓜'+test[0]]+=1
            if item[1]==test[1]:
                d['好瓜'+test[1]]+=1
            if item[2]==test[2]:
                d['好瓜'+test[2]]+=1
            if item[3]==test[3]:
                d['好瓜'+test[3]]+=1
            if item[4]==test[4]:
                d['好瓜'+test[4]]+=1
            if item[5]==test[5]:
                d['好瓜'+test[5]]+=1
        elif item[8]=='否':
            d['坏瓜']+=1
            bad_density.append(item[6])
            bad_sugar.append(item[7])
            if item[0]==test[0]:
                d['坏瓜'+test[0]]+=1
            if item[1]==test[1]:
                d['坏瓜'+test[1]]+=1
            if item[2]==test[2]:
                d['坏瓜'+test[2]]+=1
            if item[3]==test[3]:
                d['坏瓜'+test[3]]+=1
            if item[4]==test[4]:
                d['坏瓜'+test[4]]+=1
            if item[5]==test[5]:
                d['坏瓜'+test[5]]+=1

    d['好瓜密度ave']=np.average(good_density)
    d['好瓜密度var']=np.std(good_density,ddof=1)
    d['好瓜含糖率ave']=np.average(good_sugar)
    d['好瓜含糖率var']=np.std(good_sugar,ddof=1)
    d['坏瓜密度ave']=np.average(bad_density)
    d['坏瓜密度var']=np.std(bad_density,ddof=1)
    d['坏瓜含糖率ave']=np.average(bad_sugar)
    d['坏瓜含糖率var']=np.std(bad_sugar,ddof=1)

    p_1_p=round(d['好瓜']/len(data['好瓜']),3)
    p_0_p=round(d['坏瓜']/len(data['好瓜']),3)
    p_x0_1=round(d['好瓜'+test[0]]/d['好瓜'],3)
    p_x0_0=round(d['坏瓜'+test[0]]/d['坏瓜'],3)
    p_x1_1=round(d['好瓜'+test[1]]/d['好瓜'],3)
    p_x1_0=round(d['坏瓜'+test[1]]/d['坏瓜'],3)
    p_x2_1=round(d['好瓜'+test[2]]/d['好瓜'],3)
    p_x2_0=round(d['坏瓜'+test[2]]/d['坏瓜'],3)
    p_x3_1=round(d['好瓜'+test[3]]/d['好瓜'],3)
    p_x3_0=round(d['坏瓜'+test[3]]/d['坏瓜'],3)
    p_x4_1=round(d['好瓜'+test[4]]/d['好瓜'],3)
    p_x4_0=round(d['坏瓜'+test[4]]/d['坏瓜'],3)
    p_x5_1=round(d['好瓜'+test[5]]/d['好瓜'],3)
    p_x5_0=round(d['坏瓜'+test[5]]/d['坏瓜'],3)

    p_x6_1=Continuous_p(test[6],d['好瓜密度ave'],d['好瓜密度var'])
    p_x6_0=Continuous_p(test[6],d['坏瓜密度ave'],d['坏瓜密度var'])
    p_x7_1=Continuous_p(test[7],d['好瓜含糖率ave'],d['好瓜含糖率var'])
    p_x7_0=Continuous_p(test[7],d['坏瓜含糖率ave'],d['坏瓜含糖率var'])
    return p_0_p,p_1_p,p_x0_0,p_x0_1,p_x1_0,p_x1_1,p_x2_0,p_x2_1,p_x3_0,p_x3_1,p_x4_0,p_x4_1,p_x5_0,p_x5_1,p_x6_0,p_x6_1,p_x7_0,p_x7_1


p_0_p,p_1_p,p_x0_0,p_x0_1,p_x1_0,p_x1_1,p_x2_0,p_x2_1,p_x3_0,p_x3_1,p_x4_0,p_x4_1,p_x5_0,p_x5_1,p_x6_0,p_x6_1,p_x7_0,p_x7_1=Bayes(data,test)
p_1=p_1_p*p_x0_1*p_x1_1*p_x2_1*p_x3_1*p_x4_1*p_x5_1*p_x6_1*p_x7_1
p_0=p_0_p*p_x0_0*p_x1_0*p_x2_0*p_x3_0*p_x4_0*p_x5_0*p_x6_0*p_x7_0

#看一下结果是否和博客相同
#和博客不同，博客中的好瓜凹陷是6个其实数据中只有五个，数据中好瓜蜷缩是0.375其实应该是0.625
print(p_1,p_0)
if p_1>p_0:
    print('这是好瓜')
else:
    print('这是坏瓜')




#拉普拉斯修正
#对于某个离散值 如果训练集没有与某个类同时出现过，那么其p(x|c)=0，这会导致连乘时不管其他属性取值，其概率都为0
#采用拉普拉斯修正  p(c)=(Dc+1)/(D+N)  p(c|x)=(Dcxi+1)/(Di+Ni)
test=['青绿','蜷缩','清脆','清晰','凹陷','硬滑',0.697,0.460]
#在好瓜中没有清脆这一类别，因此采用拉普拉斯修正
def Bayes(data,test):
    d={'好瓜':0,'坏瓜':0,'好瓜'+test[0]:0,'坏瓜'+test[0]:0,'好瓜'+test[1]:0,'坏瓜'+test[1]:0,'好瓜'+test[2]:0,'坏瓜'+test[2]:0,
    '好瓜'+test[3]:0,'坏瓜'+test[3]:0,'好瓜'+test[4]:0,'坏瓜'+test[4]:0,'好瓜'+test[5]:0,'坏瓜'+test[5]:0,
    '好瓜密度ave':0,'好瓜密度var':0,'好瓜含糖率ave':0,'好瓜含糖率var':0,'坏瓜密度ave':0,'坏瓜密度var':0,'坏瓜含糖率ave':0,'坏瓜含糖率var':0}
    good_density=[]
    bad_density=[]
    good_sugar=[]
    bad_sugar=[]
    for item in np.array(data):
        if item[8]=='是':
            d['好瓜']+=1
            good_density.append(item[6])
            good_sugar.append(item[7])
            if item[0]==test[0]:
                d['好瓜'+test[0]]+=1
            if item[1]==test[1]:
                d['好瓜'+test[1]]+=1
            if item[2]==test[2]:
                d['好瓜'+test[2]]+=1
            if item[3]==test[3]:
                d['好瓜'+test[3]]+=1
            if item[4]==test[4]:
                d['好瓜'+test[4]]+=1
            if item[5]==test[5]:
                d['好瓜'+test[5]]+=1
        elif item[8]=='否':
            d['坏瓜']+=1
            bad_density.append(item[6])
            bad_sugar.append(item[7])
            if item[0]==test[0]:
                d['坏瓜'+test[0]]+=1
            if item[1]==test[1]:
                d['坏瓜'+test[1]]+=1
            if item[2]==test[2]:
                d['坏瓜'+test[2]]+=1
            if item[3]==test[3]:
                d['坏瓜'+test[3]]+=1
            if item[4]==test[4]:
                d['坏瓜'+test[4]]+=1
            if item[5]==test[5]:
                d['坏瓜'+test[5]]+=1

    d['好瓜密度ave']=np.average(good_density)
    d['好瓜密度var']=np.std(good_density,ddof=1)
    d['好瓜含糖率ave']=np.average(good_sugar)
    d['好瓜含糖率var']=np.std(good_sugar,ddof=1)
    d['坏瓜密度ave']=np.average(bad_density)
    d['坏瓜密度var']=np.std(bad_density,ddof=1)
    d['坏瓜含糖率ave']=np.average(bad_sugar)
    d['坏瓜含糖率var']=np.std(bad_sugar,ddof=1)

    num=len(np.unique(list(data['好瓜'])))
    num0=len(np.unique(list(data['色泽'])))
    num1=len(np.unique(list(data['根蒂'])))
    num2=len(np.unique(list(data['敲声'])))
    num3=len(np.unique(list(data['纹理'])))
    num4=len(np.unique(list(data['脐部'])))
    num5=len(np.unique(list(data['触感'])))
    p_1_p=round((d['好瓜']+1)/(len(data['好瓜'])+num),3)
    p_0_p=round((d['坏瓜']+1)/(len(data['好瓜'])+num),3)
    p_x0_1=round((d['好瓜'+test[0]]+1)/(d['好瓜']+num0),3)
    p_x0_0=round((d['坏瓜'+test[0]]+1)/(d['坏瓜']+num0),3)
    p_x1_1=round((d['好瓜'+test[1]]+1)/(d['好瓜']+num1),3)
    p_x1_0=round((d['坏瓜'+test[1]]+1)/(d['坏瓜']+num1),3)
    p_x2_1=round((d['好瓜'+test[2]]+1)/(d['好瓜']+num2),3)
    p_x2_0=round((d['坏瓜'+test[2]]+1)/(d['坏瓜']+num2),3)
    p_x3_1=round((d['好瓜'+test[3]]+1)/(d['好瓜']+num3),3)
    p_x3_0=round((d['坏瓜'+test[3]]+1)/(d['坏瓜']+num3),3)
    p_x4_1=round((d['好瓜'+test[4]]+1)/(d['好瓜']+num4),3)
    p_x4_0=round((d['坏瓜'+test[4]]+1)/(d['坏瓜']+num4),3)
    p_x5_1=round((d['好瓜'+test[5]]+1)/(d['好瓜']+num5),3)
    p_x5_0=round((d['坏瓜'+test[5]]+1)/(d['坏瓜']+num5),3)

    p_x6_1=Continuous_p(test[6],d['好瓜密度ave'],d['好瓜密度var'])
    p_x6_0=Continuous_p(test[6],d['坏瓜密度ave'],d['坏瓜密度var'])
    p_x7_1=Continuous_p(test[7],d['好瓜含糖率ave'],d['好瓜含糖率var'])
    p_x7_0=Continuous_p(test[7],d['坏瓜含糖率ave'],d['坏瓜含糖率var'])
    return p_0_p,p_1_p,p_x0_0,p_x0_1,p_x1_0,p_x1_1,p_x2_0,p_x2_1,p_x3_0,p_x3_1,p_x4_0,p_x4_1,p_x5_0,p_x5_1,p_x6_0,p_x6_1,p_x7_0,p_x7_1


p_0_p,p_1_p,p_x0_0,p_x0_1,p_x1_0,p_x1_1,p_x2_0,p_x2_1,p_x3_0,p_x3_1,p_x4_0,p_x4_1,p_x5_0,p_x5_1,p_x6_0,p_x6_1,p_x7_0,p_x7_1=Bayes(data,test)
p_1=p_1_p*p_x0_1*p_x1_1*p_x2_1*p_x3_1*p_x4_1*p_x5_1*p_x6_1*p_x7_1
p_0=p_0_p*p_x0_0*p_x1_0*p_x2_0*p_x3_0*p_x4_0*p_x5_0*p_x6_0*p_x7_0
#查看p(清脆|好瓜)是否等于博客中修正过的值
#print(p_x2_1)

print(p_1,p_0)
if p_1>p_0:
    print('这是好瓜')
else:
    print('这是坏瓜')


