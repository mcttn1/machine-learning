# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:57:50 2018

@author: Administrator
"""

from numpy import * #科学计算包
import operator #运算符模块
from os import listdir

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels


def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0] #返回dataSet这个array的行数
    #距离计算
    diffMat=tile(inX,(dataSetSize,1))-dataSet #tile函数作用是将inX向量补成大小为(dataSetSize,1)的矩阵，方便和dataSet做减法
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)#参数axis影响对矩阵求和时的顺序，axis=1按照矩阵的行求和，axis=0按照矩阵的列求和
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()#argsort（）函数对向量的中的每个元素排序，结果是元素的索引形成的向量
    classCount={}
    #选择距离最小的k个点
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1#返回字典classCount中votellabel元素对应的值，
        #若无，则字典classCount中生成votellabel元素，并使其对应的数字为0，即classCount={votellbel：0}
        #此时classCount.get(votellabel,0)作用是检测并生成新元素，括号中的0只用作初始化
        #当字典中有votellabel元素时，classCount.get(votellabel,0)作用是返回该元素对应的值
    #排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #classCount.iteritems()作用是将字典classCount分解为元祖列表，若classCount={'A':1,'B':2}则分解为['A','B']与[1,2]两组
    #key=operator.itemgetter(1)以元祖的第二列排序
    return sortedClassCount[0][0]

# =============================================================================
# group,labels=createDataSet()
# print(classify0([0,0],group,labels,3))
# =============================================================================

#将文本记录到转换NumPy的解析程序
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines) #1得到文件行数
    returnMat=zeros((numberOfLines,3)) #2创建返回的NumPy矩阵zeros(shape,dtype=float)指定数据类型的数组，元素值为0
    classLabelVector=[]
    index=0
    #3解析文件数据到列表
    for line in arrayOLines:
        line=line.strip()#strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        #该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
        listFromLine=line.split('\t')#split()通过指定分隔符对字符串进行切片
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

datingDatMat,datingLabels=file2matrix('datingTestSet2.txt')

# =============================================================================
# print(datingDatMat)
# print(datingLabels[0:20])
# =============================================================================
# =============================================================================
# #创建散点图
# import matplotlib
# import matplotlib.pyplot as plt
# fig=plt.figure()#创建一个新的figure对象，创建一个新画布
# ax=fig.add_subplot(111)#将画布分为1行1列，area为从左往右从上往下第一块
# plt.scatter(datingDatMat[:,0],datingDatMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
# plt.xlabel('每年获得的飞行常客里程数',fontProperties='SimHei')#设置横坐标的名称，字体为黑体
# plt.ylabel('玩视频游戏所消耗时间百分比',fontProperties='SimHei')#设置纵坐标的名称，字体为黑体
# plt.legend(loc=0,ncol=3)
# plt.show()
# =============================================================================
    
#归一化特征值
def autoNorm(dataSet):
    minVals=dataSet.min(0)#使得函数可以从列中选取最小值
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
    
normMat,ranges,minVals=autoNorm(datingDatMat)
# =============================================================================
# print(normMat)
# print(ranges)
# print(minVals)
# =============================================================================
    
#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print('the classifier came back with:%d,the real answer is:%d'%(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount+=1.0
    print('the total error rate is: %f'%(errorCount/float(numTestVecs)))

#datingClassTest()   
    
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input('percentage of time spent playing video games?'))
    ffMiles=float(input('frequent flier miles earned per year?'))
    iceCream=float(input('liters of ice cream consumed per year?'))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('You will probably like this person:',resultList[classifierResult-1])
    
#classifyPerson()
    
    
    
#将图像格式化为向量
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])        
    return returnVect
 
testVector=img2vector('testDigits/0_13.txt')
# =============================================================================
# print(testVector)
# print(testVector[0,0:31])  
# print(testVector[0,31:62])  
# =============================================================================

#手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels=[]
    traingFileList=listdir('trainingDigits')#获取目录内容
    m=len(traingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        #从文件名解析分类数字
        fileNameStr=traingFileList[i]
        fileStr=fileNameStr.split('.')[0]#输出看一下
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s'%fileNameStr)
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s'%fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the classifier came back with:%d,the real answer is:%d'%(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            errorCount+=1.0
            print(fileNameStr)
    print('\nthe total number of error is:%d'%(errorCount))
    print('\nthe total error rate is:%f'%(errorCount/float(mTest)))

handwritingClassTest()



















