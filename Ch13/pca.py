'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    #使用两个list来构建矩阵
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

# ~ 在numpy.linalg模块中：
# ~ eigvals() 计算矩阵的特征值
# ~ eig() 返回包含特征值和对应特征向量的元组
# ~ 函数中用numpy中的mean方法来求均值，axis=0表示按列求均值。
# ~ 该函数返回两个变量，newData是零均值化后的数据，meanVal是每个特征的均值，是给后面重构数据用的。

def pca(dataMat, topNfeat=9999999):  #topNfeat为可选参数，记录特征值个数
    meanVals = mean(dataMat, axis=0)  #求均值
    meanRemoved = dataMat - meanVals #remove mean #归一化数据
    covMat = cov(meanRemoved, rowvar=0)  #求协方差
    eigVals,eigVects = linalg.eig(mat(covMat)) #计算特征值和特征向量
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest #对特征值进行排序，默认从小到大
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    #将数据转换到新空间
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat
    
    
dataMat = loadDataSet("testSet.txt")
lowDMat, reconMat = pca(dataMat,1)
print( shape(lowDMat) )

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^',  s = 90 )
ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0],marker='o', s = 50 , c ='red' )
plt.show()
