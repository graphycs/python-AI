'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        # 将每行映射成浮点数，python3中map()返回值改变，所以需要修改源代码
        # ~ fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
	"""
	函数说明： 数据切分函数（切分数据集为两个子集:左子集和右子集）
	dataSet: 数据集 
	feature: 待切分特征 
	value: 特征值
	"""
	mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
	mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :] 
	#nonzero()的返回值是一个数组array([x,..., x], dtype=int64)，其中第一项是索引值列表,（）应用数组过滤
	#[[x,..., x],:]将每个索引值所对应的行进行全部复制
	"""
    下面的原书代码报错 index 0 is out of bounds,使用上面两行代码
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
	"""
	return mat0,mat1

def regLeaf(dataSet):#returns the value used for each leaf
	"""
    函数说明： 当chooseBestSplit()确定不再对数据进行切分时，调用regLeaf()生成叶结点
    返回值： 叶结点（在回归树中,其实就是目标变量的均值）
    """
	return mean(dataSet[:,-1])

"""
函数说明：平方误差函数
返回值：平方误差
"""
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    """
    函数说明：回归树切分函数（找到数据的最佳二元切分方式）
    返回值：若找不到好的切分方式，本函数在3种情况下不会切分，直接创建叶结点；
        若找到好的切分方式，返回最好的切分的特征编号和切分特征值
    """
    tolS = ops[0]                          #允许的误差下降值，用户指定参数，用于控制函数的停止时机
    tolN = ops[1]                          #切分的最小样本数，用户指定参数，用于控制函数的停止时机
    #if all the target variables are the same value: quit and return value
    #tolS,tolN进行的实际上是一种预剪枝处理
    #################################若所有值都相同，则退出##############################
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1  用set()对当前所有目标变量建立一个集合,不含相同项，用len()统计不同剩余特征值数目，若为1，则无需切分
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    #该误差S将用于与新切分的误差进行对比，来检查新切分能否降低误差
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        # for splitVal in set(dataSet[:,featIndex]): python3报错修改为下面
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):  #遍历每个特征里不同的特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal) #对每个特征进行二元切分
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:  #更新为误差最小的特征
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:   #如果切分后误差效果下降不大，则取消切分，直接创建叶结点
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3 #另外，检查切分后子集大小，若小于最小允许样本数tolN，则停止切分
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split  #返回特征编号和用于切分的特征值

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    """
    函数说明： 树构建函数
    dataSet： 数据集
    leafType： 建立叶节点函数
    errType： 误差计算函数
    ops： 包含树构建所需的其他参数的元组
    返回值： 构建好的树retTree
    """
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split #满足停止条件时返回叶结点值，见chooseBestSplit()
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val) #切分数据集
    retTree['left'] = createTree(lSet, leafType, errType, ops) #建立左子树
    retTree['right'] = createTree(rSet, leafType, errType, ops)  #建立右子树
    return retTree  

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    """
    函数说明：从上往下遍历树直到找到叶节点为止，若找到两个叶节点，计算他们的平均值
    返回值：树的平均值（对树进行塌陷式处理）
    """
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):
    """
    函数说明：树的后剪枝
    tree：待剪枝的树 
    testData：剪枝所需的测试数据
    返回值：剪枝后的树tree
    """
    print("testData*****************",testData)
    if shape(testData)[0] == 0: 
        return getMean(tree) #if we have no test data collapse the tree #没有测试数据则对树进行塌陷式处理
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them  #左右子树有一个非空
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']): #剪枝后判断是否还是有子树
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))  #计算合并前的错误率
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))  #计算合并后的错误率
        if errorMerge < errorNoMerge:  #如果合并后误差变小
            print("merging")
            return treeMean
        else: return tree
    else: return tree
    
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat



# ~ 对于X[:,0];
# ~ 是取二维数组中第一维的所有数据
# ~ 对于X[:,1]
# ~ 是取二维数组中第二维的所有数据
# ~ 对于X[:,m:n]
# ~ 是取二维数组中第m维到第n-1维的所有数据
# ~ 对于X[:,:,0]
# ~ 是取三维矩阵中第一维的所有数据
# ~ 对于X[:,:,1]
# ~ 是取三维矩阵中第二维的所有数据
# ~ 对于X[:,:,m:n]
# ~ 是取三维矩阵中第m维到第n-1维的所有数据

# ~ data_list=[[1,2,3],[1,2,1],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[6,7,9],[0,4,7],[4,6,0],[2,9,1],[5,8,7],[9,7,8],[3,7,9]]
    # ~ # data_list.toarray()
# ~ dMat=mat(data_list)
# ~ print(dMat[:,-1])

"""
myDat= loadDataSet("ex00.txt")
# ~ print("*************mydat****************")
# ~ print(myDat)
myMat=mat(myDat)

# ~ print("*************myMat****************")
# ~ print(myMat)

myTree=createTree(myMat)
print("*************myTree****************")
print(myTree)
"""
"""
myDat= loadDataSet("ex0.txt")
# ~ print("*************mydat****************")
# ~ print(myDat)
myMat=mat(myDat)

# ~ print("*************myMat****************")
# ~ print(myMat)

myTree=createTree(myMat)
print("*************myTree****************")
print(myTree)
"""


""" 比较树
myTree=createTree(myMat,ops=(0,1))
print("*************myTree****************")
print(myTree)
"""
"""

myDat2= loadDataSet("ex2.txt")
myMat2=mat(myDat2)
myTree2=createTree(myMat2,ops=(0,1))
print("*************myTree2****************")
print(myTree2)


myDat2Test= loadDataSet("ex2test.txt")
myMat2Test=mat(myDat2Test)
myTree2Test=createTree(myMat2Test)
print("*************myTree2Test****************")
print(myTree2Test)


myTreeCompare=prune(myTree2,myMat2Test)
print("*************myTreeCompare****************")
print(myTreeCompare)
"""
trainMat= mat(loadDataSet("bikeSpeedVsIq_train.txt"))
trainTestMat= mat(loadDataSet("bikeSpeedVsIq_test.txt"))
# ~ fig=plt.figure()
# ~ ax=fig.add_subplot(111)
# ~ ax.scatter(trainTestMat[:,0].A.reshape(1,-1),trainTestMat[:,1].A.reshape(1,-1), c = 'blue')
# ~ plt.xlabel('speed of riding')
# ~ plt.ylabel('IQ')
# ~ plt.show()
    
    
trainTree=createTree(trainMat,ops=(1,20))
print('trainTree***********',trainTree)
yHat=createForeCast(trainTree,trainTestMat[:,0])
print('yHat***********',yHat)
print('trainTestMat**********',trainTestMat[:,1])

print("corrcoef**************",corrcoef(yHat,trainTestMat[:,1],rowvar=0)[0,1])


trainTree2=createTree(trainMat,modelLeaf,modelErr,ops=(1,20))
print('trainTree***********',trainTree2)
yHat2=createForeCast(trainTree2,trainTestMat[:,0],modelTreeEval)
print('yHat2***********',yHat2)

print("corrcoef2**************",corrcoef(yHat2,trainTestMat[:,1],rowvar=0)[0,1])


ws,x,y=linearSolve(trainTestMat)
print("ws***************",ws)
