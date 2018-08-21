'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list( map(float,curLine) )#map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)
    
    
"""
        numpy中有一些常用的用来产生随机数的函数，randn()和rand()就属于这其中。 
        numpy.random.randn(d0, d1, …, dn)是从标准正态分布中返回一个或多个样本值。 
        numpy.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中。
"""
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
        #random.rand(k,1)会生成一个k行1列值在[0，1)间的array
        #保证随机质心的生成在min和max之间，不能超过整个数据的边界
    return centroids
    
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
                                      #创建一个矩阵clusterAssment来储存每个点的簇分配结果，第一列记录索引值，第二列储存误差
                                      #这里的误差是指当前点到簇质心，该误差用来评价聚类的效果
    centroids = createCent(dataSet, k)
    clusterChanged = True
    ###############################更新质心的大循环#############################
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid  #遍历m个样本点
            minDist = inf; minIndex = -1
            for j in range(k): #遍历k个质心，找到最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:]) #计算样本点i和质点j的欧氏距离
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True #此处判断簇分配是否变化  #只要有一个点变化就重设为True，再次迭代
            clusterAssment[i,:] = minIndex,minDist**2  #更新clusterAssmen
        print (centroids)
        for cent in range(k):#recalculate centroids   #遍历k个簇
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
            #clusterAssment[:,0].A==cent是数组过滤，来获得给定簇中所有点
            #clusterAssment的第一列储存的k个簇的编号，所以通过数组过滤将当前编号的簇中数据筛选出来
            #nonzero会将筛选出来的数据在clusterAssment中的索引值进行返回，而返回的第一项就是索引值列表
            #dataSet通过索引将所有当前编号簇中样本点取出，储存在ptsInClust中
    return centroids, clusterAssment
#@distMeas:距离计算方法，默认欧氏距离distEclud()
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid #将所有点看成一个簇，计算质心
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf  #init SSE
        ###########for循环会筛选出当前簇中最优分割的簇##########
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  # k=2,kMeans
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
				#这一轮的sseSplit会作为下一轮的sseNotSplit出现，这样的话就筛选出使sse减小最大的分割族
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        ########################################################
        ########################修正分割簇后结果，为下一轮迭代做准备###########################
        """
        #一开始，我们的簇编号为0，经过第一次分割后簇个数变为2，
        #事实上，经过上述for循环我们选定了最优化分簇，由于上述过程中k=2，所以选定的最优分割簇会被分成两部分
        #还应注意到，我们只分割选定的最优分割簇，原来的所有的簇的编号只有选定的最优分割簇发生变化，另外，
        #且分割后会多出一个簇，下面我们将分割后变化和多出的簇进行修正后，继续下一轮迭代
        #然后我们将分割后簇编号为1的簇的编号修改为当前簇个数，这是由于python从0开始计数的，
        #下一步，将分割后簇编号是0的簇的编号修改为最优分割簇，关于关于这一步，需要注意如下：
        #千万不要理解为分割后簇编号是0的簇将成为最优分割簇，上面已经解释的很详细
        """
                
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print ('the bestCentToSplit is: ',bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment

import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print( yahooApi)
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print ("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print ("error fetching")
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()



"""    

dataMat=mat(loadDataSet('testSet.txt'))
print('第一列最小值：',min(dataMat[:,0]))
print('第二列最小值：',min(dataMat[:,1]))
print('第一列最大值：',max(dataMat[:,0]))
print('第二列最大值：',max(dataMat[:,1]))
print('质心矩阵:',randCent(dataMat, 2))
print('欧氏距离：',distEclud(dataMat[0],dataMat[1]))

datMat=mat(loadDataSet('testSet.txt'))
centroids,clusterAssment= kMeans(datMat,4)
print('centroids=',centroids)
########################散点图######################
ptsInClust=datMat[nonzero(clusterAssment[:,0].A==0)[0]]
plt.scatter(ptsInClust[:,0].flatten().A[0],ptsInClust[:,1].flatten().A[0],c='blue',marker='o')
ptsInClust=datMat[nonzero(clusterAssment[:,0].A==1)[0]]
plt.scatter(ptsInClust[:,0].flatten().A[0],ptsInClust[:,1].flatten().A[0],c='green',marker='^')
ptsInClust=datMat[nonzero(clusterAssment[:,0].A==2)[0]]
plt.scatter(ptsInClust[:,0].flatten().A[0],ptsInClust[:,1].flatten().A[0],c='red',marker='s')
ptsInClust=datMat[nonzero(clusterAssment[:,0].A==3)[0]]
plt.scatter(ptsInClust[:,0].flatten().A[0],ptsInClust[:,1].flatten().A[0],c='orange',marker='*') 
for i, centroid in enumerate(centroids):#画质心
	x,y = centroid.A[0] #或者x,y = centroids.A[i]也可以做到
	plt.scatter(x, y, s=150, c='black', alpha=0.7, marker='+')
plt.show()

"""    
    
    
datMat3=mat(loadDataSet('testSet2.txt'))
centList,myNewAssment=biKmeans(datMat3,3)
print('centList=',centList)
########################散点图########################
ptsInClust=datMat3[nonzero(myNewAssment[:,0].A==0)[0]]
plt.scatter(ptsInClust[:,0].flatten().A[0],ptsInClust[:,1].flatten().A[0],c='blue',marker='o')
ptsInClust=datMat3[nonzero(myNewAssment[:,0].A==1)[0]]
plt.scatter(ptsInClust[:,0].flatten().A[0],ptsInClust[:,1].flatten().A[0],c='green',marker='^')
ptsInClust=datMat3[nonzero(myNewAssment[:,0].A==2)[0]]
plt.scatter(ptsInClust[:,0].flatten().A[0],ptsInClust[:,1].flatten().A[0],c='red',marker='s') 
for i, centroid in enumerate(centList):#画质心
    x,y = centroid.A[0]                     #或者x,y = centroids.A[i]也可以做到
    plt.scatter(x, y, s=150, c='black', alpha=0.7, marker='+')
plt.show()
######################################################

