'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# ~ set(可变集合)与frozenset(不可变集合)的区别：
# ~ set无序排序且不重复，是可变的，有add（），remove（）等方法。既然是可变的，所以它不存在哈希值。基本功能包括关系测试和消除重复元素. 集合对象还支持union(联合), intersection(交集), difference(差集)和sysmmetric difference(对称差集)等数学运算. 
# ~ sets 支持 x in set, len(set),和 for x in set。作为一个无序的集合，sets不记录元素位置或者插入点。因此，sets不支持 indexing, 或其它类序列的操作。
# ~ frozenset是冻结的集合，它是不可变的，存在哈希值，好处是它可以作为字典的key，也可以作为其它集合的元素。缺点是一旦创建便不能更改，没有add，remove方法。
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    return list(map(frozenset, C1))#use frozen set so we
                            #can use it as a key in a dict    


#
def scanD(D,Ck,minSupport):
    """
    函数说明：该函数用于从大小为1的所有候选项集的集合C1生成频繁项集列表L1，即retList。
    数据集:D
    候选项集列表:Ck
    最小支持度:minSupport
    返回值： 频繁项集列表：retList
            包含支持度值的字典：supportData
    """
    print("------------D------",D)
    print("------------Ck------",Ck)
    ssCnt={}
    for tid in D:                                #遍历数据集
        print("------------tid------",tid)
        for can in Ck:                           #遍历候选项
            print("can-----",can)
            if can.issubset(tid):                #判断候选项中是否含数据集的各项
                #if not ssCnt.has_key(can): # python3 can not support
                print( "not can in ssCnt--------------",(can in ssCnt))
                if not can in ssCnt:
                    ssCnt[can]=1                 #不含设为1
                else: ssCnt[can]+=1              #有则计数加1
    numItems=float(len(D))                       #数据集大小
    retList = []                                 #L1初始化
    supportData = {}                             #记录候选项中各个数据的支持度
    print("ssCnt-------",ssCnt)
    for key in ssCnt:
        support = ssCnt[key]/numItems            #计算支持度
        print("the key--------------",key,"-----------support is ----------",support)
        if support >= minSupport:
            retList.insert(0,key)                #满足条件加入L1中
        supportData[key] = support
    return retList, supportData
"""
    函数说明：通过输入参数Lk、k, 输出候选项集Ck
    频繁项集列表： Lk
    项集元素个数： k
"""
def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2] #关于k-2的疑惑书上解释得很清楚，为了避免重复操作
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
            
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print (itemMeaning[item])
        print ("           -------->")
        for item in ruleTup[1]:
            print (itemMeaning[item])
        print ("confidence: %f" % ruleTup[2])
        print ()      #print a blank line
        
"""           
from time import sleep
from votesmart import votesmart
votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
#votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) #api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print ('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print ("problem getting bill %d" % billNum)
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList
        
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print ('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print ("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning
"""

dataSet=loadDataSet()
C1=createC1(dataSet)
print('C1=',list(C1))
D=list(map(set,dataSet))
print('D=',D)
L1,suppData0=scanD(D,C1,0.5)
print('L1=',L1)
print('suppData0=',suppData0)



L2, supportData2=apriori(dataSet)
print('L2=',L1)
print('suppData2=',suppData0)


L3,suppData3=apriori(dataSet,minSupport=0.5)
print('L=',L3)
print('suppData=',suppData3)
#rules=generateRules(L,suppData,minConf=0.7)
#print('rules=',rules)
print('##########################################################')
rules=generateRules(L3,suppData3,minConf=0.5)
print('rules=',rules)
