'''
Created on Jun 14, 2011
FP-Growth FP means frequent pattern
the FP-Growth algorithm needs: 
1. FP-tree (class treeNode)
2. header table (use dict)

This finds frequent itemsets similar to apriori but does not 
find association rules.  

@author: Peter
'''
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode      #needs to be updated
        self.children = {} 
    
    def inc(self, numOccur):
        self.count += numOccur
        
    def disp(self, ind=1):
        print( ind,'  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

def createTree(dataSet, minSup=1): #create FP-tree from dataset but don't mine
    """
    函数说明：FP-tree构建函数
    输入参数：数据集 dataSet
             (此处的数据集dataSet经过处理的，它是一个字典，键是每个样本，值是这个样本出现的频数)
             最小支持度 minSup
    返回值：构建好的树retTree
            头指针表headerTable
    """
    headerTable = {}  #初始化空字典作为一个头指针表
    #go over dataSet twice
    for trans in dataSet:#first pass counts frequency of occurance ##############第一次遍历数据集：统计每个元素项出现的频度###############
        print("this is trans value",trans)
        for item in trans:
            print("this is trans item",item)
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
            print("this is trans headerTable[",item,"]",headerTable[item])
    for k in list(headerTable):   
    # ~ for k in headerTable.keys():  #remove items not meeting minSup #######删除未达到最小频度的元素########
	#此处headerTable要取list，因为字典要进行删除del操作，字典在迭代过程中长度发生变化是会报错的
        if headerTable[k] < minSup: 
            print("will del key ",k," and the value",headerTable[k])
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    #print 'freqItemSet: ',freqItemSet
    if len(freqItemSet) == 0: return None, None  #if no items meet min support -->get out #若达到要求的数目为0，返回
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] #reformat headerTable to use Node link 
    #print 'headerTable: ',headerTable
    retTree = treeNode('Null Set', 1, None) #create tree #创建只包含空集的根节点
    for tranSet, count in dataSet.items():  #go through dataset 2nd time
        localD = {}
        for item in tranSet:  #put transaction items in order #通过for循环，把每个样本中频繁项集及其频数储存在localD字典中
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            #items()方法返回一个可迭代的dict_items类型，其元素是键值对组成的2-元组
            #sorted(排序对象，key，reverse),当待排序列表的元素由多字段构成时，
            #我们可以通过sorted(iterable，key，reverse)的参数key来制定我们根据哪个字段对列表元素进行排序 
            #这里key=lambda p: p[1]指明要根据键对应的值，即根据频繁项的频数进行从大到小排序 
            updateTree(orderedItems, retTree, headerTable, count)#populate tree with ordered freq itemset #使用排序后的频繁项集对树进行填充
    return retTree, headerTable #return tree and header table

def updateTree(items, inTree, headerTable, count):
	
    """
    函数说明：让FP树生长
    """
    if items[0] in inTree.children:#check if orderedItems[0] in retTree.children 
		#首先检查事物项items中第一个元素是否作为子节点存在#如果树中存在现有元素  #则更新增加该元素项的计数，
        inTree.children[items[0]].inc(count) #incrament count
    else:   #add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree) #创建一个新的树节点，并更新了父节点inTree，父节点是一个类对象，包含很多特性
        if headerTable[items[0]][1] == None: #update header table  #若原来指向每种类型第一个元素项的指针为 None，则需要更新头指针列表
            headerTable[items[0]][1] = inTree.children[items[0]]  #更新头指针表，把指向每种类型第一个元素项放在头指针表里
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]]) #更新生成链表，注意，链表也是每过一个样本，更一次链表，且链表更新都是从头指针表开始的
    if len(items) > 1:#call updateTree() with remaining ordered items #仍有未分配完的树，迭代
        updateTree(items[1::], inTree.children[items[0]], headerTable, count) #由items[1::]可知，每次调用updateTree时都会去掉列表中第一个元素，递归
        
def updateHeader(nodeToTest, targetNode):   #this version does not use recursion 函数说明：它确保节点链接指向树中该元素项的每一个实例
    while (nodeToTest.nodeLink != None):    #Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink #从头指针表的 nodeLink 开始,一直沿着nodeLink直到到达链表末尾，这就是一个链表
    nodeToTest.nodeLink = targetNode #链表链接的都是相似元素项，通过ondeLink 变量用来链接相似的元素项。
        
def ascendTree(leafNode, prefixPath): #ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
    
def findPrefixPath(basePat, treeNode): #treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]#(sort header table)
    for basePat in bigL:  #start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #print 'finalFrequent Item: ',newFreqSet    #append to set
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #print 'condPattBases :',basePat, condPattBases
        #2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        #print 'head from conditional tree: ', myHead
        if myHead != None: #3. mine cond. FP-tree
            #print 'conditional tree for: ',newFreqSet
            #myCondTree.disp(1)            
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict
"""

import twitter
from time import sleep
import re

def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)    
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY, 
                      access_token_secret=ACCESS_TOKEN_SECRET)
    #you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1,15):
        print( "fetching page %d" % i)
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages

def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList
"""
#minSup = 3
#simpDat = loadSimpDat()
#initSet = createInitSet(simpDat)
#myFPtree, myHeaderTab = createTree(initSet, minSup)
#myFPtree.disp()
#myFreqList = []
#mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)



rootNode=treeNode('pyramid',9,None)
rootNode.children['eye']=treeNode('eye',13,None)
print('rootNode.disp()=',rootNode.disp())
rootNode.children['phoenix']=treeNode('phoenix',3,None) 
print('rootNode.disp()=',rootNode.disp())


simpDat=loadSimpDat()
print('simpDat=',simpDat)
initSet=createInitSet(simpDat)
print('initSet=',initSet)
 
myFPtree,myHeaderTab=createTree(initSet,3)
myFPtree.disp()
print('myFPtree=',myFPtree)    #myFPtree是一个类对象
print('myHeaderTab=',myHeaderTab)
print('x的条件模式基:',findPrefixPath('x', myHeaderTab['x'][1]))
print('z的条件模式基:',findPrefixPath('z', myHeaderTab['z'][1]))
print('r的条件模式基:',findPrefixPath('r', myHeaderTab['r'][1]))
print('t的条件模式基:',findPrefixPath('t', myHeaderTab['t'][1]))
    
