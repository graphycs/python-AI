#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Oct 19, 2010

@author: Peter
'''
import os
import chardet
from numpy import *
import feedparser
import nltk  
import nltk.data
from nltk.tokenize import WordPunctTokenizer

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
                 
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb) ) 
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print( testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb) )

def textParse(bigString):    #input is big string, #output is word list
    import re
    print('the context ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',bigString)
    listOfTokens = re.split(r'\W*', bigString)
    print('the listOfTokens ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',listOfTokens)
    wordList=[]
    for tok in listOfTokens:
		
        if len(tok) > 2 :
           print('the tok ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',tok)
           wordList.append(tok.lower())
	
    print('the word ^^^^^^^^^^^^^^^^^^^^',wordList)
    return wordList
    
def splitSentence(paragraph):  
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')  
    sentences = tokenizer.tokenize(paragraph)  
    return sentences  
 
 '''
	USE   nltk to split word into list
 '''   
def wordtokenizer(sentence):  
    #分段  
    words = WordPunctTokenizer().tokenize(sentence)  
    return words
    
    
def spamTest():
    import io  
    import sys   
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码 
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        filename = 'email/spam/%d.txt' % (i)
        print(filename)
        wordList = textParse(open(filename).read())
       
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        filename = 'email/ham/%d.txt' % (i)
        print(filename)
        filecontext=open(filename).read()
        wordList = textParse(filecontext)
    
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet =  list(range(50)); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print ("classification error",docList[docIndex])
    print( 'the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText
    
def getEncode(filename):
	bytes = min(32, os.path.getsize(filename))  
	raw = open(filename, 'rb').read(bytes)  
	result = chardet.detect(raw)  
	encoding = result['encoding'] 
	return encoding
	
	
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
   
    docList=[]; classList = []; fullText =[]
    print( 'feed1 entries length: ',len(feed1['entries']),'\nfeed0 entries length: ',len(feed0['entries']) )
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    print( minLen)
    for i in range(minLen):
        wordList = wordtokenizer(feed1['entries'][i]['summary'])
        print('wordList  feed1 ***********',wordList)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = wordtokenizer(feed0['entries'][i]['summary'])
        print('wordList  feed0 ***********',wordList)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    print('docList-------------',docList)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    print(vocabList,'**************',top30Words)
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = list( range(2*minLen) ); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet) )
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print( item[0] )
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print (item[0])
        
# ~ postingList,classVec=loadDataSet()
# ~ mywordlist=createVocabList(postingList)
# ~ print(mywordlist)
# ~ tranWord=setOfWords2Vec(mywordlist,postingList[0])
# ~ print(tranWord)
# ~ tranWord=setOfWords2Vec(mywordlist,postingList[3])
# ~ print(tranWord)

# ~ testingNB()

# ~ spamTest()
nf=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
sf=feedparser.parse('https://sports.yahoo.com/nba/rss.xml')

# ~ print(nf)
# ~ print(sf)

vocalist,psf,pnf=localWords(sf,nf)

