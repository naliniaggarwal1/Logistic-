# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:24:19 2018

@author: Nalini
"""

import sys
import collections
import os 
import re
import codecs
import numpy

train=r'C:\Users\laksh\Downloads\ML\Assignment 2\Submission\hw2_train\train'
test =r'C:\Users\laksh\Downloads\ML\Assignment 2\Submission\hw2_test\test'

if(len(sys.argv) != 6):#sys.argv[0] is the name of the file
    sys.exit("Please give valid Arguments- \n<path to TRAIN FOLDER that has both ham and spam folder> \
              \n<path to TEST FOLDER that has both ham and spam folder>\
              \n<yes or no to remove stop words\
              \n<Regularization parameters>\
              \n<iteration>")
else:
    train = sys.argv[1]
    test = sys.argv[2]
    StopWords = sys.argv[3]
    Lamda = float(sys.argv[4])
    Iteration = sys.argv[5]

ham = list()
spam = list()
countTrainHam = 0 
countTrainSpam = 0 
dictProbHam = dict()
dictProbSpam = dict()
learningRate = 0.001
regularization = Lamda
stopWords = ["a","about","above","after","again","against","all","am","an","and",
"any","are","aren't","as","at","be","because","been","before","being","below",
"between","both","but","by","can't","cannot","could","couldn't","did","didn't",
"do","does","doesn't","doing","don't","down","during","each","few","for","from",
"further","had","hadn't","has","hasn't","have","haven't","having","he","he'd",
"he'll","he's","her","here","here's","hers","herself","him","himself","his","how",
"how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its",
"itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of",
"off","on","once","only","or","other","ought","our","ours","ourselves","out","over",
"own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some",
"such","than","that","that's","the","their","theirs","them","themselves","then","there",
"there's","these","they","they'd","they'll","they're","they've","this","those","through",
"to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've",
"were","weren't","what","what's","when","when's","where","where's","which","while","who",
"who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll",
"you're","you've","your","yours","yourself","yourselves"]
bias = 0 
xnode = 1
#location of the folder for ham & spam for train and test 
directoryHam = train + '\ham'
directorySpam =train + '\spam'
testHam = test + '\ham'
testSpam =test + '\spam'

#Regualar expression to clean the data given in train ham and spam folder 
regex = re.compile(r'[A-Za-z0-9\']')

def FileOpen(filename,path):
    fileHandler = codecs.open(path+"\\" + filename,'rU','latin-1') # codecs handles -> UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 1651: character maps to <undefined>
    words = [Findwords.lower() for Findwords in re.findall('[A-Za-z0-9\']+', fileHandler.read())]
    fileHandler.close()
    return words

def browseDirectory(path):
    wordList = list()
    fileCount = 0
    for files in os.listdir(path):
        if files.endswith(".txt"):
            wordList +=FileOpen(files,path)
            fileCount+=1
    return wordList, fileCount
#iterating throvugh  Ham train to get the list of ham words used to form combined bag of words 
ham,countTrainHam= browseDirectory(directoryHam)
spam,countTrainSpam = browseDirectory(directorySpam)

##########FOR TEST DATA################

hamTest, countTestHam = browseDirectory(testHam)
SpamTest, countTestSpam = browseDirectory(testSpam)

def removeStopWords():
    for word in stopWords:
        if word in ham:
            ham.remove(word)
        if word in spam:
            spam.remove(word)
        if word in hamTest:
            hamTest.remove(word)
        if word in SpamTest:
            SpamTest.remove(word)
if(sys.argv[3] == "yes"):
    removeStopWords()   
         
rawHam = dict(collections.Counter(w.lower() for w in ham))
dictHam = dict((k,int(v)) for k,v in rawHam.items())
rawSpam = dict(collections.Counter(w.lower() for w in spam))
dictSpam = dict((k,int(v)) for k,v in rawSpam.items())

bagOfWords = ham + spam
dictBagOfWords = collections.Counter(bagOfWords)
listBagOfWords = list(dictBagOfWords.keys())
TargetList = list() #final value of ham or spam, ham = 1 & spam = 0
totalFiles = countTrainHam + countTrainSpam


####################TEST####################################
rawTestHam = dict(collections.Counter(w.lower() for w in hamTest))
dictTestHam = dict((k,int(v)) for k,v in rawTestHam.items())
rawTestSpam = dict(collections.Counter(w.lower() for w in SpamTest))
dictTestSpam = dict((k,int(v)) for k,v in rawTestSpam.items())


testBagOfWords = ham + spam
testDictBagOfWords = collections.Counter(testBagOfWords)
testListBagOfWords = list(testDictBagOfWords.keys())
testTargetList = list() #final value of ham or spam, ham = 1 & spam = 0
totalTestFiles = countTestHam + countTestSpam

#initialize matrix to zero


def initiliazeMatrix(row, column):
    featureMatrix = [0] * row
    for i in range(row):
        featureMatrix[i] = [0] * column
    return featureMatrix

trainFeatureMatrix = initiliazeMatrix(totalFiles,len(listBagOfWords) )
testFeatureMatrix = initiliazeMatrix(totalTestFiles, len(testListBagOfWords))


rowMatrix = 0
testRowMatrix = 0

sigMoidList=list()    #for each row
for i in range(totalFiles):
    sigMoidList.append(-1)
    TargetList.append(-1)
    
for i in range(totalTestFiles):
    testTargetList.append(-1)
weightOfFeature = list()
for feature in range(len(listBagOfWords)):
    weightOfFeature.append(0) #initializinf weight = 0

def makeMatrix(featureMatrix,path,listBagOfWords,rowMatrix,classifier,TargetList):
    for fileName in os.listdir(path):
        words = FileOpen(fileName,path)
        temp = dict(collections.Counter(words))
    #    print (temp)
        for key in temp:
            if key in listBagOfWords:
                column = listBagOfWords.index(key)
                featureMatrix[rowMatrix][column] = temp[key]
    #            print(str(a[rowMatrix][column]) + str(temp[key]))
        
        if(classifier == "ham"):
            TargetList[rowMatrix] =0
        elif(classifier == "spam"):
            TargetList[rowMatrix] = 1
        rowMatrix +=1
    return featureMatrix,rowMatrix,TargetList

#train matrix including ham and spam
trainFeatureMatrix,rowMatrix,TargetList= makeMatrix(trainFeatureMatrix,directoryHam,listBagOfWords,rowMatrix,"ham",TargetList)
trainFeatureMatrix,rowMatrix,TargetList= makeMatrix(trainFeatureMatrix,directorySpam,listBagOfWords,rowMatrix,"spam",TargetList)
        
testFeatureMatrix,testRowMatrix,testTargetList= makeMatrix(testFeatureMatrix,testHam,testListBagOfWords,testRowMatrix,"ham",testTargetList)
testFeatureMatrix,testRowMatrix,testTargetList= makeMatrix(testFeatureMatrix,testSpam,testListBagOfWords,testRowMatrix,"spam",testTargetList)

 # for each column
def sigmoid(x):
    den= (1 + numpy.exp(-x))
    sigma = 1/den
    return sigma

#Calculate for each file 
def sigmoidFunction(totalFiles,totalFeatures,featureMatrix):
    global sigMoidList
    for files in range(totalFiles):
        summation = 1.0
        
        for features in range(totalFeatures):
            summation +=featureMatrix[files][features] * weightOfFeature[features]
        sigMoidList[files] = sigmoid(summation)
#        return sigMoidList
    

def calculateWeightUpdate(totalFiles,numberOfFeature,featureMatrix,TargetList):
    global sigMoidList
    
    for feature in range(numberOfFeature):
        weight = bias# xnode =1
        for files in range(totalFiles):
            frequency = featureMatrix[files][feature]
            y = TargetList[files]
            sigmoidValue = sigMoidList[files]
            weight += frequency * (y - sigmoidValue)
        
        oldW = weightOfFeature[feature]
        weightOfFeature[feature] += ((weight * learningRate) - (learningRate * regularization * oldW ) )
    #print(weightOfFeature[0:6])
    return weightOfFeature

def trainingFunction(totalFiles, numbeOffeatures,trainFeatureMatrix,TargetList):
    #print("Iteration part a for " + str(i))
    sigmoidFunction(totalFiles, numbeOffeatures,trainFeatureMatrix)
    #print("Iteration part b for " + str(i))
    calculateWeightUpdate(totalFiles, numbeOffeatures,trainFeatureMatrix,TargetList)
     

def classifyData():    
    correctHam=0
    incorrectHam=0
    correctSpam=0
    incorrectSpam=0
    
    for file in range(totalTestFiles):
        summation = 1.0
        for i in range(len(testListBagOfWords)):
            word= testListBagOfWords[i]
            
            if word in listBagOfWords:
                index= listBagOfWords.index(word)
                weight= weightOfFeature[index]
                wordcount= testFeatureMatrix[file][i]
                
                summation+= weight*wordcount
        
        sigSum = sigmoid(summation)
        if(testTargetList[file]==0):
            if sigSum<0.5:
                correctHam+=1.0
            else:
                incorrectHam+=1.0
        else:
            if sigSum>=0.5:
                correctSpam +=1.0
            else:
                incorrectSpam+=1.0
    
    print("Accuracy on Ham:"+str( (correctHam /(correctHam+incorrectHam))*100))

    print("Accuracy on Spam:"+str((correctSpam/(correctSpam+incorrectSpam))*100))

print("Training the algorithm - ")
for i in range(int(Iteration)):
    print(i, end = ' ')
    trainingFunction(totalFiles, len(listBagOfWords),trainFeatureMatrix,TargetList)
    
#testHam = r'E:\DOWNLOADS\hw2_test\test\ham'
#testSpam =r'E:\DOWNLOADS\hw2_test\test\spam'    
print("Training completed successfully")
print("\nPlease wait while classifying the data..\nIt may take few minutes")
classifyData()


