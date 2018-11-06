# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 14:29:17 2018

@author: Nalini
"""

import sys
import math
import collections
import os 
import re
import codecs

train=r'C:\Users\Dell\Downloads\hw2_train\train'
test =r'E:\DOWNLOADS\hw2_test\test'

if(len(sys.argv) != 3):#sys.argv[0] is the name of the file
    sys.exit("Please give valid Arguments- \n<path to TRAIN FOLDER that has both ham and spam folder> \
              \n<path to TEST FOLDER that has both ham and spam folder>")
else:
    train = sys.argv[1]
    test = sys.argv[2]

ham = list()
spam = list()
countTrainHam = 0 
countTrainSpam = 0 
dictProbHam = dict()
dictProbSpam = dict()
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

def HamVsSpam(classifier):
    if classifier == "ham":
        return (countTrainHam/(countTrainSpam + countTrainHam))
    else:
        return (countTrainSpam/(countTrainSpam + countTrainHam))

def getMissingWords(superSet,Subset):
    for words in superSet:
        if words not in Subset:
            Subset[words] = 0

def wordProbaility(classifier,stopfilter):
    Counter = 0

    if(stopfilter ==1):
            for word in stopWords:
                if word in dictHam:
                    del dictHam[word]
                if word in dictSpam:
                    del dictSpam[word]
                if word in dictBagOfWords:
                    del dictBagOfWords[word]
    if classifier == "ham":
        for word in dictHam:
            Counter += (dictHam[word] + 1)
        for word in dictHam:
            dictProbHam[word] = math.log((dictHam[word] + 1)/Counter ,2)
    elif classifier == "spam":
        for word in dictSpam:
            Counter += (dictSpam[word] + 1)
        for word in dictSpam:
            dictProbSpam[word] = math.log((dictSpam[word] + 1)/Counter ,2)        
 
def makePrediction(pathToFile, classifier):
    inaccurate = 0
    filecount = 0
    valueHam = 0 
    valueSpam = 0
    
    if classifier == "ham":
        for fileName in os.listdir(pathToFile):
            words =FileOpen(fileName,pathToFile)#[word.lower() for word in re.findall('[A-Za-z0-9\']+', fileHandler.read())] 
            valueHam = math.log(HamVsSpam("ham"),2)
            valueSpam = math.log(HamVsSpam("spam"),2)
            for word in words:
                if word in dictProbHam:
                    valueHam += dictProbHam[word]
                if word in dictProbSpam:
                    valueSpam += dictProbSpam[word]
            filecount +=1
            if(valueHam <= valueSpam):
                inaccurate+=1
    if classifier == "spam":
        for fileName in os.listdir(pathToFile):
            words =FileOpen(fileName,pathToFile)
            valueHam = math.log(HamVsSpam("ham"),2)
            valueSpam = math.log(HamVsSpam("spam"),2)
            for word in words:
                if word in dictProbHam:
                    valueHam += dictProbHam[word]
                if word in dictProbSpam:
                    valueSpam += dictProbSpam[word]
            filecount +=1
            if(valueHam >= valueSpam):
                inaccurate+=1
    return filecount,inaccurate  

def printComment(after, before,setName):
    if (before>after):
        print("Accuracy reduced on "+setName + " after removing stop words")
    if (before<after):
        print("Accuracy improved on "+setName + " after removing stop words")
        
#Executing NaiveBaiyes
        
#calculating the ham/spam words and file count in each folder           
ham,countTrainHam= browseDirectory(directoryHam)
spam,countTrainSpam = browseDirectory(directorySpam)

#getting ham/spam Distinct words and find their count in each folder
rawHam = dict(collections.Counter(w.lower() for w in ham))
dictHam = dict((k,int(v)) for k,v in rawHam.items())
rawSpam = dict(collections.Counter(w.lower() for w in spam))
dictSpam = dict((k,int(v)) for k,v in rawSpam.items())

#making bag of words for both ham and spam and further counting the count of each Distinct word in it
bagOfWords = ham + spam
dictBagOfWords = collections.Counter(bagOfWords)

#getting missing words in each Ham and Spam list and adding them and intializing their count= 0
getMissingWords(dictBagOfWords,dictHam)
getMissingWords(dictBagOfWords,dictSpam)
           
#caluculating probability for each word in ham and Spam folders 
wordProbaility("ham",0)
wordProbaility("spam",0) 

print("Calcualting Accuracy on Ham & Spam folders :")  
        
totalHam, incorrectHam = makePrediction(testHam, "ham")
accuracyHamD = round(((totalHam - incorrectHam )/(totalHam ))*100,2)
print("Accuracy on Ham :" + str(accuracyHamD) + "%")  

totalSpam,incorrectSpam = makePrediction(testSpam,"spam")
accuracySpamD = round(((totalSpam -  incorrectSpam )/(totalSpam))*100,2)
print("Accuracy on Spam : " + str(accuracySpamD) + "%") 


accuracyDefault = round(((totalHam + totalSpam - incorrectHam - incorrectSpam )/(totalHam + totalSpam))*100,2)
print("Total accuracy on Test :" + str(accuracyDefault) + "%")

print("\n")
print("Removing Stop Words from the bag of words...")
wordProbaility("ham",1)
wordProbaility("spam",1) 

totalHam, incorrectHam = makePrediction(testHam, "ham")
accuracyHam = round(((totalHam - incorrectHam )/(totalHam ))*100,2)
print("Accuracy on Ham :" + str(accuracyHam) + "%")  

totalSpam,incorrectSpam = makePrediction(testSpam,"spam")
accuracySpam = round(((totalSpam -  incorrectSpam )/(totalSpam))*100,2)
print("Accuracy on Spam : " + str(accuracySpam) + "%") 


accuracy = round(((totalHam + totalSpam - incorrectHam - incorrectSpam )/(totalHam + totalSpam))*100,2)
print("Total accuracy on Test :" + str(accuracy) + "%")

print("\nConclusion")
printComment(accuracyHam,accuracyHamD,"Ham")
printComment(accuracySpam,accuracySpamD,"Spam")
printComment(accuracy,accuracyDefault,"overall data")

