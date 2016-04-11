'''
Peter Menh
CSE 4334
Spring 2016
Programming Assignment 2

References used:
http://www.nltk.org/index.html
http://www.nltk.org/book/ch01.html

Code snippets from programming assignment 1
'''
import math
import time
import os
import operator
from math import log10, sqrt
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas
import numpy
import matplotlib.pyplot as plt

mytokenizer = RegexpTokenizer(r'[a-zA-Z0-9]{2,}')
stemmer = PorterStemmer()
sortedstopwords = sorted(stopwords.words('english'))
total_puid_att = []
total_puid_prod_des = []
dfs = {}
idfs = {}
total_word_counts = {}
attTermDict = {}
attPuidDict = {}
attVectorLength = 0

def tokenize(doc):
    tokens = mytokenizer.tokenize(doc)
    lowertokens = [token.lower() for token in tokens]
    filteredtokens = [stemmer.stem(token) for token in lowertokens if not token in sortedstopwords]
    return filteredtokens

def incdfs(tfvec):
    for token in set(tfvec):
        if token not in dfs:
            dfs[token]=1
            total_word_counts[token] = tfvec[token]
        else:
            dfs[token] += 1
            total_word_counts[token] += tfvec[token]
            

def getcount(token):
    if token in total_word_counts:
        return total_word_counts[token]
    else:
        return 0

def readfiles(corpus_root):
    for filename in os.listdir(corpus_root):
        f = open(os.path.join(corpus_root, filename), "r", encoding='UTF-8')
        doc = f.read()
        f.close() 
        doc = doc.lower()  
        tokens = tokenize(doc)
        tfvec = Counter(tokens)     
        attributesDict[filename] = tfvec
        incdfs(tfvec)
    
    ndoc = len(attributesDict)
    for token,df in dfs.items():
        idfs[token] = log10(ndoc/df)

def calctfidfvec(tfvec, withidf):
    tfidfvec = {}
    veclen = 0.0

    for token in tfvec:
        if withidf:
            tfidf = (1+log10(tfvec[token])) * getidf(token)
        else:
            tfidf = (1+log10(tfvec[token]))
        tfidfvec[token] = tfidf 
        veclen += pow(tfidf,2)

    if veclen > 0:
        for token in tfvec: 
            tfidfvec[token] /= sqrt(veclen)
    
    return tfidfvec
   
def cosinesim(vec1, vec2):
    commonterms = set(vec1).intersection(vec2)
    sim = 0.0
    for token in commonterms:
        sim += vec1[token]*vec2[token]
        
    return sim

def getqvec(qstring):
    tokens = tokenize(qstring)
    tfvec = Counter(tokens)
    qvec = calctfidfvec(tfvec, False)
    return qvec
    
def query(qstring):
    qvec = getqvec(qstring.lower())
    scores = {filename:cosinesim(qvec,tfidfvec) for filename, tfidfvec in speechvecs.items()}  
    return max(scores.items(), key=operator.itemgetter(1))[0]
    
def gettfidfvec(filename):
    return speechvecs[filename]
    
def getidf(token):
    if token not in idfs: 
        return 0
    else: 
        return idfs[token]
    
def docdocsim(filename1,filename2):
    return cosinesim(gettfidfvec(filename1),gettfidfvec(filename2))
    
def querydocsim(qstring,filename):
    return cosinesim(getqvec(qstring),gettfidfvec(filename))

def check_if_num(a):
   try:
       float(a)
   except ValueError:
       return False
   return True

#----------------Main------------------------------------------

#----------read attributes, calc tfidf--------------
start_time = time.time()

csvAtt = pandas.read_csv('attributes2.csv')

for i in range(0,len(csvAtt)):
    if csvAtt.product_uid[i] not in total_puid_att:
        total_puid_att.append(csvAtt.product_uid[i])

    if type(csvAtt.name[i])==float:
        if math.isnan(csvAtt.name[i]):
            nameTok = tokenize(" ")
    else:
        nameTok = tokenize(csvAtt.name[i])

    if type(csvAtt.value[i])==float:
        if math.isnan(csvAtt.value[i]):
            nameTok = tokenize(" ")
    else:
        valueTok = tokenize(csvAtt.value[i])

    for term in nameTok:
        if term not in attTermDict:
            attTermDict[term] = {'puids':{}, 'df':0}
            attTermDict[term]['puids'][csvAtt.product_uid[i]] = {'tf':1, 'tfidf':0, 'cosNormWt':0}
            attTermDict[term]['df'] = len(attTermDict[term]['puids'])
        else:
            if csvAtt.product_uid[i] not in attTermDict[term]['puids']:
                attTermDict[term]['puids'][csvAtt.product_uid[i]] = {'tf':1, 'tfidf':0, 'cosNormWt':0}
                attTermDict[term]['df'] = len(attTermDict[term]['puids'])
            else:
                attTermDict[term]['puids'][csvAtt.product_uid[i]]['tf'] +=1

print('Attribute time: ', time.time()-start_time)

#---------attribute tfidf/ cosine length normalized--------------
att_tf = 0
att_idf = 0
att_veclen = 0
attN = len(total_puid_att)
start_time = time.time()
for term in attTermDict:
    for p in attTermDict[term]['puids']:
        att_tf = 1 + log10(attTermDict[term]['puids'][p]['tf'])
        att_idf = log10(attN/attTermDict[term]['df'])
        attTermDict[term]['puids'][p]['tfidf'] = att_tf * att_idf
        


print('Att tfidf/cosine length time: ', time.time()-start_time)


#----------read product descriptions----------------
start_time = time.time()

#csvAtt = pandas.read_csv('product_descriptions2.csv')




print('Product Description time: ', time.time()-start_time)

