'''
Peter Menh
CSE 4334
Spring 2016
Programming Assignment 2

References used:
http://www.nltk.org/index.html
http://www.nltk.org/book/ch01.html

Code snippets from programming assignment 1 solution
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
total_puid_prod_des = []
dfs = {}
idfs = {}
total_word_counts = {}
attTermDict = {}
attPuidDict = {}
prodDesTermDict = {}
prodDesPuidDict = {}
attVectorLength = 0
prodDesVectorLength = 0

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
    if csvAtt.product_uid[i] not in attPuidDict:
        attPuidDict[csvAtt.product_uid[i]] = {'tfidf_list':[], 'veclen':0}

    if type(csvAtt.name[i])==float:
        if math.isnan(csvAtt.name[i]):
            nameTok = tokenize(" ")
    else:
        nameTok = tokenize(csvAtt.name[i])

    if type(csvAtt.value[i])==float:
        if math.isnan(csvAtt.value[i]):
            valueTok = tokenize(" ")
    else:
        valueTok = tokenize(csvAtt.value[i])

    for term in nameTok:
        if term not in attTermDict:
            attTermDict[term] = {'puids':{}, 'df':0, 'idf':0}
            attTermDict[term]['puids'][csvAtt.product_uid[i]] = {'tf':1, 'tfidf':0, 'cosNormWt':0}
            attTermDict[term]['df'] = len(attTermDict[term]['puids'])
        else:
            if csvAtt.product_uid[i] not in attTermDict[term]['puids']:
                attTermDict[term]['puids'][csvAtt.product_uid[i]] = {'tf':1, 'tfidf':0, 'cosNormWt':0}
                attTermDict[term]['df'] = len(attTermDict[term]['puids'])
            else:
                attTermDict[term]['puids'][csvAtt.product_uid[i]]['tf'] +=1

    for term in valueTok:
        if term not in attTermDict:
            attTermDict[term] = {'puids':{}, 'df':0, 'idf':0}
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
start_time = time.time()
att_tf = 0
att_idf = 0
attN = len(attPuidDict)

for term in attTermDict:
    for p in attTermDict[term]['puids']:
        att_tf = 1 + log10(attTermDict[term]['puids'][p]['tf'])
        att_idf = log10( 1 + (attN/attTermDict[term]['df']) )
        attTermDict[term]['idf'] = att_idf
        attTermDict[term]['puids'][p]['tfidf'] = att_tf * att_idf
        attPuidDict[p]['tfidf_list'].append(att_tf * att_idf)

print('Attribute tfidf time: ', time.time()-start_time)

start_time = time.time()

for p in attPuidDict:
    attVectorLength=0
    for i in attPuidDict[p]['tfidf_list']:
        attVectorLength = attVectorLength + (i**2)
    attVectorLength = sqrt(attVectorLength)
    attPuidDict[p]['veclen'] = attVectorLength

for term in attTermDict:
    for p in attTermDict[term]['puids']:
        if p in attPuidDict:
            attTermDict[term]['puids'][p]['cosNormWt'] = attTermDict[term]['puids'][p]['tfidf']/attPuidDict[p]['veclen']

print('Attribute cosine length time: ', time.time()-start_time)

print()

#----------read product descriptions----------------
start_time = time.time()

csvProdDes = pandas.read_csv('product_descriptions2.csv')
for i in range(0,len(csvProdDes)):
    if csvProdDes.product_uid[i] not in prodDesPuidDict:
        prodDesPuidDict[csvProdDes.product_uid[i]] = {'tfidf_list':[], 'veclen':0}

    if type(csvProdDes.product_description[i])==float:
        if math.isnan(csvProdDes.product_description[i]):
            prodDesTok = tokenize(" ")
    else:
        prodDesTok = tokenize(csvProdDes.product_description[i])

    for term in prodDesTok:
        if term not in prodDesTermDict:
            prodDesTermDict[term] = {'puids':{}, 'df':0, 'idf':0}
            prodDesTermDict[term]['puids'][csvProdDes.product_uid[i]] = {'tf':1, 'tfidf':0, 'cosNormWt':0}
            prodDesTermDict[term]['df'] = len(prodDesTermDict[term]['puids'])
        else:
            if csvProdDes.product_uid[i] not in prodDesTermDict[term]['puids']:
                prodDesTermDict[term]['puids'][csvProdDes.product_uid[i]] = {'tf':1, 'tfidf':0, 'cosNormWt':0}
                prodDesTermDict[term]['df'] = len(prodDesTermDict[term]['puids'])
            else:
                prodDesTermDict[term]['puids'][csvProdDes.product_uid[i]]['tf'] +=1



print('Product Description time: ', time.time()-start_time)

#------------product description tfidf/cosine lengths---------
start_time = time.time()
prodDes_tf = 0
prodDes_idf = 0
prodDesN = len(prodDesPuidDict)

for term in prodDesTermDict:
    for p in prodDesTermDict[term]['puids']:
        prodDes_tf = 1 + log10(prodDesTermDict[term]['puids'][p]['tf'])
        prodDes_idf = log10( 1 + (prodDesN/prodDesTermDict[term]['df']) )
        prodDesTermDict[term]['idf'] = prodDes_idf
        prodDesTermDict[term]['puids'][p]['tfidf'] = prodDes_tf * prodDes_idf
        prodDesPuidDict[p]['tfidf_list'].append(prodDes_tf * prodDes_idf)

print('Product Description tfidf time: ', time.time()-start_time)

start_time = time.time()

for p in prodDesPuidDict:
    prodDesVectorLength=0
    for i in prodDesPuidDict[p]['tfidf_list']:
        prodDesVectorLength = prodDesVectorLength + (i**2)
    prodDesVectorLength = sqrt(prodDesVectorLength)
    prodDesPuidDict[p]['veclen'] = prodDesVectorLength

for term in prodDesTermDict:
    for p in prodDesTermDict[term]['puids']:
        if p in prodDesPuidDict:
            prodDesTermDict[term]['puids'][p]['cosNormWt'] = prodDesTermDict[term]['puids'][p]['tfidf']/prodDesPuidDict[p]['veclen']

print('Product Description cosine length time: ', time.time()-start_time)