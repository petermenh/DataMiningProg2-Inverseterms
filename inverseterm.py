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
import csv as csv

#mytokenizer = RegexpTokenizer(r'[a-zA-Z0-9]{2,}')
mytokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
stemmer = PorterStemmer()
sortedstopwords = sorted(stopwords.words('english'))
total_puid_prod_des = []
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

def check_if_num(a):
   try:
       float(a)
   except ValueError:
       return False
   return True

#----------------Main------------------------------------------

#----------read attributes, calc tfidf--------------
start_time = time.time()
count = 0
csvAtt = pandas.read_csv('attributes.csv')
print('0% done...')
for i in range(0,len(csvAtt)):
    count +=1
    if count == 100000:
        print('%{0:.0f} done...'.format((i/len(csvAtt))*100) )
        count = 0

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
print('done')
print()
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

'''
#----------read product descriptions----------------
start_time = time.time()

csvProdDes = pandas.read_csv('product_descriptionsshort.csv')
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
'''
#---------read test and output result
start_time = time.time()

csvTest = pandas.read_csv('test.csv', sep=",", encoding="ISO-8859-1")

with open('subtest.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',', quotechar = "'")
    writer.writerow(['"id"']+['"relevance"'])
    for i in range(0,len(csvTest)):
        searchDict = {}
        relevance = 2
        searchVecLen = 0
        searchPUID = csvTest.product_uid[i]

        searchTok = tokenize(csvTest.search_term[i])
        for term in searchTok:
            if term in attTermDict:
                searchDict[term] = {'idf':attTermDict[term]['idf'], 'cosNormWt':0, }
        
        for term in searchDict:
            searchVecLen = searchVecLen + (searchDict[term]['idf']**2)
        searchVecLen = sqrt(searchVecLen)

        for term in searchDict:
            searchDict[term]['cosNormWt'] = searchDict[term]['idf']/searchVecLen

        for term in searchDict:
            if searchPUID in attTermDict[term]['puids']:
                relevance = relevance + searchDict[term]['cosNormWt']*attTermDict[term]['puids'][searchPUID]['cosNormWt']
            else: 
                relevance = relevance - (1/len(searchDict))
        #print(csvTest.id[i],' , ', '%.1f'%relevance)
        writer.writerow([csvTest.id[i], '%.1f'%relevance])


print('test time: ', time.time()-start_time)