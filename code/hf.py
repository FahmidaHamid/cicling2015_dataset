#from __future__ import division
import os
import sys
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
#from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import nltk.tag
import string
from sentiwordnet import SentiWordNetCorpusReader, SentiSynset
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import math
from itertools import izip
import pprint




def get_wordnet_pos(treebank_tag):
    
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''



def findSimilarity(x,y):
    
    stop = stopwords.words('english')
    for i in string.punctuation :
        stop.append(i)
    spX = [i for i in nltk.word_tokenize(x.lower()) if i not in stop]
    spY = [i for i in nltk.word_tokenize(y.lower()) if i not in stop]
    
    
    setX = set(spX)
    setY = set(spY)
    
    if(len(setX) == 0 or len(setY) == 0) :
        return 0
    else :
        i =  len(setX.intersection(setY))
        j =  len(setX.union(setY))
        return float (i) / float (j)

def overlap(x,y) :
    spX = nltk.word_tokenize(x.translate(None, string.punctuation))
    spY = nltk.word_tokenize(y.translate(None, string.punctuation))
    
    if(len(spX) == 0 or len(spY) == 0) :
        return 0
    else :
        mc = 0
        for wxs in spX :
            if wxs not in engStopWords:
                if wxs in spY :
                    mc = mc + 1
        return mc




def median(mylist):
    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return (sorts[int(length / 2)] + sorts[int(length / 2) - 1]) / 2.0
    return sorts[int(length / 2)]

def cleanhtml(raw_html):
    
    cleanr =re.compile('<.*?>')
    cleantext = re.sub(cleanr,'', raw_html)
    return cleantext

def setFilename(folderName, fileName) :
    
    name = folderName.upper() + '.P.10.T.TextRank.'+fileName.upper()+'.html'
    return name


def writeFile(path, fileName, mySummary) :
    f = open(path +'/'+ fileName,'w')
    print >> f , '<html>'
    print >> f , '<head>\n<title>'+fileName +'</title>'
    print >> f, '</head>\n<body bgcolor="white">'
    for index, item in enumerate(mySummary):
        x = index + 1
        print >> f, '<a size="10" name="' + str(x)+'">['+str(x)+']</a> <a href="#'+str(x) + '" id='+ str(x) + '>' + item + '</a>'
    
    print >> f, '</body>\n</html>'
    f.close()

def cleanRawText(rawText):
    cleanText = cleanhtml(rawText)
    cleanText = '\n'.join(cleanText.split('\n')[4:])
    re.sub(r'[^\x00-\x7f]',r' ',cleanText)
    cleanText = cleanText.replace('\'\'', '\"').replace('``','\"')
    cleanText = cleanText.replace('?\"', '\"?').replace('!\"', '\"!').replace('.\"', '\".')
    return cleanText



def printWordSample(s) :
    
    print 'Lemma: ', s.name
    print 'POS: ', s.pos
    print 'Definition: ', s.definition
    print 'Synsets:', s.syns
