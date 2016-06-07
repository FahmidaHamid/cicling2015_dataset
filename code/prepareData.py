#from __future__ import division
import os
import sys
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
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
import hf

class Sample:
  wid = ''
  name = ''
  pos = ''
  definition = ''
  syns = []
  val = 0.0

class sentenceSample:
    ssen = ''
    weight = 0.000
    index = 0

class polarsentenceSample:
    ssen = ''
    weightP = 0.000
    weightN = 0.000
    index = 0

def myS (myData) :
    if myData < 0.000 :
          return -1.00
    else :
          return 1.00



swn_filename = 'SentiWordNet_3.0.0_20130122.txt'
swn = SentiWordNetCorpusReader(swn_filename)
lmtzr = WordNetLemmatizer()
lemmaList = set()
engStopWords = set(stopwords.words('english'))
#############

# set test size, input directory and output directory

##############
testSize = 3 #no of sents. to be appeared in the summary and anti summary
testSentSize = 12 #maximum no of sents expected in the document
windowRange = 4 #window size
inPath = "/Users/fahmida/Desktop/randomnessTesting/dataSetDUC2004"
outPath = "/Users/fahmida/Desktop/randomnessTesting/textRankDUC2004L"
outPath2 = "/Users/fahmida/Desktop/randomnessTesting/polarityRankDUC2004L/positive"
outPath3 = "/Users/fahmida/Desktop/randomnessTesting/polarityRankDUC2004L/negative"
statFileName = "/Users/fahmida/Desktop/randomnessTesting/auxdataFiles/sentenceOverlapping.txt"
myStatFile = open(statFileName, 'w')
myStatFile.write("(Filename, (P /\ T), (N /\ T), (P /\ N), SummaryLen, #sentences)\n")
#infoPath = '/Users/fahmida/Desktop/randomnessTesting/'



#read all the files
for dirName, subdirList, fileList in os.walk(inPath):
 if dirName == ".DS_Store":
     print "Ignoring..."
 else:
  #print('Found directory: %s' % dirName)
  for fname in fileList:
   if fname != ".DS_Store" :
    print 'Filename : ', fname, '\n'
    rem = open(dirName + '/' + fname, 'r')
    txtBody =  rem.read()
    txtBody = hf.cleanRawText(txtBody)
    txtBody = '\n'.join(txtBody.split('\n')[4:]) # each article starts from line 4
    txtBody = ' '.join(txtBody.split())
    re.sub(r'[^\x00-\x7f]',r' ',txtBody)
    txtBody.rstrip()
    ##############
    ## Tokenize ##
    ##############
    sentences = nltk.sent_tokenize(txtBody)
    numberOfSent = len(sentences)
    
    
    print txtBody + '>>'+ str(numberOfSent) + '\n'
    # work only on files that have sentences <= testSentSize #
    
    if numberOfSent <= testSentSize :
        testSize = 3
        folderName = dirName.rsplit('/', 1)[1]
        summaryFileName = hf.setFilename(folderName.replace('t',''), fname.replace('.txt',''))
        graph = nx.DiGraph() #textRank graph
        pgraph = nx.DiGraph() #polar graph
        
        wordList = []
        d = defaultdict(list)
        i = 0
        sid = 0
        
        for s in sentences:
            print sid, '>>', s,'\n'
            ids = []
            y = s.lower()
            tokens = nltk.word_tokenize(y.translate(None, string.punctuation))
            tags = nltk.pos_tag(tokens)
            wid = 0
            previousStopWord = ''
            sign = 1
            for ws in tags:
                if ws[0] in engStopWords:
                      previousStopWord = ws[0].lower()
                if ws[0] not in engStopWords:
                      w = ws[0].lower()
                      poswn = hf.get_wordnet_pos(ws[1])
                      if poswn: #do not accept punctuations
                          myWord = lmtzr.lemmatize(w, poswn)
                          wsynset =  wn.synsets(myWord, poswn)
                          #print wsynset
                          s1 = Sample() 
                          s1.wid = str(wid) + '#'+str(sid) #+'#' + poswn
                          s1.name = str(myWord)
                          s1.pos = poswn

                          if previousStopWord  in ['no', 'not', 'neigther', 'nor', 'though','but','except']:
                              sign = -1
                          else :
                              sign = 1
                          
                          xBias = 0.000
                          if wsynset :
                            s1.definition = wsynset[0].definition
                            s1.syns = [i for i in wsynset]
                            x = swn.senti_synset(wsynset[0].name)
                            if x :
                                  if x.pos_score >= x.neg_score :
                                      xBias = x.pos_score * sign
                                  
                                  else :
                                      xBias = x.neg_score * (-1)* sign

                          s1.val = xBias
                          wordList.append(s1)	#global
                          ids.append(s1) #local --> for each sentence
                          graph.add_node(s1.name, pos = s1.pos, rank = 0.01)
                          pgraph.add_node(s1.name, pos = s1.pos, bias = s1.val, prestige = 0.01)

                      wid += 1
                        
                for x in ids :
                    for y in ids :
                        if x.wid != y.wid : # not the same word
                            idx = x.wid
                            idy = y.wid
                            partx = idx.split('#')
                            party = idy.split('#')
                            if abs(int(partx[0]) - int(party[0])) < windowRange :
                                    graph.add_edge(y.name,x.name, weight = 0.01)
                                    graph.add_edge(x.name,y.name, weight = 0.01)
                                    pgraph.add_edge(y.name,x.name, weight = y.val * 0.01)
                                    pgraph.add_edge(x.name,y.name, weight = x.val * 0.01)
                                                
            sid +=1
        
        #for v in wordList :
        #      d[v.name].append(v.wid)
              #print v.name, ':', v.definition,':',v.val, ':', v.syns
        
        wordConsidered = []

        for v1 in wordList :
          for v2 in wordList :
            if (v1.name != v2.name):
              pair = v1.name + '#' + v2.name
              pair2 = v2.name + '#' + v1.name
              if (pair not in wordConsidered) and (pair2 not in wordConsidered) :
                   wordConsidered.append(pair)
                   wordConsidered.append(pair2)
                   set1 = v1.definition
                   set2 = v2.definition
                   similarity = hf.findSimilarity(set1,set2)
                   commonSynset = None
                   if len(v1.syns) > 0 and len(v2.syns) > 0:
                       commonSynset = v1.syns[0]. wup_similarity(v2.syns[0], simulate_root = False)
                   csimilarity = float(0.00 if commonSynset is None else commonSynset)
                   temp = similarity + csimilarity
                   if similarity > 0.00 or csimilarity > 0.00 :
                      if graph.has_edge(v1.name,v2.name) :
                          graph.edge[v1.name][v2.name]['weight'] += temp
                          pgraph.edge[v1.name][v2.name]['weight'] += temp * myS(v1.val)
                      elif graph.has_edge(v2.name, v1.name) :
                          graph.edge[v2.name][v1.name]['weight'] += temp
                          pgraph.edge[v2.name][v1.name]['weight'] += temp * myS(v2.val)

                      else :
                          graph.add_edge(v1.name,v2.name, weight = temp )
                          graph.add_edge(v2.name,v1.name, weight = temp)
                          pgraph.add_edge(v1.name, v2.name, weight = temp * myS(v1.val))
                          pgraph.add_edge(v2.name, v1.name, weight = temp * myS(v2.val))
        
        
        print "Before Ranking:"
        for n in graph.nodes() :
            print n,'>>', pgraph.node[n], ' >> ', graph.node[n]['rank'], '\n'

        sz = graph.number_of_edges()
        ns = graph.number_of_nodes()
        print "Total Edges: ", sz
        print "Total Words: ", ns

        #################
        ## Text Rank  ##
        #################
        alpha = float(0.85)
        change = 0.1000
        threshold = 0.001
        for i in xrange(1,sz) :
           for n in graph.nodes() :
               r = 0.000
               for p in graph.predecessors(n) :
                   weightPQ = 0.00
                   for q in graph.successors(p):
                        weightPQ += graph.edge[p][q]['weight']
                   if weightPQ > 0.00 :
                        r += float(graph.node[p]['rank'] * graph.edge[p][n]['weight'])/weightPQ
               if(graph.in_degree(n)!= 0) :
                   previous = graph.node[n]['rank']
                   graph.node[n]['rank'] = r * alpha + (1 - alpha)
                   nChange = math.fabs(previous - graph.node[n]['rank'])
                   if nChange > change :
                      change = nChange
           if change < threshold :
               break
                              
        ######################
        ### Polarity Rank  ###
        ######################
        
        for i in xrange(1,sz) :
            for n in pgraph.nodes() :
              r = 0.000
              for p in pgraph.predecessors(n) :
                  x_kj = 0.000
                  if  pgraph.edge[p][n]['weight'] * pgraph.node[p]['bias'] > 0.00 :
                      x_kj = math.fabs(pgraph.node[p]['bias'])
                  r += float(pgraph.edge[p][n]['weight'] * (1.00 - x_kj))
              if pgraph.in_degree(n) > 0 :
                  #previous = pgraph.node[n]['prestige']
                  pgraph.node[n]['prestige'] = r /float(pgraph.in_degree(n))
            for n in pgraph.nodes() :
                b = 0.000
                for s in pgraph.successors(n) :
                    b += pgraph.edge[n][s]['weight'] - pgraph.node[s]['prestige']
                if pgraph.out_degree(n) > 0 :
                    pgraph.node[n]['bias'] = b / (float(2.00 * pgraph.out_degree(n)))
            for n in pgraph.nodes() :
                for p in pgraph.predecessors(n) :
                  x_kj = 0.000
                  if  pgraph.edge[p][n]['weight'] * pgraph.node[p]['bias'] > 0.00 :
                      x_kj = math.fabs(pgraph.node[p]['bias'])
                      pgraph.edge[p][n]['weight'] = pgraph.edge[p][n]['weight'] * ( 1.00 - x_kj)

       
        highestR = -1.00
        lowestR = 1.00
        for n in pgraph.nodes() :
           if highestR < pgraph.node[n]['prestige'] and pgraph.node[n]['prestige'] != float('inf') :
                        highestR = pgraph.node[n]['prestige']
           if lowestR > pgraph.node[n]['prestige'] and pgraph.node[n]['prestige'] != float('-inf') :
                        lowestR = pgraph.node[n]['prestige']
        for n in pgraph.nodes() :
            if pgraph.node[n]['prestige'] == float('inf') :
                        pgraph.node[n]['prestige'] = highestR
            if pgraph.node[n]['prestige'] == float('-inf') :
                        pgraph.node[n]['prestige'] = lowestR


        print 'After Ranked:\nNode >> Rank >> Prestige >> Bias\n'
        ###############################
        ## Print Rank and Prestige
        ###############################
        for n in pgraph.nodes() :
              print n, '>>', pgraph.node[n], ' >> ', graph.node[n]['rank'],'\n'


        ########################################
        #### TEXTRANK Summary Sentences ########
        ########################################
        totalWeight = 0.00
        for n in graph.nodes() :
               totalWeight += math.fabs(graph.node[n]['rank'])

        sentenceList = []
        myNodes = graph.number_of_nodes() 
        index = 0        
        for s in sentences :
            xs = sentenceSample()
            xs.ssen = s
            xs.index = index
            index += 1
            total = 0.00
            totalNodes = 0
            for n in s.split() :
                if n in graph.nodes() :
                    total += math.fabs(graph.node[n]['rank'])
                    totalNodes += 1
            if totalNodes > 0 and totalWeight > 0.00 :
                xs.weight = (float)((total * myNodes) /(totalNodes * totalWeight))
            sentenceList.append(xs)

        ################################
        ###   control testSize       ###
        ################################

        testSize2 = math.ceil(int(float(numberOfSent)/3.00))
        if testSize < testSize2 :
           testSize = testSize2
        if testSize > numberOfSent :
           testSize = numberOfSent



        topTR = []
        setSize = 0
        print "TextRank output:\n"
        for ps1 in sorted(sentenceList, key=lambda ps1: ps1.weight, reverse = True) :
            print ps1.ssen, '>>', ps1.weight, '>>', ps1.index, '\n'
            if(setSize < testSize) :
                  topTR.append(ps1.index)
            setSize += 1


        sumSentences = []
        for ps1 in sorted(sentenceList, key=lambda ps1: ps1.index, reverse = False) :
           if ps1.index in topTR :
               sumSentences.append(ps1.ssen)

        hf.writeFile(outPath, summaryFileName, sumSentences)

        ############################################
        ### Poalrity Ranked Sentences  ###
        ############################################


        h2 = pgraph.to_undirected()
        totalWeight2 = 0.00
        positiveNode = 0
        negativeNode = 0
        for n in h2.nodes() :
             totalWeight2 += math.fabs(h2.node[n]['prestige'])
             if h2.node[n]['prestige'] < 0.00 :
                  positiveNode += 1
             else :
                  negativeNode += 1
    
        sentenceList2 = []
        myNodes = h2.number_of_nodes()
        index = 0
        for s in sentences :
            xs = polarsentenceSample()
            xs.ssen = s
            xs.senIndex = index
            index += 1
        
            totalP = 0.00
            totalN = 0.00
            totalNodes = 0
        
            for n in s.split() :
                if n in h2.nodes() :
                   if(h2.node[n]['prestige'] > 0.00) :
                         totalP += math.fabs(h2.node[n]['prestige'])
                   else :
                         totalN += math.fabs(h2.node[n]['prestige'])
                totalNodes += 1
            if  totalNodes > 0 and totalWeight2 > 0.00:
                xs.weightP = (float) (totalP/ (totalWeight2 * totalNodes))
                xs.weightN = (float) (totalN/ (totalWeight2 * totalNodes))
            sentenceList2.append(xs)

        topFiveP = []
        five = 0
        print "Positive Summary:\n"
        for ps1 in sorted(sentenceList2, key=lambda ps1: ps1.weightP, reverse = True) :
           print ps1.ssen, '>>', ps1.weightP, '>>', ps1.senIndex, '\n'
           if(five < testSize) :
               topFiveP.append(ps1.senIndex)
           five += 1
    
        topFiveN = []
        five = 0
        print "Anti-Summary:\n"
        for ps1 in sorted(sentenceList2, key=lambda ps1: ps1.weightN, reverse = True) :
           print ps1.ssen, '>>', ps1.weightN, '>>', ps1.senIndex, '\n'
           if(five < testSize) :
               topFiveN.append(ps1.senIndex)
           five += 1

        psumSentences = []
        for ps1 in sorted(sentenceList2, key = lambda ps1 : ps1.senIndex, reverse = False) :
            if ps1.senIndex in topFiveP:
                 sent = re.sub(r'\s+', ' ', ps1.ssen)
                 psumSentences.append(sent)
        hf.writeFile(outPath2, summaryFileName, psumSentences)

        psumSentences = []
        for ps1 in sorted(sentenceList2, key = lambda ps1 : ps1.senIndex, reverse = False) :
            if ps1.senIndex in topFiveN:
               sent = re.sub(r'\s+', ' ', ps1.ssen)
               psumSentences.append(sent)


        hf.writeFile(outPath3, summaryFileName, psumSentences)


       #########################
       ## Find Sentence Intersection
       #########################
        matchPN = 0
        matchP = 0
        matchN = 0
           
        for x in topTR :
               if x in topFiveP :
                   matchP += 1
               if x in topFiveN :
                   matchN += 1
        for x in topFiveP :
               if x in topFiveN :
                   matchPN += 1
        myStatList = [matchP, matchN, matchPN, testSize, numberOfSent]
        print myStatList
        print >> myStatFile, summaryFileName, ', ', matchP, ', ', matchN, ', ',matchPN, ', ', testSize, ', ',numberOfSent, '\n'


myStatFile.close()