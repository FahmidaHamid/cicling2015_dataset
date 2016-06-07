      READ ME

Main file: prepareData.py

Auxiliary File: hf.py

Instruction to Execute: python prepareData.py

Prerequisites: The followings should be installed to execute the code :-
             1. Python 
             2. NLTK
             3. SentiWordNet (keep it in the same folder)


My system configuration:

           Mac OSX Version 10.9.4
                 [1.8 GHz Intel Core i7 Processor]
           Python version: 2.7.5   

Change Settings on the souce code:

   in `prepareData.py', change the followings to tune output :-
      1. update path information (inPath, outPath, outPath2, outPath3, statFileName) (mandatory)
      2. update variable "testSize" to control the summary size (optional)
      3. update variable "testSentSize" to control input file size (optional)

[The given one works for files having less than or equal to 12 sentences (n) and produces summary with at least 3 to at-most (n/3) sentences] 
[It works with file written in DUC 2004 format]   
[This code generates html file for summary, anti-summary, textrank summary]
[The website has data files written in txt format as well.]
[Sample output for this specific setup is stored in 'textRankDUC2004L', 'polarityRankDUC2004L', 'auxdataFiles']
