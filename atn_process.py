import re
import enchant
from scipy.sparse import coo_matrix

import os
from os import path, mkdir

"""
make directory if not exists
"""
def makedir(newdir):
    if not path.exists(newdir):
        mkdir(newdir)

def create_clean(news):
    unwantedPatterns = [r'\'s', r'[0-9]+', r'[\%\(\)\.\,\;\:\"\-]', r'\s+']
    raw=open(path.join('atn_data_files', news),"r", encoding="ISO-8859-1")
    cleaned=open(path.join('atn_data_filesv2', news),"w")
    processed=raw.read()
    processed=processed.lower()
    for x in unwantedPatterns:
        processed=re.sub(x, r' ', processed)
    cleaned.write(processed)

def preprocessing():
    makedir('atn_data_filesv2')
    # list all classes (business, tech, etc.) found in data_files
    for newsClass in os.listdir('atn_data_files'):
        # parse file found in each class
        makedir(path.join('atn_data_filesv2', newsClass))
        for filename in os.listdir(path.join('atn_data_files', newsClass)):
            create_clean(path.join(newsClass, filename))

def create_dictionary():
    d = enchant.Dict("EN-US")
    dictionary=set()
    for newsClass in os.listdir('atn_data_filesv2'):
        for filename in os.listdir(path.join('atn_data_filesv2', newsClass)):
            raw=open(path.join('atn_data_filesv2', newsClass, filename), "r")
            text=raw.read()
            words=text.split()
            for i in words:
                if(i[0]=='-'):
                    continue
                if(i[len(i)-1]=='-'):
                    continue
                if(i[len(i)-1]=='S'):
                    if(i[len(i)-2]=='\''):
                        i=i[:-2]
                if d.check(i):
                    dictionary.add(i)
    dictfile = open("dictionary.txt", "w")
    wordlist=list(dictionary)
    wordlist.sort()
    for i in wordlist:
       dictfile.write(i+"\n")

def create_sparse_matrix(lowerPercent, upperPercent):
    dictionary=[]
    file=open("dictionary.txt", "r")
    for i in file.readlines():
        i=i.strip()
        dictionary.append(i)
    dictIndex = {dictionary[i]: i for i in range(len(dictionary))}
    rowIndex = []
    colIndex = []
    data = []

    Y = []
    i = 0
    for newsClass in sorted(os.listdir('atn_data_filesv2')):
        sortedFiles = sorted(os.listdir(path.join('atn_data_filesv2', newsClass)))
        lowerBound = int(len(sortedFiles) * lowerPercent)
        upperBound = int(len(sortedFiles) * upperPercent)
        for filename in sortedFiles[lowerBound:upperBound]:
            current = path.join('atn_data_filesv2', newsClass, filename)
            words = open(current, 'r').read().split()
            for word in words:
                j = dictIndex.get(word)
                if j:
                    rowIndex.append(i)
                    colIndex.append(j)
                    data.append(1)
            Y.append(newsClass)
            i += 1
    X = coo_matrix((data, (rowIndex, colIndex)), shape=(i, len(dictionary)))
    return (X, Y)
