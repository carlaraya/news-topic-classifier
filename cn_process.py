import csv
import re

def preprocess(inFilename, outFilename):
    with open(inFilename, 'r') as inF, open(outFilename, 'w') as outF:
        reader = csv.reader(inF, delimiter=',', quotechar='"')
        next(inF)
        for row in inF:
            row = row.strip()
            rowTuple = clean_and_chop(row)
            if len(rowTuple) <= 3:
                topic = rowTuple[2]
            else:
                topic = rowTuple[4]
            ppStr = rowTuple[1]
            ppStr = re.sub(r'[^a-z-$]', ' ', ppStr.lower())
            print(ppStr + ',' + topic, file=outF)

def clean_and_chop(row):
    #print(row)
    if '\t' in row:
        if row.count('"') % 2 == 1:
            if row.split(',', maxsplit=1)[0].isdigit():
                row = row.replace(',', '\t', 1)
            else:
                row = row[::-1]
                row = row.replace(',', '\t', 1)
                row = row[::-1]
        return next(csv.reader([row], delimiter='\t', quoting=csv.QUOTE_NONE))
    return next(csv.reader([row], delimiter=',', quotechar='"'))



from collections import defaultdict

def generate_dict(inFilename, outFilename, limit=50000):
    dictionary = defaultdict(int)
    with open(inFilename, 'r') as inF, open(outFilename, 'w') as outF:
        for row in inF:
            row = row.split(',')[0]
            for word in row.split():
                dictionary[word] += 1

        sortedTuples = sorted(dictionary.items(), key=lambda i: i[1], reverse=True)
        dictText = '\n'.join(map(lambda i: i[0], sortedTuples[:limit]))
        print(dictText, file=outF)



import numpy as np
from scipy.sparse import coo_matrix
import mmap
import itertools

def make_all_matrices(ppFilename, dictFilename, percentTrain=0.6):
    rows = open(dictFilename, 'r').readlines()
    words = [x.strip() for x in rows]
    lenDict = len(rows)
    wordsIndex = {words[i]: i for i in range(lenDict)}

    numRows = 0
    with open(ppFilename, 'r') as obj:
        buf = mmap.mmap(obj.fileno(), 0, prot=mmap.PROT_READ)
        while buf.readline():
            numRows += 1
    numTrain = int(numRows * percentTrain)
    numTest = numRows-numTrain

    inF = open(ppFilename, 'r')
    XTrain, YTrain = make_sparse(wordsIndex, inF, numRows=numTrain)
    XTest, YTest = make_sparse(wordsIndex, inF, numRows=numTest)
    return (XTrain, YTrain, XTest, YTest)

def make_sparse(wordsIndex, inF, numRows=None):
    lenDict = len(wordsIndex)
    row = []
    col = []
    data = []
    y = []
    if numRows:
        maxRowRange = range(numRows)
    else:
        maxRowRange = itertools.count(start=0, step=1)
    for i, currRow in zip(maxRowRange, inF):
        if i % 10000 == 0: print(i)
        currRow = currRow.split(',')
        y.append(currRow[1].strip())
        headline = currRow[0].split()
        for word in headline:
            j = wordsIndex.get(word)
            if j:
                row.append(i)
                col.append(j)
                data.append(1)
    numRows = len(y)
    #print(row, col, data)
    X = coo_matrix((data, (row, col)), shape=(numRows, lenDict))
    print(i)
    print('rows:', numRows)
    return (X, y)
