import csv
import re
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
import mmap
import itertools

#Setting variables
limitDict = 50000
#Percent of Training dataset
percentTrain = 0.6
#CN Dataset
dataFilename = 'cn_data.csv'
#Filename for preprocessed dataset
ppFilename = 'cn_pp.csv'
#Filename for dictionary
dictFilename = 'cn_dictionary.txt'

#Accepts no parameter
#Gets all the headline in the csv file, cleans it, and places it to another csv
def preprocess():
    #Each headline in the original csv is the iterated, cleaned, and placed to the outF file.
    with open(dataFilename, 'r') as inF, open(ppFilename, 'w') as outF:
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

#Accepts parameterr row
#row is the unprocessed headline from the csv input
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


#Accepts no parameter
#Gets all the words in the deadline and places it in the dictionary text file
def generate_dict():
    dictionary = defaultdict(int)
    with open(ppFilename, 'r') as inF, open(dictFilename, 'w') as outF:
        for row in inF:
            row = row.split(',')[0]
            for word in row.split():
                dictionary[word] += 1

        sortedTuples = sorted(dictionary.items(), key=lambda i: i[1], reverse=True)
        dictText = '\n'.join(map(lambda i: i[0], sortedTuples[:limitDict]))
        print(dictText, file=outF)

#Accepts no parameters
#Creates the matrices needed for the tranining and predicting process
#Returns a 4-tuple containing the following
#XTrain (CN Training Sparse Feature Vectors)
#YTrain (CN Training Labels)
#XTest (CN Test Sparse Feature Vectors)
#YTrain (CN Test Labels)
def make_all_matrices():
    rows = open(dictFilename, 'r').readlines()
    words = [x.strip() for x in rows]
    lenDict = len(rows)
    wordsIndex = {words[i]: i for i in range(lenDict)}

    # Count number of rows in preprocessed file
    numRows = 0
    with open(ppFilename, 'r') as obj:
        # Use mmap for faster line reading of file
        buf = mmap.mmap(obj.fileno(), 0, prot=mmap.PROT_READ)
        while buf.readline():
            numRows += 1
    numTrain = int(numRows * percentTrain)
    numTest = numRows-numTrain
    inF = open(ppFilename, 'r')
    XTrain, YTrain = make_sparse(wordsIndex, inF, numRows=numTrain)
    XTest, YTest = make_sparse(wordsIndex, inF, numRows=numTest)
    return (XTrain, YTrain, XTest, YTest)

#Accepts the following parameters
#wordsIndex (Simply contains the words in the dictionary)
#inF (preprocessed file)
#numRows (number of headlines)
#Returns the Sparse Feature Vectors and Labels
def make_sparse(wordsIndex, inF, numRows=None):
    lenDict = len(wordsIndex)
    rowIndex = []
    colIndex = []
    data = []
    Y = []
    if numRows:
        maxRowRange = range(numRows)
    else:
        maxRowRange = itertools.count(start=0, step=1)
    for i, currRow in zip(maxRowRange, inF):
        currRow = currRow.split(',')
        Y.append(currRow[1].strip())
        headline = currRow[0].split()
        for word in headline:
            j = wordsIndex.get(word)
            if j:
                rowIndex.append(i)
                colIndex.append(j)
                data.append(1)
    numRows = len(Y)
    X = coo_matrix((data, (rowIndex, colIndex)), shape=(numRows, lenDict))
    return (X, Y)
