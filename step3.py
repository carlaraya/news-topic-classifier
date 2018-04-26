import numpy as np
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB
import mmap

def use_model(trainppFilename, dictFilename):
    words, lenDict = get_dict(dictFilename)
    wordsIndex = {words[i]: i for i in range(lenDict)}
    clf = MultinomialNB(alpha=0.1)
    numRows = count_lines(trainppFilename)
    numTrain = 120000
    numTest = numRows-numTrain

    inF = open(trainppFilename, 'r')
    XTrain, yTrain = make_sparse(wordsIndex, inF, numRows=numTrain)
    print('Fitting')
    clf.fit(XTrain, yTrain)
    print('Predicting on training set')
    print('Accuracy:', clf.score(XTrain, yTrain))
    XTest, yTest = make_sparse(wordsIndex, inF, numRows=numTest)
    print('Predicting on test set')
    print('Accuracy:', clf.score(XTest, yTest))


def make_sparse(wordsIndex, inF, numRows=None):
    lenDict = len(wordsIndex)
    X = lil_matrix((numRows, lenDict), dtype=np.uint8)
    y = []
    for i, row in zip(range(numRows), inF):
        if i % 1000 == 0: print(i)
        row = row.split(',')
        y.append(row[1].strip())
        row = row[0].split()
        for word in row:
            j = wordsIndex.get(word)
            if j:
                X[i, j] += 1
    print(i)
    print('rows:', len(y))
    return (X, y)

def get_dict(inFilename):
    rows = open(inFilename, 'r').readlines()
    return ([x.strip() for x in rows], len(rows))

def count_lines(filename):
    obj = open(filename, 'r')
    buf = mmap.mmap(obj.fileno(), 0, prot=mmap.PROT_READ)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
