import numpy as np
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB
import mmap

def use_model(trainppFilename, testppFilename, dictFilename):
    words, lenDict = get_dict(dictFilename)
    wordsIndex = {words[i]: i for i in range(lenDict)}
    clf = MultinomialNB(alpha=0.01)

    XTrain, yTrain = make_sparse(wordsIndex, trainppFilename)
    print('Fitting')
    clf.fit(XTrain, yTrain)
    print('Predicting on training set')
    print('Accuracy:', clf.score(XTrain, yTrain))
    #XTest, yTest = make_sparse(wordsIndex, testppFilename)
    #print('Predicting on test set')
    #print('Accuracy:', clf.score(XTest, yTest))


def make_sparse(wordsIndex, filename):
    numRows = count_lines(filename)
    XTrain = lil_matrix((numRows, len(wordsIndex)), dtype=np.uint8)
    yTrain = []
    with open(filename, 'r') as inF:
        for row, i in zip(inF, range(numRows)):
            if i % 1000 == 0: print(i)
            row = row.split(',')
            yTrain.append(row[1].strip())
            row = row[0].split()
            for word in row:
                j = wordsIndex.get(word)
                if j:
                    XTrain[i, j] += 1
    return (XTrain, yTrain)

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
