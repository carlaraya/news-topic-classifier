import csv
import re
from nltk.stem.porter import PorterStemmer

def preprocess(inFilename, outFilename, stem=True):
    stemmer = PorterStemmer()
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
            tokens = ppStr.split()
            print(' '.join(tokens) + ',' + topic, file=outF)

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
