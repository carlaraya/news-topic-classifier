import csv
import re


def preprocess(inFilename, outFilename):
    with open(inFilename, 'r') as inF, open(outFilename, 'w') as outF:
        reader = csv.reader(inF, delimiter=',', quotechar='"')
        next(inF)
        for row in reader:
            ppStr = row[1]
            ppStr = re.sub(r'[^a-z-$]', ' ', ppStr.lower())
            print(ppStr + ',' + row[2], file=outF)
