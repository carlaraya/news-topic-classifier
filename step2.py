from collections import defaultdict

def generate_dict(inFilename, outFilename, limit=50000):
    dictionary = defaultdict(int)
    with open(inFilename, 'r') as inF, open(outFilename, 'w') as outF:
        for row in inF:
            row = row.split(',')[0]
            for word in row.split():
                dictionary[word] += 1

        sortedTuples = sorted(dictionary.items(), key=lambda i: i[1], reverse=True)
        #dictText = '\n'.join(map(lambda i: i[0] + ' ' +str(i[1]), sortedTuples[:limit]))
        dictText = '\n'.join(map(lambda i: i[0], sortedTuples[:limit]))
        print(dictText, file=outF)
