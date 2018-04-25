from collections import defaultdict

def generate_dict(inFilename, outFilename):
    lenDict = 50000
    dictionary = defaultdict(int)
    with open(inFilename, 'r') as inF, open(outFilename, 'w') as outF:
        for row in inF:
            row = row.split(',')[0]
            for word in row.split():
                dictionary[word] += 1

        print(len(dictionary))
        sortedTuples = sorted(dictionary.items(), key=lambda i: i[1], reverse=True)
        dictText = '\n'.join(map(lambda i: i[0] + ' ' + str(i[1]), sortedTuples[:lenDict]))
        print(dictText, file=outF)
