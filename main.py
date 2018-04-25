import step1, step2


def main():
    trainFilename = 'train_data.csv'
    ppFilename = 'train_pp.csv'
    dictFilename = 'dictionary.txt'
    step1.preprocess(trainFilename, ppFilename)
    step2.generate_dict(ppFilename, dictFilename) 

if __name__ == '__main__':
    main()
