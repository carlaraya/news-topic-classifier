import step1, step2, step3


def main():
    limitDict = 50000
    trainFilename = 'train_data.csv'
    trainppFilename = 'train_pp.csv'
    dictFilename = 'dictionary.txt'
    print('STEP 1')
    step1.preprocess(trainFilename, trainppFilename)
    print('STEP 2')
    step2.generate_dict(trainppFilename, dictFilename, limit=limitDict) 
    print('STEP 3')
    step3.use_model(trainppFilename, dictFilename) 

if __name__ == '__main__':
    main()
