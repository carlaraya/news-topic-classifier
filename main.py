import cn_process, step3

def main():
    limitDict = 50000
    percentTrain = 0.6
    dataFilename = 'cn_data.csv'
    ppFilename = 'cn_pp.csv'
    dictFilename = 'dictionary.txt'
    print('STEP 1')
    cn_process.preprocess(dataFilename, ppFilename)
    print('STEP 2')
    cn_process.generate_dict(ppFilename, dictFilename, limit=limitDict) 
    print('STEP 3')
    XTrain, YTrain, XTest, YTest = cn_process.make_all_matrices(ppFilename, dictFilename, percentTrain=percentTrain)
    print('STEP 4')
    step3.train(XTrain, YTrain, XTest, YTest)

if __name__ == '__main__':
    main()
