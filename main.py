import step1


def main():
    trainFilename = 'train_data.csv'
    ppFilename = 'train_pp.csv'
    step1.preprocess(trainFilename, ppFilename)

if __name__ == '__main__':
    main()
