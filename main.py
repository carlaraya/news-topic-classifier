import cn_process, atn_process, machinelearn

def main():

    print("========CN DATASET========")
    print('Preprocessing...')
    cn_process.preprocess()
    print('Creating dictionary...')
    cn_process.generate_dict() 
    print('Generating matrix...')
    XTrain, YTrain, XTest, YTest = cn_process.make_all_matrices()

    machinelearn.use_models(XTrain, YTrain, XTest, YTest)

    print("========ATN DATASET========")
    print("Preprocessing...")
    #atn_process.preprocessing()
    print("Creating dictionary...")
    #atn_process.create_dictionary()
    print("Generating matrix...")
    (XTrain, YTrain) = atn_process.create_sparse_matrix(0, 0.6)
    (XTest, YTest) = atn_process.create_sparse_matrix(0.6, 1)

    machinelearn.use_models(XTrain, YTrain, XTest, YTest)

if __name__ == '__main__':
    main()
