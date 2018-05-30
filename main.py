import cn_process, atn_process, machinelearn

#Binds all the python files earlier
def main():
    #Classifying ATN Dataset
    print("========ATN DATASET========")
    print("Preprocessing...")
    atn_process.preprocessing()
    print("Creating dictionary...")
    atn_process.create_dictionary()
    print("Generating matrix...")
    #Getting the Sparse Training Feature Vectors and Training Labels
    (XTrain, YTrain) = atn_process.create_sparse_matrix(0, 0.6)
    #Getting the Sparse Test Feature Vectors and TestLabels
    (XTest, YTest) = atn_process.create_sparse_matrix(0.6, 1)
    #Train/Test the Models using the ATN Dataset
    machinelearn.use_models(XTrain, YTrain, XTest, YTest)
    #Classifying CN Dataset
    print("========CN DATASET========")
    print('Preprocessing...')
    cn_process.preprocess()
    print('Creating dictionary...')
    cn_process.generate_dict() 
    print('Generating matrix...')
    #Getting the Sparse Training/Test Feature Vectors and Training/Test Labels
    XTrain, YTrain, XTest, YTest = cn_process.make_all_matrices()
    #Train/Test the Models using the CN Dataset
    machinelearn.use_models(XTrain, YTrain, XTest, YTest)

if __name__ == '__main__':
    main()
