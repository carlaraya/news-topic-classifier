from sklearn.naive_bayes import MultinomialNB

def train(XTrain, YTrain, XTest, YTest):
    clf = MultinomialNB(alpha=0.1)
    print('Fitting')
    clf.fit(XTrain, YTrain)
    print('Predicting on training set')
    print('Accuracy:', clf.score(XTrain, YTrain))
    print('Predicting on test set')
    print('Accuracy:', clf.score(XTest, YTest))
