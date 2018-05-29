import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from time import time

# List of models (list of tuples containing model, description)
models = [
    (BernoulliNB(), 'Bernoulli NB with default'),
    (MultinomialNB(alpha=0.3), 'Multinomial NB with alpha=0.3'),
    (MultinomialNB(alpha=0.1), 'Multinomial NB with alpha=0.1'),
    (MultinomialNB(alpha=0.03), 'Multinomial NB with alpha=0.03'),
    (MultinomialNB(alpha=0.01), 'Multinomial NB with alpha=0.01'),
    (SGDClassifier(max_iter=1000, tol=1e-3), 'Linear SVM'),
]

def use_models(XTrain, YTrain, XTest, YTest):
    # Fit, predict, and show accuracies of each model on training and test sets
    print("CLASSIFIER ACCURACIES")
    for modelTuple in models:
        fit_predict_show(modelTuple, XTrain, YTrain, XTest, YTest)

def fit_predict_show(modelTuple, XTrain, YTrain, XTest, YTest):
    model = modelTuple[0]
    description = modelTuple[1]
    print('"' + description + '"')
    startTime = time()
    model.fit(XTrain, YTrain)
    print("\tTraining time:   %.3fs" % (time() - startTime))
    startTime = time()
    PTrain = model.predict(XTrain)
    PTest = model.predict(XTest)
    print("\tPredicting time: %.3fs" % (time() - startTime))
    print("\tTraining accuracy: %.3f%%" % (accuracy_score(YTrain, PTrain) * 100))
    print("\tTest accuracy:     %.3f%%" % (accuracy_score(YTest, PTest) * 100))
