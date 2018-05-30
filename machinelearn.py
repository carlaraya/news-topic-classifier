import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from time import time

# List of models (list of tuples containing model, description)
models = [
    (BernoulliNB(alpha=1), 'Bernoulli NB with alpha=1'),
    (BernoulliNB(alpha=0.3), 'Bernoulli NB with alpha=0.3'),
    (BernoulliNB(alpha=0.1), 'Bernoulli NB with alpha=0.1'),
    (BernoulliNB(alpha=0.03), 'Bernoulli NB with alpha=0.03'),
    (BernoulliNB(alpha=0.01), 'Bernoulli NB with alpha=0.01'),
    (MultinomialNB(alpha=1), 'Multinomial NB with alpha=1'),
    (MultinomialNB(alpha=0.3), 'Multinomial NB with alpha=0.3'),
    (MultinomialNB(alpha=0.1), 'Multinomial NB with alpha=0.1'),
    (MultinomialNB(alpha=0.03), 'Multinomial NB with alpha=0.03'),
    (MultinomialNB(alpha=0.01), 'Multinomial NB with alpha=0.01'),
    (SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=0), 'Linear SVM with stochastic gradient descent'),
]

#Accepts the following parameters
#XTrain (Traning Feature Vectors)
#YTrain (Label of the Training Dataset)
#XTest (Test Feature Vectors)
#YTest  (Label of the Test Dataset)
#Calls the function fit_predict_show for every model
def use_models(XTrain, YTrain, XTest, YTest):
    # Fit, predict, and show accuracies of each model on training and test sets
    print("MODELS")
    for modelTuple in models:
        fit_predict_show(modelTuple, XTrain, YTrain, XTest, YTest)

#Accepts the following parameters
#modelTuple (Tuple of the form (model, description))
#XTrain (Traning Feature Vectors)
#YTrain (Label of the Training Dataset)
#XTest (Test Feature Vectors)
#YTest  (Label of the Test Dataset)
#Trains modelTuple[0] in XTrain and YTrain
#Prints accuracy score on predict and test
def fit_predict_show(modelTuple, XTrain, YTrain, XTest, YTest):
    model = modelTuple[0]
    description = modelTuple[1]
    print('"' + description + '"')
    startTime = time()
    #Training
    model.fit(XTrain, YTrain)
    print("\tTraining time:   %.3fs" % (time() - startTime))
    startTime = time()
    #Predicting
    PTrain = model.predict(XTrain)
    PTest = model.predict(XTest)
    #Getting Accuracy Score
    print("\tPredicting time: %.3fs" % (time() - startTime))
    print("\tTraining accuracy: %.3f%%" % (accuracy_score(YTrain, PTrain) * 100))
    print("\tTest accuracy:     %.3f%%" % (accuracy_score(YTest, PTest) * 100))
