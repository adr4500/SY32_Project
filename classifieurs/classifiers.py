import numpy as np
import random as rd
from skimage.feature import hog
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt


class CrossVerification:
    '''
    This class automates the cross verification process with a given classifier.
    '''
    def __init__(self, X, y, clf, n_folds=10):
        '''
        X: data
        y: labels
        clf: classifier
        n_folds: number of folds
        '''
        self.X = X
        self.y = y
        self.clf = clf
        self.n_folds = n_folds
        self.kf = KFold(n_splits=n_folds, shuffle=True)
        self.scores = []
    
    def run(self):
        '''
        Run the cross verification
        '''
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            self.clf.train(X_train, y_train)
            self.scores.append(self.clf.score(X_test, y_test))
    
    def get_mean_score(self):
        '''
        Return the mean score of the cross verification
        '''
        return np.mean(self.scores)

class AdaBoost:
    def __init__(self, n_estimators):
        self.clf = AdaBoostClassifier(n_estimators=n_estimators)
    
    def train(self, X, y):
        self.clf.fit(X, y)
    
    def predict(self, X):
        return self.clf.predict(X)
    
    def score(self, X, y):
        return self.clf.score(X, y)
    
    def prediction_scores(self, X):
        return self.clf.decision_function(X)
    
class BestClassifierFinder:
    '''
    This class finds the best classifier for a given dataset.
    '''
    def __init__(self, X, y, n_folds=10):
        '''
        X: data
        y: labels
        n_folds: number of folds
        '''
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.classifiers = [AdaBoost(k) for k in range(40, 400)]
        self.scores = []
    
    def run(self):
        '''
        Run the cross verification for each classifier
        '''
        for i in range(len(self.classifiers)):
            print(str(i/len(self.classifiers)*100) + "%")
            cv = CrossVerification(self.X, self.y, self.classifiers[i], self.n_folds)
            cv.run()
            self.scores.append(cv.get_mean_score())
    
    def display_scores(self):
        '''
        Display the scores of each classifier
        '''
        plt.plot(self.scores)
        plt.show()
    
    def get_best_classifier(self):
        '''
        Return the best classifier
        '''
        best_score = max(self.scores)
        best_clf = self.classifiers[self.scores.index(best_score)]
        return best_clf, best_score