# -*- coding:utf-8 -*-

import numpy as np
import cPickle as pickle 
import json
import re
import time
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random, copy, sys

class PrerequisiteClassifier:
    """Summary of PrerequisiteClassifier

    1. Create training and testing data from feature data.
    2. Classifying with different classfiers

    Attributes:
        # Features
            DataSet: The original features, Format: {'conceptA::;conceptB': {'features': {'VrefD': 0.0, 'SrefD': 0.0, 'GVrefD': 0.0, 'GSrefD': 0.0, 'WrefD': 0.0, 'TocD': 0.0, 'AsyD': 0.0, 'SimD': 0.0, 'ClD': 0.0}, 'label': 0}}
            PosSet: a set containing all positive concept pairs
            NegSet: a set containing all negtive concept pairs
        # Training Set & Test Set
            Xtrain: training matrix for training set
            Ytrain: label vector for training set
            Ltrain: title vector for training set
            Xtest: training matrix for test set
            Ytest: label vector for test set
            Ltest: title vector for test set
        # Class
            featuresDir: Raw features location.
            feature_names: feature names and their order in the feature vector
    """

    def __init__(self, if_baseline3, dataset, attribution_type):
        ATTR_BY = attribution_type
        IS_BSL3 = if_baseline3
        self.featuresDir = '../features/20170109/'
        self.dataSetName = dataset
        self.feature_names = ['GVrefD', 'GSrefD', 'WrefD', 'TocD', 'AsyD', 'SimD', 'ClD']
        if ATTR_BY=='con': del self.feature_names[0:3] #['GVrefD', 'GSrefD', 'WrefD']
        elif ATTR_BY=='str': del self.feature_names[3:5]; del self.feature_names[-1] #['TocD', 'AsyD', 'ClD']
        elif ATTR_BY=='sem': del self.feature_names[5:6] #['SimD']
        elif 0<=int(ATTR_BY) and int(ATTR_BY)<len(self.feature_names): del self.feature_names[int(ATTR_BY)] 
        print 'Ignore ', attribution_type
        print 'features in expr: ', self.feature_names
        self.loadDataSet()
    
    def loadDataSet(self):
        self.DataSet = pickle.load(file(self.featuresDir + self.dataSetName, 'rb'))
        self.PosSet = set()
        self.NegSet = set()
        print type(self.DataSet)
        for c in self.DataSet:
            if self.DataSet[c]["label"]==1:
                self.PosSet.add(c)
            elif self.DataSet[c]["label"]==0:
                self.NegSet.add(c)
        print 'DataSet Stastics...'
        print '#Examples', len(self.DataSet)
        print '#Positive Examples', len(self.PosSet)
        print '#Negtive Examples', len(self.NegSet)
    
    def N_K_FoldCrossValidation(self, K, fold_size, classfier_type):
        posX, posY, posL, negX, negY, negL = [], [], [], [], [], []
        for c in self.PosSet:
            fV = []
            for fn in self.feature_names:
                fV.append(self.DataSet[c]["features"][fn])
            posX.append(fV)
            posY.append(1)
            posL.append(c)
        for c in self.NegSet:
            fV = []
            for fn in self.feature_names:
                fV.append(self.DataSet[c]["features"][fn])
            negX.append(fV)
            negY.append(0)
            negL.append(c)
        
        posZip = zip(posX, posY, posL)
        negZip = zip(negX, negY, negL)
        np.random.shuffle(posZip)
        np.random.shuffle(negZip)
        zipFolds = [[] for _ in xrange(K)]
        print 'Fold num:', K, '\tFold size:', fold_size, '\tPos len:', len(posZip), '\tNeg len:', len(negZip) 
        for i in xrange(K):
            if len(posZip)/K>=(fold_size/2):
                zipFolds[i].extend(random.sample(posZip[i*len(posZip)/K:(i+1)*len(posZip)/K], fold_size/2))
            else:
                zipFolds[i].extend(random.sample(posZip[i*len(posZip)/K:(i+1)*len(posZip)/K], len(posZip)/K))
                zipFolds[i].extend([random.choice(posZip[i*len(posZip)/K:(i+1)*len(posZip)/K]) for _ in xrange(fold_size/2-len(posZip)/K)])
            if len(negZip)/K>=(fold_size/2):
                zipFolds[i].extend(random.sample(negZip[i*len(negZip)/K:(i+1)*len(negZip)/K], fold_size/2))
            else:
                zipFolds[i].extend(random.sample(negZip[i*len(negZip)/K:(i+1)*len(negZip)/K], len(negZip)/K))
                zipFolds[i].extend([random.choice(negZip[i*len(negZip)/K:(i+1)*len(negZip)/K]) for _ in xrange(fold_size/2-len(negZip)/K)])
            np.random.shuffle(zipFolds[i])
        for i in range(K):
            for j in range(K):
                if i!=j:
                    [a1, a2, a3] = [list(t) for t in zip(*zipFolds[i])]
                    [b1, b2, b3] = [list(t) for t in zip(*zipFolds[j])]
                    intersectionSet = list(set(a3).intersection(set(b3)))
                    unionSet = list(set(a3).union(set(b3)))
                    print '(',i, j,'): pre check data set, intersectionSet:', len(intersectionSet), 'unionSet:', len(unionSet)
        Ps = 0.0
        Rs = 0.0
        remain = []
        for i in xrange(K):
            temp_holder = zipFolds.pop(0)
            [self.Xtrain, self.Ytrain, self.Ltrain] = [list(t) for t in zip(*[y for x in copy.deepcopy(zipFolds) for y in x])]
            [self.Xtest, self.Ytest, self.Ltest] = [list(t) for t in zip(*temp_holder)]
            if i == 0:
                remain = copy.deepcopy(self.Ltrain)
            else:
                if remain == self.Ltrain:
                    print 'what?'
            zipFolds.append(temp_holder)
            print 'in fold:', i, 'train data size is:', len(self.Xtrain), 'test data size is:', len(self.Xtest)
            intersectionSet = list(set(self.Ltrain).intersection(set(self.Ltest)))
            unionSet = list(set(self.Ltrain).union(set(self.Ltest)))
            print '__check data set, intersectionSet:', len(intersectionSet), 'unionSet:', len(unionSet)
            P, R = 0.0, 0.0
            if classfier_type=='LR': P, R = self.LRClassifier()
            elif classfier_type=='SVM-L': P, R = self.SVMClassifier('linear')
            elif classfier_type=='SVM-R': P, R = self.SVMClassifier('rbf')
            elif classfier_type=='RF': P, R = self.RFClassifier()
            else: print 'classfier???'
            Ps += P
            Rs += R

        Ps = Ps/K
        Rs = Rs/K
        Fs = 2.0*Ps*Rs/(Ps+Rs)
        print '-------------'
        print str(K) + " fold cross validation..."
        print "average precision:", Ps
        print "average recall:", Rs
        print "f1 score:", Fs


    def LRClassifier(self):
        # TODO
        logreg = LogisticRegression(penalty='l2', C=1.8, n_jobs=-1)
        logreg.fit(self.Xtrain, self.Ytrain)    
        ypred = logreg.predict(self.Xtest)
        A = metrics.accuracy_score(self.Ytest, ypred)
        P = metrics.precision_score(self.Ytest, ypred)
        R = metrics.recall_score(self.Ytest, ypred)
        F1 = metrics.f1_score(self.Ytest, ypred)
        print "accuracy:", A
        print "precision:", P
        print "recall:", R
        print "f1 score:", F1
        print metrics.classification_report(self.Ytest, ypred, target_names=['Non', 'Pre'])
        TruePositive = set()
        for i in range(ypred.shape[0]):
            if ypred[i]==1 and self.Ytest[i]==1:
                #print self.Ltest[i]+"\t\tTrue:"+str(self.Ytest[i])+"\t\tPredict:"+str(ypred[i])
                TruePositive.add(self.Ltest[i])
        # Whether to show True Positive
        # for t in TruePositive:
            # print t
        # param_test1 = {'C':[i/10.0 for i in range(1, 100)]}
        # gsearch1 = GridSearchCV(estimator = LogisticRegression(penalty='l2', n_jobs=-1), param_grid = param_test1, scoring='f1',cv=5)
        # print gsearch1.fit(self.Xtrain, self.Ytrain)
        # print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
        return P, R

    def SVMClassifier(self, kernel_func):
        # svm = SVC(C=0.5, kernel=kernel_func, max_iter=-1)
        svm = None
        if kernel_func=='linear': svm = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', max_iter=3000)
        elif kernel_func=='rbf': svm = NuSVC(nu=0.9, kernel='rbf', max_iter=-1)
        svm.fit(self.Xtrain, self.Ytrain)
        ypred = svm.predict(self.Xtest)
        A = metrics.accuracy_score(self.Ytest, ypred)
        P = metrics.precision_score(self.Ytest, ypred)
        R = metrics.recall_score(self.Ytest, ypred)
        F1 = metrics.f1_score(self.Ytest, ypred)
        print "accuracy:", A
        print "precision:", P
        print "recall:", R
        print "f1 score:", F1
        print metrics.classification_report(self.Ytest, ypred, target_names=['Non', 'Pre'])
        # param_test1 = {'nu':[i/10.0 for i in range(1, 10)]} # {'C':[i*1.0 for i in range(1, 10)]}# {'nu':[i/10.0 for i in range(1, 10)]}
        # gsearch1 = GridSearchCV(estimator = NuSVC(kernel='rbf', max_iter=-1), param_grid=param_test1, scoring='f1', n_jobs=-1, cv=5)
        # print gsearch1.fit(self.Xtrain, self.Ytrain)
        # print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
        return P, R

    def SVMOutput(self):
        svm = SVC(kernel='linear',probability = True)
        svm.fit(self.Xtrain, self.Ytrain)
        ypred = svm.predict_proba(self.Xtest)
        TruePositive = {}
        FalseNegtive = {}
        for i in range(ypred.shape[0]):
            if ypred[i][1]>=0.7:
                TruePositive[self.Ltest[i]] = ypred[i][1]
            if ypred[i][1]<=0.3:
                FalseNegtive[self.Ltest[i]] = ypred[i][1]
        print "\n\n"
        for t in TruePositive:
            print t, TruePositive[t]
        print "\n\n"
        for t in FalseNegtive:
            print t, FalseNegtive[t]

    def RFClassifier(self):
        # TODO
        clf = RandomForestClassifier(n_estimators=170, max_features='auto', n_jobs=-1)
        clf.fit(self.Xtrain, self.Ytrain)
        ypred = clf.predict(self.Xtest)
        A = metrics.accuracy_score(self.Ytest, ypred)
        P = metrics.precision_score(self.Ytest, ypred)
        R = metrics.recall_score(self.Ytest, ypred)
        F1 = metrics.f1_score(self.Ytest, ypred)
        print "accuracy:", A
        print "precision:", P
        print "recall:", R
        print "f1 score:", F1
        print metrics.classification_report(self.Ytest, ypred, target_names=['Non', 'Pre'])
        # param_test1 = {'n_estimators':range(10,201,10)} #{'max_features':['auto','log2','sqrt', None]}
        # gsearch1 = GridSearchCV(estimator = RandomForestClassifier(max_features='auto', n_jobs=-1), param_grid=param_test1, scoring='f1', cv=5)
        # print gsearch1.fit(self.Xtrain, self.Ytrain)
        # print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
        return P, R


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print '----------WARNING----------------------------'
        print '<USAGE>\n$ python PrerequisiteClassifier.py IF_BASELINE3 DATASET K SIZE CLASSFIER ATTR_BY'
        print '\t IF_BASELINE3 \t\t [ y | n ]'
        print '\t DATASET \t\t dataset name.[ML_features_1.pkl, ...]'
        print '\t K \t\t K-fold cross validation'
        print '\t SIZE \t\t one fold size'
        print '\t CLASSFIER \t\t [ LR | SVM-L | SVM-R | RF ]'
        print '\t ATTR_BY \t\t [ con | str | sem | 0, 1, 2, 3, ... ]'
        exit()
    
    print sys.argv
    
    if_baseline3 = sys.argv[1]  # is in Baseline3?
    dataset = sys.argv[2]       # dataset
    K = int(sys.argv[3])   # K-fold cross validation
    size = int(sys.argv[4])   # size of 1 fold
    classfier_type = sys.argv[5] # classfier
    attribution_type = sys.argv[6]
    
    fg = PrerequisiteClassifier(if_baseline3, dataset, attribution_type)
    fg.N_K_FoldCrossValidation(K, size, classfier_type)
    