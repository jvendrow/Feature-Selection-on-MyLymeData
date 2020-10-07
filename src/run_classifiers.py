import math
import random
import csv
from collections import Counter

# numpy libraries
import numpy as np

#sklearn library
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import metrics

import matplotlib as mpl
import matplotlib.pyplot as plt

from subsample import subsample, subsample_binary



def load_csvs():

    # Importing CSV Files 
    #------------------------------------
    #Turn the csv into a matrix a
    #filename = "../data/data_complete.csv"
    filename = "../data/dataset.csv"
    ifile = open(filename, "r")
    reader = csv.reader(ifile, delimiter=",")
    rownum = 0	
    data = []
    for row in reader:
        data.append (row)
        rownum += 1
        
    ifile.close()

    #Turn the csv into a matrix a
    filename = "../data/labels.csv"
    ifile = open(filename, "r")
    reader = csv.reader(ifile, delimiter=",")
    rownum = 0	
    labels = []
    for row in reader:
        labels.append (row)
        rownum += 1
        
    ifile.close()

    return data, labels


def get_data(subsamp=True, binary=False):

    data, labels = load_csvs()

    l = list(np.array(labels).T[0])
    l = [int(x) for x in l]

    for i in range(0, len(data)):
        data[i] = [int(x) if float(x) // 1 == x else float(x) for x in data[i]]

    sub = 0
    if(binary):
        sub = 819
    else:
        sub = 396

    if binary:
        func = subsample_binary
    else:
        func = subsample
    if(subsamp):
        X, Y = func(data[:], l[:], sub)

    else:
        X, Y = func(data[:], l[:], len(data))

    return X, Y


def error(clf, X, Y, ntrials=30, test_size=0.25, by_label=False):
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    train_sum = 0
    test_sum = 0
    train_scores = []
    test_scores = []


    if(np.max(Y) == 3):
        counts = [0, 0, 0]
        correct = [0, 0, 0]
    else:
        counts = [0, 0]
        correct = [0, 0]

    for i in range(0, ntrials):

        xtrain, xtest, ytrain, ytest = train_test_split (X,Y, test_size = test_size, random_state = i)
        clf.fit(xtrain, ytrain)                  # fit training data using the classifier

        #y_pred = clf.predict(xtrain)        # take the classifier and run it on the training data
        #train_error_trial = 1 - metrics.accuracy_score(ytrain, y_pred, normalize=True)
        #train_error_trial = hinge_loss(ytrain, y_pred)
        #train_sum += train_error_trial
        #train_scores.append(train_error_trial)

        y_pred = clf.predict(xtest)
        test_error_trial = 1 - metrics.accuracy_score(ytest, y_pred, normalize=True)
        test_sum += test_error_trial
        test_scores.append(test_error_trial)
        if by_label:
            for i in range(0, len(ytest)):
                counts[ytest[i] - 1] += 1
                if ytest[i] == y_pred[i]:
                    correct[ytest[i] - 1] += 1
    
    if(by_label):
        print(counts)
        print(correct)
        for i in range(0, np.max(Y)):
            print(float(correct[i]) / counts[i])

    train_error = train_sum# / ntrials  ## average error over all the @ntrials
    test_error = test_sum / ntrials

    return train_error, test_error


def acc_by_label(Y, y_pred):
    if(binary):
        counts = [0, 0]
        correct = [0, 0]
    else:
        counts = [0, 0, 0]
        correct = [0, 0, 0]

    for i in range(0, len(Y)):
        counts[Y[i] - 1] += 1
        if Y[i] == y_pred[i]:
            correct[Y[i] - 1] += 1

    print(counts)
    print(correct)
    for i in range(0, 3):
        print(float(correct[i]) / counts[i])


def run_classifier(clf, X, Y, binary=False, by_label=False, verbose=True):
    #random.shuffle(Y)

    clf.fit(X, Y)
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_accuracy =  metrics.accuracy_score(Y, y_pred, normalize=True)
    print('\t-- training accuary: %.5f' % train_accuracy)

    train_error, test_error = error(clf, X, Y, 20)  
    test_accuracy = 1 - test_error
    print('\t-- test accuary: %.5f' % test_accuracy)

    if(by_label):
        #Calculate Individual Label Accuracy
        acc_by_label(Y, y_pred)

    return train_accuracy, test_accuracy

def run_single(clf, X, Y):

    saved_X = X;
    for i in range(0, len(X[1])):
        X = saved_X
        #X = np.delete(X, i, 1).tolist()
        X = [[a[i]] for a in X]
        train_error, test_error = error(clf, X, Y, 20)
        test_accuracy = 1 - test_error

        print(test_accuracy)




def main():
    clf_svm = svm.SVC(kernel='linear', random_state=1)
    clf_knn = KNeighborsClassifier(n_neighbors = 41)
    clf_tree = DecisionTreeClassifier(criterion = "entropy", max_depth=3)

    names = ["svm", "knn", "tree"]
    classifier = [clf_svm, clf_knn, clf_tree]

    X, Y = get_data(subsamp=True, binary=False)
    X_bin, Y_bin = get_data(subsamp=True, binary=True)

    name = names[2]
    print(name, "subsample binary")
    #run_classifier(classifier[2], X_bin, Y_bin, binary=False)
    #run_single(classifier[0], X, Y)

    run_single(classifier[2], X_bin, Y_bin)


if __name__ == '__main__':
    main()
