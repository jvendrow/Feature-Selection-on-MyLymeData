import math
import random
import csv
from collections import Counter
import sys
import numpy as np

from subsample import subsample, subsample_binary



# Preprocessing
# ----------------------------------------------

def load_csvs():

    # Importing CSV Files 
    # Turn the csv into a matrix a
    # filename = "../data/data_complete.csv"
    filename = "../data/dataset.csv"
    ifile = open(filename, "r")
    reader = csv.reader(ifile, delimiter=",")
    rownum = 0	
    data = []
    for row in reader:
        data.append (row)
        rownum += 1
        
    ifile.close()

    # Turn the csv into a matrix a
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
        pass
        #X, Y = func(data[:], l[:], len(data))
        X = data
        Y = l

    return np.asarray(X), np.asarray(Y)

