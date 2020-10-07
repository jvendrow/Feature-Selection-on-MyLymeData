import math
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
#from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#import pydotplus

import matplotlib as mpl
import matplotlib.pyplot as plt

from subsample import subsample

# Importing CSV Files 
#------------------------------------
#Turn the csv into a matrix a
#filename = "../data/data_complete.csv"
#filename = "../data/data_change_1040.csv"
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
#filename = "../data/labels_no_antibiotics.csv"
filename = "../data/labels.csv"

#filename = "labels_responders_67.csv"
ifile = open(filename, "r")
reader = csv.reader(ifile, delimiter=",")
rownum = 0	
labels = []
for row in reader:
    labels.append (row)
    rownum += 1
    
ifile.close()

labels = list(np.array(labels).T[0])
labels = [int(int(x)/2) for x in labels if x != 1]


def entropy(l):
    t = sum(l)
    if t == 0:
        return 0
    e = 0
    for i in l:
        d = float(i) / float(t)
        if(i == 0):
            continue
        e = e - d * math.log(d) / math.log(2) 
    return e

for i in range(0, len(data)):
    data[i] = [int(x) if float(x) // 1 == x else float(x) for x in data[i]]

count = [0, 0, 0]
for i in labels:
    count[i-1] += 1

"""
resultFile = open("calculate_entropy_no_antibiotics.csv",'w')
wr = csv.writer(resultFile, delimiter=",")
wr.writerow([entropy(count)])
"""
for j in range(0, len(data[1])):
    q = [];
    for i in range(0, 100):
        q.append([0,0])

    for i in range(0, len(labels)):
        q[int(data[i][j])][int(labels[i])-1] += 1

    if j ==1:
        print(q)

    total_e = 0
    for i in q:
        total_e += float(sum(i)) * entropy(i)

    total_e = total_e / sum(count)
    print(total_e)
    #wr.writerow([total_e])

