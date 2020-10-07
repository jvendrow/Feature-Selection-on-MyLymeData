def subsample(data, labels, n):
    newData = []
    newLabels = []
    l = [n, n, n]
    for i in range(0, len(labels)):
        x = labels[i]
        if l[x-1] > 0: #x - 1 
            newData.append(data[i])
            newLabels.append(labels[i])
            l[x-1] -= 1
    return newData, newLabels

def subsample_binary(data, labels, n):
    newData = []
    newLabels = []
    l = [n, 0, n]
    for i in range(0, len(labels)):
        x = labels[i]
        if l[x-1] > 0: #x - 1 
            newData.append(data[i])
            newLabels.append(int(labels[i]/2))
            l[x-1] -= 1
    return newData, newLabels
