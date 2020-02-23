import sys
import matplotlib.pyplot as plt

def getKey(d, key):
    try:
        return d[key]
    except:
        return None

def ComputeAverage():
    x = []
    y = []
    for i in timeDict:
        x.append(i)
        y.append(timeDict[i]/countDict[i])
    return x, y

def plotAvg(x, y):
    plt.scatter(x,y)
    plt.show()
    return

timeDict = {}
countDict = {}
f = open('timeMeasures.txt', 'r')
for line in f:
    if line.__contains__('--'):
        print(timeDict)
        print(countDict)
        x, y = ComputeAverage()
        # print(x)
        # print(y)
        # plotAvg(x, y)
        for i in range(0, len(x)):
            print(f'{x[i]:36} - {y[i]:20} : {countDict[x[i]]}')
        timeDict.clear()
        countDict.clear()
        print(line)
    else:
        line = line[1:len(line)-2]
        entry = line.split(', ')
        key = entry[0][1:len(entry[0])-1]
        val = getKey(timeDict, key)
        if val is None:
            timeDict[key] = float(entry[1])
            countDict[key] = 1
        else:
            timeDict[key] += float(entry[1])
            countDict[key] += 1
