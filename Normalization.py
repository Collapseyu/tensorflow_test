import numpy as np
import csv
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 取每一列的最小值
    maxVals = dataSet.max(0) # 取每一列的最大值
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
csv_file=csv.reader(open('TrainSecond.csv','r'))
data=[]
for i in csv_file:
   data.append(i)
tuData=np.array(data,dtype=float)
#print(npData)
normData ,_ ,_ =autoNorm(tuData)
#npData=np.array(tuData)
print(type(normData))