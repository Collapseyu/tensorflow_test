import csv
csv_file=csv.reader(open('secondbpDataSet.csv','r'))
Train=[]
Test=[]
n=0
for i in csv_file:
    if n<4000:
        Train.append([i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9],i[10],i[11],i[12],i[13]])
    elif n<5000:
        Test.append([i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11], i[12], i[13]])
    n+=1
print(len(Train),len(Test))
with open('TrainSecond.csv','w',newline='') as f:
    writer=csv.writer(f)
    for row in Train:
        writer.writerow(row)
    f.close()
with open('TestSecond.csv','w',newline='') as f:
    writer=csv.writer(f)
    for row in Test:
        writer.writerow(row)
    f.close()