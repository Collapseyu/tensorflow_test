import os
import csv
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
BATCH_SIZE=500 #guiyihua 500 label[10]
LEARNING_RATE_BASE=0.001 #guiyihua 0.001 label[10]
LEARNING_RATE_DECAY=0.99
REGULARAZTION_RATE=0.0001
TRAINING_STEPS=100000
MOVING_AVERAGE_DECAY=0.99
TRAINNUM=4000
MODEL_SAVE_PATH="./"
MODE_NAME="model.ckpt"

def train(xData,yData,xTestData,yTestData):
    x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y-input')

    regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y=mnist_inference.inference(x,regularizer)
    global_step=tf.Variable(0,trainable=False)
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    #cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    #cross_entropy_mean=tf.reduce_mean(cross_entropy)
    mse=tf.reduce_mean(tf.abs(y_-y)) #归一化后损失函数
    #mse=tf.reduce_mean(tf.square(y_-y))
    #diff=(y[0][1]-y_[0][1])
    loss=mse
    #loss=mse+tf.add_n(tf.get_collection('losses'))#cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        TRAINNUM/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step) #learning_rate
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            start=(i*BATCH_SIZE)%(TRAINNUM-BATCH_SIZE)
            end=start+BATCH_SIZE
            xs=xData[start:end]
            ys=yData[start:end]
            #xs,ys=mnist.train.next_batch(BATCH_SIZE)

            _, loss_value, step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys}) #[train_op,loss,global_step]
            if i%1000==0:
                loss_value_n,b1,b2=sess.run([y[15],y_[15],mse],feed_dict={x:xTestData,y_:yTestData})
                #print("%g %g"%(loss_value_n,b1))
                print("After %d training step(s),loss on training batch is %g. %g %g " %(step,loss_value_n,b1,b2))
                #saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODE_NAME),global_step=global_step)
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 取每一列的最小值
    maxVals = dataSet.max(0) # 取每一列的最大值
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
def main(argv=None):
    #mnist = input_data.read_data_sets("/path/to/MINST_data/", one_hot=True)
    #train(mnist)
    csv_file=csv.reader(open('TrainSecond.csv','r'))
    csv_file1=csv.reader(open('TestSecond.csv','r'))
    """
    xdata=[]
    ydata=[]
    for i in csv_file:
        xdata.append(i[:9])
        ydata.append([i[-3]])
    npXData=np.array(xdata,dtype=float)
    npYData=np.array(ydata,dtype=float)
    normXData, xRange, _ = autoNorm(npXData)
    normYData, yRange, _ = autoNorm(npYData)
    xtestdata = []
    ytestdata = []
    print(yRange)
    for i in csv_file1:
        xtestdata.append(i[:9])
        ytestdata.append([i[-3]])
    npXtestData = np.array(xtestdata, dtype=float)
    npYtestData = np.array(ytestdata, dtype=float)
    normXtestData, _, _ = autoNorm(npXtestData)
    normYtestData, _, _ = autoNorm(npYtestData)
    print(npYData)
    """
    #take data from csv
    trainSet=[]
    testSet=[]
    for i in csv_file:
        trainSet.append(i)
    for i in  csv_file1:
        testSet.append(i)
    """
        #原数据处理
    trainSet1=np.array(trainSet)
    testSet1=np.array(testSet)
    normXData = trainSet1[:, :9]
    tmpData = []
    for i in trainSet1:
        tmpData.append([i[10]])
    normYData = np.array(tmpData)
    normXtestData = testSet1[:, :9]
    tmpData = []
    for i in testSet1:
        tmpData.append([i[10]])
    normYtestData = np.array(tmpData)
    """


    scaler = MinMaxScaler()
    #trainSet1=np.array(trainSet,dtype=float)
    scaler.fit(trainSet)
    trainSet1=scaler.transform(trainSet)
    testSet1=scaler.transform(testSet)
    #print(type(trainSet))
    normXData=trainSet1[:,:9]
    #print(normXData)
    #normYData=trainSet1[:,10:]
    tmpData=[]
    for i in trainSet1:
        tmpData.append([i[10]])
    normYData=np.array(tmpData)
    tmpData=[]

    normXtestData = testSet1[:,:9]
    #normYtestData = testSet1[:,10:]

    for i in testSet1:
        tmpData.append([i[10]])
    normYtestData=np.array(tmpData)

    #print('a')
    print(type(normYtestData))
    train(normXData,normYData,normXtestData,normYtestData) #0.201672. 0.367379  0.0244033  0.202092  0.36978  0.0441577
    aa=[[0,0,0,0,0,0,0,0,0,0.202603,0.363591,0.012551]]
    bb=[[0,0,0,0,0,0,0,0,0,0.202304,0.36403,0.0119415]]
    aa_1=scaler.inverse_transform(aa)
    bb_1=scaler.inverse_transform(bb)
    print('a')


    #sess = tf.Session()
    #saver = tf.train.Saver()
    #saver.restore(sess, MODEL_SAVE_PATH)
if __name__=='__main__':
    tf.app.run()