import time
import tensorflow as tf

import mnist_inference
import mnist_train

def evaluate(xdata,ydata):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed={x:xdata,y_:ydata}
        y=mnist_inference.inference(x,None)

