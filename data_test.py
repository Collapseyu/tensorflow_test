from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/path/to/MINST_data/",one_hot=True)

print("Traing data size: ",mnist.train.num_examples)