from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_inputs = 784


n_classes = 10  #or it could be called the number of outputs




x = tf.placeholder(tf.float32, shape=[None,n_inputs])
y = tf.placeholder(tf.float32, shape=[None,n_classes])





W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess = tf.InteractiveSession()

init = tf.global_variables_initializer()

sess.run(init)

y_ = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range (1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0],y: batch[1]})


correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels}))


