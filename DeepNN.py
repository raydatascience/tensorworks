import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_inputs = 784

n_node_hl1 = 1000
n_node_hl2 = 1000
n_node_hl3 = 1000
n_node_hl4 = 1000
n_node_hl5 = 1000
n_node_hl6 = 1000
n_node_hl7 = 1000
n_node_hl8 = 1000

n_classes = 10  #or it could be called the number of outputs

batch_size = 100


x = tf.placeholder('float32', shape=[None,n_inputs])
y = tf.placeholder(tf.float32, shape=[None,n_classes])


def DNN_model(data):    # this is the deep NN model with multiple layers

    h1_layer = {'weights':tf.Variable(tf.random_normal([n_inputs,n_node_hl1])),
                'biases':tf.Variable(tf.random_normal([n_node_hl1]))}

    h2_layer = {'weights':tf.Variable(tf.random_normal([n_node_hl1,n_node_hl2])),
                'biases':tf.Variable(tf.random_normal([n_node_hl2]))}


    h3_layer = {'weights':tf.Variable(tf.random_normal([n_node_hl2,n_node_hl3])),
                'biases':tf.Variable(tf.random_normal([n_node_hl3]))}

    h4_layer = {'weights':tf.Variable(tf.random_normal([n_node_hl3,n_node_hl4])),
                'biases':tf.Variable(tf.random_normal([n_node_hl4]))}

    h5_layer = {'weights':tf.Variable(tf.random_normal([n_node_hl4,n_node_hl5])),
                'biases':tf.Variable(tf.random_normal([n_node_hl5]))}

    h6_layer = {'weights':tf.Variable(tf.random_normal([n_node_hl5,n_node_hl6])),
                'biases':tf.Variable(tf.random_normal([n_node_hl6]))}

    h7_layer = {'weights':tf.Variable(tf.random_normal([n_node_hl6,n_node_hl7])),
                'biases':tf.Variable(tf.random_normal([n_node_hl7]))}

    h8_layer = {'weights':tf.Variable(tf.random_normal([n_node_hl7,n_node_hl8])),
                'biases':tf.Variable(tf.random_normal([n_node_hl8]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_node_hl8,n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes]))}

    # the model:  (input (x) * weights (W)) + biases (b)

    L1 = tf.add(tf.matmul(data, h1_layer['weights']), h1_layer['biases'])
    L1 = tf.nn.relu(L1)

    L2 = tf.add(tf.matmul(L1, h2_layer['weights']), h2_layer['biases'])
    L2 = tf.nn.relu(L2)

    L3 = tf.add(tf.matmul(L2, h3_layer['weights']), h3_layer['biases'])
    L3 = tf.nn.relu(L3)

    L4 = tf.add(tf.matmul(L3, h4_layer['weights']), h4_layer['biases'])
    L4 = tf.nn.relu(L4)

    L5 = tf.add(tf.matmul(L4, h5_layer['weights']), h5_layer['biases'])
    L5 = tf.nn.relu(L5)

    L6 = tf.add(tf.matmul(L5, h6_layer['weights']), h6_layer['biases'])
    L6 = tf.nn.relu(L6)

    L7 = tf.add(tf.matmul(L6, h7_layer['weights']), h7_layer['biases'])
    L7 = tf.nn.relu(L7)

    L8 = tf.add(tf.matmul(L7, h8_layer['weights']), h8_layer['biases'])
    L8 = tf.nn.relu(L8)


    output = tf.matmul(L8,output_layer['weights'])+ output_layer['biases']
    return output


def train_DNN(x):
    prediction = DNN_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # for AdamOptimizer the learning_rate = 0.001 is left alone
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    n_epochs = 20

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

# trainig starts ...

        for epoch in range (n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                # in reality we must mak eour own data reader like next_batch
                _,epoch_cost = sess.run([optimizer,cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += epoch_cost
            print('Epoch ',epoch,' completed out of ', n_epochs, ' Loss: ', epoch_loss)

# training end.

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float32'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_DNN(x)


