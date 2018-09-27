import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# 初始化权重
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# 初始化偏置
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 卷积操作
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 最大池化操作
def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class CnnNet:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        # 第一层卷积
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])
        # 第二层卷积
        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])
        # 全连接层
        self.W_fc = weight_variable([7 * 7 * 64, 1024])
        self.b_fc = bias_variable([1024])
        # 输出层
        self.W_op = weight_variable([1024, 10])
        self.b_op = bias_variable([10])

    def forward(self):
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
        h_pool1 = max_pool(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = max_pool(h_conv2)

        h_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc = tf.nn.relu(tf.matmul(h_flat, self.W_fc) + self.b_fc)

        # DropOut
        h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

        self.output = tf.nn.softmax(tf.matmul(h_fc_drop, self.W_op) + self.b_op)

    def backward(self):
        self.loss = -tf.reduce_sum(self.y * tf.log(self.output))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def validate(self):
        y = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_sum(tf.cast(y, dtype=tf.float32))


if __name__ == '__main__':
    net = CnnNet()
    net.forward()
    net.backward()
    net.validate()
    net.init = tf.global_variables_initializer()

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.Session() as sess:
        sess.run(net.init)
        if False:
            saver = tf.train.Saver(max_to_keep=1)
            for i in range(10000):
                train_x, train_y = mnist.train.next_batch(100)
                _loss, _ = sess.run([net.loss, net.optimizer], feed_dict={net.x: train_x, net.y: train_y,
                                                                          net.keep_prob: 1.0})
                if i % 100 == 0:
                    print("error:", _loss)
                    test_x, test_y = mnist.test.next_batch(100)
                    _accuracy = sess.run(net.accuracy, feed_dict={net.x: test_x, net.y: test_y, net.keep_prob: 1.0})
                    print("accuracy:", _accuracy)
                if _loss < 0.1:
                    break
            saver.save(sess, "model/test/my-model")
        else:
            saver = tf.train.Saver()
            saver.restore(sess, "model/test/my-model")
            test_x, test_y = mnist.test.next_batch(20)
            gs = gridspec.GridSpec(int(len(test_x) / 5), 5, wspace=0.5)
            print("真实值：\n", np.reshape(np.argmax(test_y, 1), [gs._nrows, gs._ncols]))
            _output, _acc = sess.run([net.output, net.accuracy], feed_dict={net.x: test_x, net.y: test_y,
                                                                            net.keep_prob: 1.0})
            _output = sess.run(tf.round(_output))
            print("预测值：\n", np.reshape(np.argmax(_output, 1), [gs._nrows, gs._ncols]))
            print("正确率：", _acc / len(test_y))
            for i in range(int(len(test_x) / 5)):
                tmp = i * 4
                for j in range(5):
                    ax = plt.subplot(gs[i, j])
                    ax.imshow(np.reshape(test_x[i + j + tmp, :], [28, 28]), cmap='gray')
            plt.show()

