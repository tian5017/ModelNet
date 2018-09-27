import tensorflow as tf
import numpy as np
import net_util as util


class Net:
    def __init__(self):
        # 输入节点为15（15个特征）,输出节点为1（二分类）
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 15])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        # 隐藏层1节点为32
        self.w1 = tf.Variable(tf.truncated_normal(shape=[15, 32], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([32]))
        # 隐藏层2节点为64
        self.w2 = tf.Variable(tf.truncated_normal(shape=[32, 64], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([64]))
        # 输出层节点为1
        self.w3 = tf.Variable(tf.truncated_normal(shape=[64, 1], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([1]))

        self.output = None
        self.error = None
        self.optimizer = None
        self.accuracy = None

    def forward(self):
        y = tf.nn.leaky_relu(tf.add(tf.matmul(self.x, self.w1), self.b1))
        y = tf.nn.leaky_relu(tf.add(tf.matmul(y, self.w2), self.b2))
        self.output = tf.nn.sigmoid(tf.add(tf.matmul(y, self.w3), self.b3))

    def backward(self):
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.output)))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)


if __name__ == '__main__':
    net = Net()
    net.forward()
    net.backward()
    net.init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(net.init)
        # 获取训练数据，测试数据
        train_data, test_data = util.data_cut("data/train_data_0621_0720.csv")
        if False:
            saver = tf.train.Saver(max_to_keep=1)
            # 训练
            epochs = 100
            while epochs >= 0:
                epochs -= 1
                for i in range(len(train_data)):
                    train_x, train_y = train_data[i, 2:], train_data[i, 1]
                    train_x = np.reshape(train_x, [1, 15])
                    train_y = np.reshape(train_y, [1, 1])
                    _error, _output = sess.run([net.loss, net.output], feed_dict={net.x: train_x, net.y: train_y})
                    if epochs % 10 == 0:
                        print("error:", _error)
            saver.save(sess, "model/my-model")
        else:
            # 测试
            saver = tf.train.import_meta_graph("model/my-model.meta")
            saver.restore(sess, tf.train.latest_checkpoint("model/"))
            _predict = []
            test_x, test_y = test_data[:, 2:], test_data[:, 1]
            test_x = np.reshape(test_x, [len(test_x), 15])
            test_y = np.reshape(test_y, [len(test_y), 1])
            _output = sess.run([net.output], feed_dict={net.x: test_x, net.y: test_y})
            _output = np.reshape(np.array(_output), len(test_y))
            _output[_output >= 0.5] = 1
            _output[_output < 0.5] = 0
            _output = _output.astype(np.int)
            # 预测结果
            test_y = test_y.reshape(len(test_y))
            # 获得混淆矩阵
            util.get_cm(test_y, _output)
            # 绘制ROC曲线
            util.roc_line(test_y, _output).show()