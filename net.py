# 神经网络
import pandas as pd
from keras.models import Sequential
# 神经网络层函数、激活函数
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras import losses, metrics, activations
import numpy as np
import net_util as util
# 随机函数
from random import shuffle
from imblearn.combine import SMOTEENN


# 训练数据路径
file_path = "data/0621-0720/all_feature.csv"
data = pd.read_csv(file_path)
# 训练好的模型存储路径
net_file = "model/net.model"
# 获取切分之后的数据集
train_data, test_data = util.data_cut(file_path)


# 构建神经网络模型
# 设定LM神经网络的输入节点数为15，输出节点数为1
# 第一隐层节点数为32，第二隐层节点数为64
class Net:
    def __init__(self):
        # 建立神经网络
        self.net = Sequential()
        # 添加输入层到第一隐藏层的链接, 隐藏层使用Relu激活函数
        self.net.add(Dense(128, input_shape=(39,)))
        self.net.add(LeakyReLU())
        self.net.add(Dropout(0.25))
        # 添加第一隐藏层到第二隐藏层的链接
        self.net.add(Dense(256, input_shape=(128,)))
        self.net.add(LeakyReLU())
        self.net.add(Dropout(0.25))
        # 添加第二隐藏层到输出层的链接, 输出层使用sigmoid激活函数
        self.net.add(Dense(1, input_shape=(256,), activation=activations.sigmoid))
        # 编译模型，使用Adam方法求解
        self.net.compile(optimizer="adam", loss=losses.mean_squared_error, metrics=[metrics.binary_accuracy])

    def train_model(self):
        # 训练模型
        self.net.fit(train_data[:, 3:], train_data[:, 2], class_weight={0: 23, 1: 1}, epochs=10, validation_split=0.2)
        # 保存训练完成的模型
        self.net.save_weights(net_file)

    def test_model(self, data_x, data_y):
        # 加载训练完成的模型
        self.net.load_weights(net_file)
        # 预测结果
        predict_result = self.net.predict_classes(data_x, batch_size=1).reshape(len(data_x))
        # 获得混淆矩阵
        util.get_cm(data_y, predict_result)
        # 绘制ROC曲线
        util.roc_line(data_y, predict_result).show()


    # 为所有数据输出预测结果
    def print_model_flag(self, data_x):
        # 加载训练完成的模型
        self.net.load_weights(net_file)
        # 预测结果
        predict_result = self.net.predict_classes(data_x, batch_size=1).reshape(len(data_x))
        df = pd.read_csv("data/0721-0731/data-0721-0731-model.csv", encoding="GBK")
        df["MLP"] = predict_result
        df.to_csv("data/0721-0731/data-0721-0731-model-new.csv", encoding="GBK", index=False)
        print("over")


def data_smot():
    sm = SMOTEENN()
    x_res, y_res = sm.fit_sample(test_data[:, 3:], test_data[:, 2])
    print(len(y_res[y_res == 1]))
    print(len(y_res[y_res == 0]))
    y_res = np.reshape(y_res, [-1, 1])
    x_y = np.hstack((x_res, y_res))
    col = list(data.columns[3:])
    col.append(data.columns[2])
    val = x_y
    df = pd.DataFrame(data=val, columns=col)
    df.to_csv("data/new_data_test.csv", index=False)
    print("over")



if __name__ == "__main__":
    data = pd.read_csv("data/0721-0731/0721-0731_test.csv")
    test_data = data.values
    shuffle(test_data)

    net = Net()
    # net.train_model()
    net.test_model(test_data[:, 3:], test_data[:, 2])
    # net.print_model_flag(data[:, 3:])