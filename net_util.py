import numpy as np
import pandas as pd
# 随机函数
from random import shuffle
# 导入混淆矩阵函数, ROC曲线函数
from sklearn.metrics import confusion_matrix, roc_curve
# 导入作图库
import matplotlib.pyplot as plt
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False


# 混淆矩阵，yt 真实值，yp 预测值
def get_cm(yt, yp):
    # 混淆矩阵
    cm = confusion_matrix(yt, yp)
    print("混淆矩阵：", cm)
    print("准确率为：", (cm[0, 0] + cm[1, 1]) / np.sum(cm))
    print("精确率为：", cm[1, 1] / (cm[0, 1] + cm[1, 1]))
    print("召回率为：", cm[1, 1] / (cm[1, 0] + cm[1, 1]))


# 绘制ROC曲线, yt 真实值，yp 预测值
def roc_line(yt, yp):
    fpr, tpr, thresholds = roc_curve(yt, yp, pos_label=1)
    # 作出ROC曲线
    plt.plot(fpr, tpr, linewidth=2, label="ROC曲线", color="red")
    # 坐标轴标签
    plt.xlabel("假正例率")
    plt.ylabel("真正例率")
    # 边界范围
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    # 图例
    plt.legend(loc=4)
    return plt


# 数据集划分
def data_cut(file_path):
    data = pd.read_csv(file_path)
    data = data.values
    # 随机打乱数据
    shuffle(data)
    # 设置训练数据比例
    p = 0.8
    train_data = data[:int(len(data)*p), :]
    test_data = data[int(len(data) * p):, :]
    return train_data, test_data



# 通过matplotlib绘制条形图（竖向）
# x_label : x轴刻度
# data : 数据
# x_title，y_title : x，y轴的标签
def paint_bar(x_label, data, x_title, y_title):
    # 1.确定绘图范围，由于只需要画一张图，所以我们将整张白纸作为绘图的范围(第一行第一列第一幅图)
    ax = plt.subplot(1, 1, 1)
    # 2.整理我们准备绘制的数据
    data = np.array(data)
    # 3.准备绘制条形图，绘制条形图需要确定如下要素：绘制的条形宽度、绘制的条形位置(中心)、条形图的高度（数据值）
    width = 0.4
    x_bar = np.arange(len(x_label))
    # 4.绘制条形图，left:条形中点横坐标、height:条形高度、width:条形宽度，默认值0.8
    rect = ax.bar(left=x_bar, height=data, width=width, color="b", alpha=0.8)
    # 5.向各条形上添加数据标签
    for rec in rect:
        x = rec.get_x()
        width = rec.get_width()
        height = rec.get_height()
        ax.text(x + width / 2, height, str(height), horizontalalignment="center", verticalalignment="bottom")
    # 6.绘制x，y坐标轴刻度及标签，标题
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    plt.xticks(x_bar, x_label, rotation=90)
    plt.show()



# 通过matplotlib绘制条形图（横向）
# y_label : y轴刻度
# data : 数据
# x_title，y_title : x，y轴的标签
def paint_barh(y_label, data, x_title, y_title):
    # 1.确定绘图范围，由于只需要画一张图，所以我们将整张白纸作为绘图的范围(第一行第一列第一幅图)
    ax = plt.subplot(1, 1, 1)
    # 2.整理我们准备绘制的数据
    data = np.array(data)
    # 3.准备绘制条形图，绘制条形图需要确定如下要素：绘制的条形宽度、绘制的条形位置(中心)、条形图的高度（数据值）
    height = 0.4
    y_bar = np.arange(len(y_label))
    # 4.绘制条形图，y:条形中点纵坐标、height:条形高度、width:条形宽度
    rect = ax.barh(y=y_bar, width=data, height=height, color="b", alpha=0.8)
    # 5.向各条形上添加数据标签
    for i in range(len(rect)):
        width = rect[i].get_width()
        ax.text(width, i, str(width), horizontalalignment="left", verticalalignment="center")
    # 6.绘制x，y坐标轴刻度及标签，标题
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    plt.yticks(y_bar, y_label, rotation=0)
    plt.show()


if __name__ == "__main__":
    pass
