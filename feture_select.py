# 特征筛选
import pandas as pd
import numpy as np
from sklearn.linear_model import RandomizedLogisticRegression as RLR
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
import net_util as util
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

# file_path = "data/train_data_0621_0720_new.csv"
file_path = "data/all_feature.csv"
# 获取数据集
def get_data():
    data = pd.read_csv(file_path)
    columns = list(data.columns)
    data = data.as_matrix()
    data_x = data[:, 3:]
    data_y = data[:, 2]
    return data_x, data_y, columns[2:]


# 使用稳定性选择方法中的"随机逻辑回归"算法进行特征筛选
def feture_select_RLR():
    data_x, data_y, names = get_data()
    rlr = RLR()
    rlr.fit(data_x, data_y)
    return sorted(zip(names, map(lambda x: round(x, 4), rlr.scores_)), key=lambda x: x[1], reverse=True)


# 使用递归特征消除方法进行特征筛选
def feture_select_RFE():
    data_x, data_y, names = get_data()
    lr = LR()
    rfe = RFE(lr)
    rfe.fit(data_x, data_y)
    return sorted(zip(names, map(float, rfe.ranking_)), key=lambda x: x[1], reverse=True)


# 基于随机森林的特征重要度度量方法
def feture_select_RFR():
    data_x, data_y, names = get_data()
    rfr = RandomForestRegressor()
    rfr.fit(data_x, data_y)
    return sorted(zip(names, map(lambda x: round(x, 4), rfr.feature_importances_)), key=lambda x: x[1], reverse=True)


# 基于随机森林的特征重要度度量方法
def feture_select_ADB():
    data_x, data_y, names = get_data()
    adb = AdaBoostClassifier(n_estimators=100)
    adb.fit(data_x, data_y)
    return sorted(zip(names, map(lambda x: round(x, 4), adb.feature_importances_)), key=lambda x: x[1], reverse=True)


# 验证曲线,评估参数和指标的关系(1000)
def vali_curve():
    data_x, data_y, names = get_data()
    param_range = [600, 800, 1000, 1400, 1600]
    adb = AdaBoostClassifier()
    train_scores, test_scores = validation_curve(estimator=adb, X=data_x, y=data_y, param_name="n_estimators", param_range=param_range, cv=5, scoring='accuracy')
    # 统计结果
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
    plt.xlabel("number of tree")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    r_d = feture_select_ADB()
    x_label = []
    y_data = []
    for item in r_d:
        x_label.append(item[0])
        y_data.append(item[1])

    util.paint_bar(x_label, y_data, "特征评分", "特征名称")

