import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
# 使用稳定性选择方法中的随机逻辑回归进行特征筛选
from sklearn.linear_model import RandomizedLogisticRegression as RLR


data_file = "data/train_data_0621_0720_new.csv"
data = pd.read_csv(data_file)

print("-------------进行特征筛选：开始---------------------")
x = data.iloc[:, 2:].as_matrix()
y = data.iloc[:, 1].as_matrix()
# 建立随机逻辑回归模型，筛选变量
rlr = RLR()
# 训练模型
rlr.fit(x, y)
# 获取各个特征的分数
cols = np.array(data.columns[2:])
print("各个特征得分：", cols, rlr.scores_)
out_sup = rlr.get_support()
out_sup_arr = [x for x in range(len(out_sup)) if out_sup[x] == True]
print("通过随机逻辑回归模型筛选特征结束!")
real_feature = cols[out_sup_arr]
print("有效特征为：", real_feature)
print("-------------进行特征筛选：结束---------------------")

print("-------------创建逻辑回归模型：开始---------------------")
# 用筛选好的特征重新组建输入数据x
x = data[real_feature].as_matrix()
# 建立逻辑回归模型
lr = LR()
# 用筛选过后的特征来训练模型
lr.fit(x, y)
print("逻辑回归模型训练结束!")
# 得到模型的平均正确率
sc = lr.score(x, y)
print("模型的平均正确率为：", sc)
print("-------------创建逻辑回归模型：结束---------------------")
