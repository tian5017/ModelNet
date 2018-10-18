import net_util as util
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier


lr_model_path = "model/LR-39-20180822.pkl"
adb_model_path = "model/adb_model.pkl"
xgb_model_path = "model/xgb_model_new.pkl"
mlp_model_path = "model/mlp_model.pkl"
vot_model_path = "model/vot_model.pkl"


def model_train_lr(flag, data):
    print("-------------逻辑回归模型--------------------")
    if flag:
        # lr = LogisticRegression(class_weight={0: 0.94, 1: 0.06}, n_jobs=4)
        # lr.fit(train_data[:, 2:], train_data[:, 1])
        # # 保存模型
        # joblib.dump(lr, lr_model_path)

        lr = LogisticRegressionCV(class_weight={0: 0.94, 1: 0.06}, cv=5, n_jobs=4)
        lr.fit(data[:, 2:], data[:, 1])
        # 保存模型
        joblib.dump(lr, lr_model_path)
    else:
        lr = joblib.load(lr_model_path)
    scores_accuracy = cross_val_score(lr, data[:, 2:], data[:, 1], cv=5, scoring="accuracy", n_jobs=4)
    print("LR准确率：", scores_accuracy, scores_accuracy.mean())
    scores_precision = cross_val_score(lr, data[:, 2:], data[:, 1], cv=5, scoring="precision", n_jobs=4)
    print("LR精确率：", scores_precision, scores_precision.mean())
    scores_recall = cross_val_score(lr, data[:, 2:], data[:, 1], cv=5, scoring="recall", n_jobs=4)
    print("LR召回率：", scores_recall, scores_recall.mean())


# 使用决策树构建Adaboost分类器
def model_train_adb(flag, data):
    print("-------------AdaBoost分类模型---------------------")
    if flag:
        # 基于决策树构建Adaboost分类器
        adb = AdaBoostClassifier(n_estimators=800)
        adb.fit(data[:, 3:], data[:, 2])
        # 保存模型
        joblib.dump(adb, adb_model_path)
    else:
        adb = joblib.load(adb_model_path)
    scores_accuracy = cross_val_score(adb, data[:, 3:], data[:, 2], cv=5, scoring="accuracy", n_jobs=4)
    print("ADB准确率：", scores_accuracy, scores_accuracy.mean())
    scores_precision = cross_val_score(adb, data[:, 3:], data[:, 2], cv=5, scoring="precision", n_jobs=4)
    print("ADB精确率：", scores_precision, scores_precision.mean())
    scores_recall = cross_val_score(adb, data[:, 3:], data[:, 2], cv=5, scoring="recall", n_jobs=4)
    print("ADB召回率：", scores_recall, scores_recall.mean())


# 构建XgBoost分类器
def model_train_xgb(flag, data):
    print("-------------XgBoost分类模型---------------------")
    if flag:
        xgb = XGBClassifier(max_depth=4, n_estimators=400, subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                            gamma=0.1, scale_pos_weight=0.0036, n_jobs=4)
        xgb.fit(data[:, 2:], data[:, 1])
        # 保存模型
        joblib.dump(xgb, xgb_model_path)
    else:
        xgb = joblib.load(xgb_model_path)
    scores_accuracy = cross_val_score(xgb, data[:, 2:], data[:, 1], cv=5, scoring="accuracy", n_jobs=4)
    print("XGB准确率：", scores_accuracy, scores_accuracy.mean())
    scores_precision = cross_val_score(xgb, data[:, 2:], data[:, 1], cv=5, scoring="precision", n_jobs=4)
    print("XGB精确率：", scores_precision, scores_precision.mean())
    scores_recall = cross_val_score(xgb, data[:, 2:], data[:, 1], cv=5, scoring="recall", n_jobs=4)
    print("XGB召回率：", scores_recall, scores_recall.mean())


# 构建神经网络分类器（MLP）
def model_train_mlp(flag, data):
    print("-------------神经网络分类模型---------------------")
    if flag:
        mlp = MLPClassifier(hidden_layer_sizes=(128, 256, 512), activation='relu', solver='adam', alpha=0.0025,
                            max_iter=600)
        mlp.fit(data[:, 3:], data[:, 2])
        # 保存模型
        joblib.dump(mlp, mlp_model_path)
    else:
        mlp = joblib.load(mlp_model_path)
    scores_accuracy = cross_val_score(mlp, data[:, 3:], data[:, 2], cv=5, scoring="accuracy", n_jobs=4)
    print("MLP准确率：", scores_accuracy, scores_accuracy.mean())
    scores_precision = cross_val_score(mlp, data[:, 3:], data[:, 2], cv=5, scoring="precision", n_jobs=4)
    print("MLP精确率：", scores_precision, scores_precision.mean())
    scores_recall = cross_val_score(mlp, data[:, 3:], data[:, 2], cv=5, scoring="recall", n_jobs=4)
    print("MLP召回率：", scores_recall, scores_recall.mean())




# 投票分类器
def model_train_vot(flag, data):
    print("-------------投票分类模型---------------------")
    if flag:
        lr = LogisticRegression(class_weight={0: 0.94, 1: 0.06})
        xgb = XGBClassifier(max_depth=4, n_estimators=400, subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                            gamma=0.1, scale_pos_weight=0.0036, n_jobs=4)
        adb = AdaBoostClassifier(n_estimators=800)
        mlp = MLPClassifier(hidden_layer_sizes=(128, 256, 512), activation='relu', solver='adam', alpha=0.0025,
                            max_iter=600)
        vot = VotingClassifier(estimators=[('lr', lr), ('adb', adb), ('mlp', mlp), ('xgb', xgb)], voting='soft',
                               weights=[1, 1, 2, 1.6], n_jobs=4)
        vot.fit(data[:, 3:], data[:, 2])
        # 保存模型
        joblib.dump(vot, vot_model_path)
    else:
        vot = joblib.load(vot_model_path)
    scores_accuracy = cross_val_score(vot, data[:, 3:], data[:, 2], cv=5, scoring="accuracy", n_jobs=4)
    print("VOT准确率：", scores_accuracy, scores_accuracy.mean())
    scores_precision = cross_val_score(vot, data[:, 3:], data[:, 2], cv=5, scoring="precision", n_jobs=4)
    print("VOT精确率：", scores_precision, scores_precision.mean())
    scores_recall = cross_val_score(vot, data[:, 3:], data[:, 2], cv=5, scoring="recall", n_jobs=4)
    print("VOT召回率：", scores_recall, scores_recall.mean())




# 在测试集上测试模型
def model_test(model_path, data_x, data_y):
    print("-------------测试分类模型：开始---------------------")
    # 加载模型
    model = joblib.load(model_path)
    pre_result = model.predict(data_x)
    print("预测结果：", pre_result)
    rea_result = data_y
    print("真实结果：", rea_result)
    # 获得混淆矩阵
    util.get_cm(rea_result, pre_result)
    # 绘制ROC曲线
    util.roc_line(rea_result, pre_result).show()
    print("-------------测试分类模型：结束---------------------")


# 评估各个参数的取值是否合理
def cv_attu(data):
    param_test = {
        'max_depth': [4, 6, 8],
        'n_estimators': [400, 600, 800],
        'scale_pos_weight': [0.0020, 0.0022, 0.0024, 0.0026]
    }
    xgb = XGBClassifier(max_depth=8, n_estimators=600, subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                        gamma=0.1, scale_pos_weight=0.0028, n_jobs=4)
    gsearch = GridSearchCV(estimator=xgb, param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(data[:, 2:], data[:, 1])
    print(gsearch.cv_results_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)


def print_model_flag(model_path, data_x):
    model = joblib.load(model_path)
    pre_result = model.predict(data_x)
    pre_result = pre_result.astype(int)
    print(pre_result)
    df = pd.read_csv("data/mj-data-12345-new.csv", encoding="GBK")
    df["new_flag"] = pre_result
    df.to_csv("data/mj-data-12345-model.csv", encoding="GBK", index=False)
    print("over")


if __name__ == "__main__":
    # cv_attu()

    # data_file = "data/train_data_0621_0720_new.csv"
    # data = pd.read_csv(data_file)
    # train_data, test_data = util.data_cut(data_file)

    # model_train_lr(True)

    # 测试集测试数据
    data = pd.read_csv("data/train-mj-data-12345.csv")
    test_data = data.values

    # model_test(lr_model_path, test_data[:, 2:], test_data[:, 1])

    print_model_flag(lr_model_path, test_data[:, 2:])



