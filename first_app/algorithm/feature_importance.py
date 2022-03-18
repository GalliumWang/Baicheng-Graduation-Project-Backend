from pytest import Testdir
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import true
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
import matplotlib.pyplot as plt
plt.style.use('seaborn')
warnings.filterwarnings("ignore")


# """导入数据集"""
# user_train = pd.read_csv(
#     "D:\\Captain America\\大学高等教育\\毕业设计\\算法\\数据集\\train_del.csv", encoding="gbk")
# user_test = pd.read_csv(
#     "D:\\Captain America\\大学高等教育\\毕业设计\\算法\\数据集\\test.csv", encoding="gbk")
# 将系统用户上传的数据作为'user_train'


def XGBoost_function(train, test):
    train_labels = train['emd_lable2']
    train.drop('emd_lable2', axis=1, inplace=True)

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    train_labels.reset_index(drop=True, inplace=True)
    train, vaild, train_labels, vaild_labels = train_test_split(
        train, train_labels, test_size=0.2, random_state=1)
    train.reset_index(drop=True, inplace=True)
    train_labels.reset_index(drop=True, inplace=True)

    print('Start training...')

    model = XGBClassifier(max_depth=5,
                          min_child_weights=8,
                          n_estimators=77,
                          min_samples_split=0.06601,
                          gamma=0.01,
                          subsample=0.8,
                          colsample_bytree=0.8,
                          objective='binary:logistic',
                          nthread=8,
                          scale_pos_weight=1,
                          seed=27,
                          early_stopping_rounds=100,
                          verbosity=0
                          )
    model.fit(train, train_labels)
    # # 解决xgb中utf-8不能编码问题
    # # 新版save_raw()开头多4个字符'binf'
    # model_modify = model.save_raw()[4:]
    # def myfun(self=None):
    #     return model_modify
    # model.save_raw = myfun

    result_data = pd.DataFrame()
    temp = list(user_test[0:1])
    temp.pop(1)
    t = np.array(temp)
    result_data['feature_name'] = t
    result_data['value'] = model.feature_importances_
    return result_data

    # 绘制特征重要性（bar chart）
    # xgb.plot_importance(model, height=.5,
    #                     max_num_features=20,
    #                     show_values=true)
    # plt.show()


# print(XGBoost_function(user_train, user_test))
