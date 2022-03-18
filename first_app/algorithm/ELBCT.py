# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:00:21 2022

@author: 百城之王
"""

from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve, accuracy_score
import time
import random
import pickle
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import mpl
import warnings
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score
import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.style.use('seaborn')


mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
color = sns.color_palette()
sns.set_style('darkgrid')
warnings.filterwarnings('ignore')
# from random import sample


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


def plot_pr(labels, predictions):
    plt.figure(figsize=(5, 5))  # 创建图表1
    plt.title('Precision/Recall Curve')  # give plot a title
    plt.xlabel('Precision')  # make axis labels
    plt.ylabel('Recall')
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    plt.plot(precision, recall)
    plt.show()
    return thresholds


def plot_roc(test_labels, y_proba):

    fpr, tpr, thresholds = roc_curve(test_labels, y_proba)
    roc_auc = auc(fpr, tpr)

    # 确定最佳阈值

    right_index = (tpr + (1 - fpr) - 1)

    yuzhi = max(right_index)
    index = list(right_index).index([max(right_index)])
    tpr_val = list(tpr)[index]
    fpr_val = list(fpr)[index]

    # 绘制roc曲线图
    plt.subplots(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


"""导入数据集"""
train = pd.read_csv(
    'D:\\Captain America\\开发项目、科技竞赛\\2020服务外包\\上交材料\\基于差异度的级联模型\\train_del.csv', encoding="gbk")
test = pd.read_csv(
    'D:\\Captain America\\开发项目、科技竞赛\\2020服务外包\\上交材料\\基于差异度的级联模型\\test.csv', encoding="gbk")


train_labels = train['emd_lable2']
train.drop('emd_lable2', axis=1, inplace=True)

test_labels = test['emd_lable2']
test.drop('emd_lable2', axis=1, inplace=True)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
train_labels.reset_index(drop=True, inplace=True)
test_labels.reset_index(drop=True, inplace=True)
train, vaild, train_labels, vaild_labels = train_test_split(train, train_labels,
                                                            test_size=0.2,
                                                            random_state=1)
train.reset_index(drop=True, inplace=True)
vaild.reset_index(drop=True, inplace=True)
train_labels.reset_index(drop=True, inplace=True)
vaild_labels.reset_index(drop=True, inplace=True)


print('Start training...')

"""选择阈值"""


def choose_thresholds(test, all_models, f):

    sum_prob = np.zeros((test.shape[0], 2))
    sum_prob = sum_prob +\
        all_models.predict_proba(test)

    sum_prob = sum_prob[:, 1]

    med_threshold = [0.01*i for i in range(100)]
    med_threshold.reverse()

    for i in range(len(med_threshold)):
        pred = []
        for k in range(sum_prob.shape[0]):
            if sum_prob[k] > med_threshold[i]:
                pred.append(1)
            else:
                pred.append(0)

        if sum(pred)/sum_prob.shape[0] > f:
            threshold = med_threshold[i]
            break

    return threshold, pred


clfs = [
    XGBClassifier(max_depth=5,
                  min_child_weights=8,
                  n_estimators=77,
                  min_samples_split=0.06601,
                  gamma=0.01,
                  subsample=0.8,
                  colsample_bytree=0.8,
                  objective='binary:logistic',
                  nthread=8,
                  scale_pos_weight=1,
                  seed=27
                  ),

    RandomForestClassifier(max_depth=39,
                           max_features=0.7,
                           n_estimators=598,
                           min_samples_split=80,
                           random_state=2021,
                           class_weight="balanced"
                           ),

    CatBoostClassifier(iterations=904,
                       learning_rate=0.012,
                       max_depth=9,
                       verbose=100,
                       early_stopping_rounds=500,
                       eval_metric='AUC',
                       random_state=2021),

    LGBMClassifier(
        boosting_type='gbdt',  # 提升树的类型 gbdt,dart,goss,rf
        objective='regression',
        learning_rate=0.034,  # 学习率
        num_leaves=34,  # 树的最大叶子数
        max_depth=9,  # 最大树的深度
        subsample=0.8,  # 子样本频率
        n_estimators=102,  # 拟合的树的棵树，相当于训练轮数
        min_child_weight=0.002,  # 分支结点的最小权重
        min_child_samples=22,
        class_weight="balanced",
        random_state=2021)
]


def Degree_of_difference(pred1, pred2):
    sum = 0
    for i in range(len(pred1)):
        sum += pow((pred1[i]-pred2[i]), 2)
    return math.sqrt(sum)


def Weight_calculation(pred, labels):
    n = len(pred)
    w = 0
    for i in range(n):
        if pred[i] != labels[i]:
            w += 1/n
    return w


"""BalanceCascade模型构建"""
T = 10  # 从N中采样的子集个数
all_models = []
all_theta = []
# 20180606 201806061817
seed = time.time()
Ran = random.Random(425)


def ChangeClf(rfc):
    neg_loc = train_labels.loc[(train_labels == 0)].index
    neg_loc = neg_loc.tolist()
    neg_train = train.iloc[neg_loc]
    neg_train_labels = train_labels.iloc[neg_loc]
    neg_train.reset_index(drop=True, inplace=True)
    neg_train_labels.reset_index(drop=True, inplace=True)
    neg_loc = list(neg_train.index)

    pos_loc = train_labels.loc[(train_labels == 1)].index
    pos_loc = pos_loc.tolist()
    pos_train = train.iloc[pos_loc]
    pos_train_labels = train_labels.iloc[pos_loc]
    pos_train.reset_index(drop=True, inplace=True)
    pos_train_labels.reset_index(drop=True, inplace=True)
    pos_loc = list(pos_train.index)

    # f = pow(len(pos_loc)/len(neg_loc),1/(T-1))
    f = 0.6
    pos = sum(train_labels.tolist())
    for i in range(T):
        ran_neg_loca = Ran.sample(neg_loc, pos)
        ran_neg_train = neg_train.iloc[ran_neg_loca]
        ran_neg_train_labels = neg_train_labels.iloc[ran_neg_loca]
        ran_neg_train.reset_index(drop=True, inplace=True)
        ran_neg_train_labels.reset_index(drop=True, inplace=True)

        med_train = pd.concat([ran_neg_train, pos_train], axis=0)
        med_train_labels = pd.concat(
            [ran_neg_train_labels, pos_train_labels], axis=0)
        med_train.reset_index(drop=True, inplace=True)
        med_train_labels.reset_index(drop=True, inplace=True)

        models = rfc.fit(med_train, med_train_labels)
        threshold, pred = choose_thresholds(neg_train, models, f)
        all_models.append(pickle.dumps(models))
        zero_loc = [neg_loc[x] for x in range(len(pred)) if pred[x] == 0]
        for m in range(len(zero_loc)):
            neg_loc.remove(zero_loc[m])

        neg_train = neg_train.iloc[neg_loc]
        neg_train_labels = neg_train_labels.iloc[neg_loc]
        neg_train.reset_index(drop=True, inplace=True)
        neg_train_labels.reset_index(drop=True, inplace=True)
        neg_loc = list(neg_train.index)
        if len(neg_loc) < pos:
            break


for clf in clfs:
    ChangeClf(clf)
models_auc = []
for model in all_models:
    model = pickle.loads(model)
    pred = model.predict_proba(vaild)[:, 1]
    pre = model.predict(vaild)
    auc = roc_auc_score(vaild_labels, pred)
    print("=============  Single_Model  ================")
    print("auc:", auc)
    # print('recall:',recall_score(vaild_labels,pre))
    models_auc.append({'model': model, 'auc': auc})

models_auc.sort(key=lambda x: x['auc'], reverse=True)
models_diff = []
N = 23
Nauc = []
Nx = []
all_models = []
for i in range(N):
    model = models_auc[i]['model']
    pred = model.predict(vaild)
    auc = roc_auc_score(vaild_labels, pred)
    w = Weight_calculation(pred, vaild_labels.values)
    all_models.append({'model': model, 'pred': pred,
                      'weight': 0.5*math.log((1-w)/w)})
a = []
x = [[], [], [], []]
y = [[], [], [], []]

for i, item in enumerate(all_models):
    p = item['model'].predict_proba(vaild)
    auc = roc_auc_score(vaild_labels, p[:, 1])


h = all_models[0]
all_models.remove(all_models[0])


def sgn(x):
    if x > 0:
        return 1
    else:
        return 0


# 记录融合过程中模型顺序和权重
modelsStrategy = []
weightStrategy = []
modelsStrategy.append(h['model'])
pred = h['pred']
pred_prob = h['model'].predict_proba(vaild)
weightStrategy.append(h['weight'])
auc = [roc_auc_score(vaild_labels, pred_prob[:, 1])]
name = []


while len(all_models):
    index = 0
    maxdiff = 0
    for i in range(len(all_models)):
        diff = Degree_of_difference(h['pred'], all_models[i]['pred'])
        if maxdiff < diff:
            maxdiff = diff
            index = i
    w1 = weightStrategy[-1]
    w2 = all_models[index]['weight']
    prob1 = pred_prob
    prob2 = all_models[index]['model'].predict_proba(vaild)
    pred_new = []
    pred_p_new = []
    for i in range(len(h['pred'])):
        pred_new.append(sgn((w1*prob1[i][1]+w2*prob2[i][1])/(w1+w2)))
        pred_p_new.append([0, (w1*prob1[i][1]+w2*prob2[i][1])/(w1+w2)])
    pred_p_new = np.array(pred_p_new)
    if roc_auc_score(vaild_labels, pred_prob[:, 1]) < roc_auc_score(vaild_labels, pred_p_new[:, 1]):
        # ==================================================
        auc.append(roc_auc_score(vaild_labels, pred_p_new[:, 1]))

        modelsStrategy.append(all_models[index]['model'])
        pred = pred_new
        pred_prob = pred_p_new
        weightStrategy.append(w2)
        w = Weight_calculation(pred, vaild_labels.values)
        weightStrategy.append(w)
    all_models.remove(all_models[index])


test_pred = modelsStrategy[0].predict_proba(test)
auc = []
for i in range(len(modelsStrategy)-1):
    w1 = weightStrategy[i*2]
    w2 = weightStrategy[i*2+1]
    new_pred = modelsStrategy[i+1].predict_proba(test)
    for i in range(len(new_pred)):
        test_pred[i][1] = (w1*test_pred[i][1]+w2*new_pred[i][1])/(w1+w2)


test_pred_ = []
for i in range(len(test_pred)):
    test_pred_.append(sgn(test_pred[i][1]-0.5))


# 划分数据集
def chose_fscore(test_lables, pred_prob):
    precision, recall, thresholds = precision_recall_curve(
        test_lables, pred_prob[:, 1])

    fscore = []
    for i in range(len(precision)):
        score = 5 * precision[i] * recall[i] / (4 * precision[i] + recall[i])
        fscore.append(score)

    return fscore, thresholds, precision, recall


def the_pred(pred_prob, the):
    test_pred = []
    pred_prob = pred_prob[:, 1]
    for i in range(len(pred_prob)):
        if pred_prob[i] > the:
            test_pred.append(1)
        else:
            test_pred.append(0)

    return test_pred


print("============  ELBCT_Model  ==============")
fscore, the, pre, rec = chose_fscore(test_labels, test_pred)
max_fscore = max(fscore)
max_the = the[fscore.index(max(fscore))]
max_pre = pre[fscore.index(max(fscore))]
max_rec = rec[fscore.index(max(fscore))]
Nauc.append(roc_auc_score(test_labels, test_pred[:, 1]))
print("AUC", roc_auc_score(test_labels, test_pred[:, 1]))
print('fscore:', max_fscore)
print('recall:', max_rec)
print('precision:', max_pre)
print('accuracy:', accuracy_score(test_labels, the_pred(test_pred, max_the)))
