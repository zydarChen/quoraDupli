# -*- coding: utf-8 -*-

from preprocess import *
import numpy as np
from tqdm import tqdm
from sklearn import svm

# df = pd.read_pickle('data\questionVector.pickle')  # 全数据集
df = pd.read_pickle('data\questionVector2k.pickle')  # head(2000)
# 统计信息
# stats(df)

# 生成训练集测试集
def createTrain(df):
    df0 = df[df['is_duplicate'] == 0]
    df1 = df[df['is_duplicate'] == 1]
    # 训练集：测试集 = 5:1
    train0 = df0.sample(n=int(df0.shape[0]*0.8))
    train1 = df1.sample(n=int(df1.shape[0]*0.8))
    test0List = [i for i in df0.index if i not in train0.index]
    test1List = [i for i in df1.index if i not in train1.index]
    # test0 = df.iloc[test0List, :]
    # test1 = df.iloc[test1List, :]
    test0 = df0[df0.index.isin(test0List)]
    test1 = df1[df1.index.isin(test1List)]

    train = train0.append(train1)
    test = test0.append(test1)

    return train, test

def featureAndLabel(df):
    trainX = []
    trainY = []
    for i in tqdm(range(df.shape[0])):
        row = df.iloc[i]
        features = np.append(row['q1Vector'], row['q2Vector'])
        label = row['is_duplicate']
        trainX.append(features)
        trainY.append(label)
    return trainX, trainY

train, test = createTrain(df)
trainX, trainY = featureAndLabel(train)
testX, testY = featureAndLabel(test)
clf = svm.SVC()
clf.fit(trainX, trainY)
print list(clf.predict(trainX)) == trainY
