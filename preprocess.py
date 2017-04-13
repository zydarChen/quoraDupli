# -*- coding: utf-8 -*-

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import json

# 数据载入
def dataLoad(path='data\quora_duplicate_questions.tsv'):
    df = pd.read_csv(path, delimiter='\t')
    df = df.dropna(how='any')  # 删除两行缺失数据
    # 转换编码
    df['question1'] = df['question1'].apply(lambda x: unicode(str(x), 'utf-8'))
    df['question2'] = df['question2'].apply(lambda x: unicode(str(x), 'utf-8'))

    print '数据量: %d行' % (df.shape[0])
    rowNum0 = df['is_duplicate'].value_counts()[0]
    rowNum1 = df['is_duplicate'].value_counts()[1]
    print '重复: 不重复 = %d : %d = 1 : %f' \
          % (rowNum1, rowNum0, (rowNum0 * 1.0 / rowNum1))
    uniqueId = set(list(df['qid2'].unique()) + list(df['qid1'].unique()))
    print '问题总数: %d' % (len(uniqueId))
    return df

# 绘图
def drawing(df):
    counter = Counter([len(x.split()) for x in df['question1'] + df['question2']])
    labels = ['0-9', '10-19', '20-29', '30-39', '>=40']
    sizes = [0, 0, 0, 0, 0]
    for k, v in counter.items():
        if 0 <= k < 10:
            sizes[0] += v
        elif 10 <= k < 20:
            sizes[1] += v
        elif 20 <= k < 30:
            sizes[2] += v
        elif 30 <= k < 40:
            sizes[3] += v
        else:
            sizes[4] += v
    plt.pie(sizes, labels=labels, explode=(0.05, 0, 0, 0.05, 0.05),
            autopct='%2.2f%%', pctdistance=0.8)
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.show()

# 计算TF-IDF分数
def getTfIdfScore(df):
    from sklearn.feature_extraction.text import TfidfVectorizer

    questions = list(df['question1']) + list(df['question2'])

    tfIdf = TfidfVectorizer(lowercase=False)
    tfIdf.fit_transform(questions)

    word2tfIdf = dict(zip(tfIdf.get_feature_names(), tfIdf.idf_))
    return word2tfIdf


if __name__ == '__main__':
    data = dataLoad()
    drawing(data)
    # with open('data\\tfIdfScore.json', 'w') as fp:
    #     fp.write(json.dumps(getTfIdfScore(data)))
