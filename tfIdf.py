# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
from preprocess import *
import numpy as np
import en_core_web_md as md

# 数据载入
df = dataLoad()
if os.path.exists('data\\tfIdfScore.json'):
    with open('data\\tfIdfScore.json', 'r') as fp:
        tfIdfScore = json.loads(fp.read())
else:
    print '载入tfIdfScore.json失败'
    tfIdfScore = getTfIdfScore(df)

nlp = md.load()
questionList = [list(df['question1']), list(df['question1'])]
finalVec = []
for i, questions in enumerate(questionList):
    print '正在处理question%i......' % (i+1)
    vec = []
    for question in tqdm(questions):
        words = nlp(question)
        mean_vec = np.zeros([len(words), 300])
        for word in words:
            wordVector = word.vector
            try:
                idf = tfIdfScore[str(word)]
            except:
                idf = 0
            mean_vec += wordVector * idf
        mean_vec = mean_vec.mean(axis=0)
        vec.append(mean_vec)
    finalVec.append(vec)

df['q1Vector'] = finalVec[0]
df['q2Vector'] = finalVec[1]

# 保存
pd.to_pickle(df, 'data\questionVector.pickle')
