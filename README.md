# quoraDupli
identifying duplicate questions on Quora

- 'LabGuide_2017S.pdf'课程任务要求
- [Quora上任务详情](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)
- [参考博客1](http://www.erogol.com/duplicate-question-detection-deep-learning/?utm_source=tuicool&utm_medium=referral)
- [参考博客2](https://explosion.ai/blog/quora-deep-text-pair-classification#dataset)
- [preprocess.py](preprocess.py)对数据进行预处理，删除两行存在NaN数据，数据量为404288
- [tfIdf.py](tfIdf.py)提取每个词的词向量，并乘上tf-idf值形成最终的句子特征，300维
- [quora.py](quora.py)训练并测试
