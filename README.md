# quoraDupli
identifying duplicate questions on Quora

- 'preprocess.py'对数据进行预处理，删除两行存在NaN数据，数据量为404288
- 'tfIdf.py'提取每个词的词向量，并乘上tf-idf值形成最终的句子特征，300维
- 'quora.py'训练并测试
