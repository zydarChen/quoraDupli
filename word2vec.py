# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import gensim
from tqdm import tqdm

df = pd.read_csv("data\quora_duplicate_questions.tsv", delimiter='\t')
df = df.fillna('miss')

