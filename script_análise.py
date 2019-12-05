import sys
import csv
import quaternion_toolbox
from madgwickahrs import MadgwickAHRS
import numpy as np
from scipy import signal
from scipy import stats
import pandas as pd

data100 = []
data50 = []
data25 = []


filePath = 'Datasets/posicao_erro_ST_0_1.csv'

dataset = pd.read_csv(filePath)

data100 = dataset.iloc[:30, 2].values
data50 = dataset.iloc[30:60, 2].values
data25 = dataset.iloc[60:, 2].values

print(data100)
print(data50)
print(data25)

w1, pv1 = stats.shapiro(data100)
w2, pv2 = stats.shapiro(data50) 
w3, pv3 = stats.shapiro(data25)

print('----------Teste de Normalidade--------------')
print("W", "p-value")
print(w1,pv1)
print(w2,pv2)
print(w3,pv3)

t1,p1 = stats.wilcoxon(data100, data50, zero_method='wilcox', correction=False, alternative='less')
t2,p2 = stats.wilcoxon(data100, data25, zero_method='wilcox', correction=False, alternative='less')
t3,p3 = stats.wilcoxon(data50, data25, zero_method='wilcox', correction=False, alternative='less')

print('----------Teste de Wilcoxon--------------')
print("statistic","pvalue")
print(t1,p1)
print(t2,p2)
print(t3,p3)
