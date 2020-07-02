#pip.install.efficient_apriori
import pandas as pd
import numpy as np
from efficient_apriori import apriori
# 数据加载
data = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
print(data.shape)
# 将数据放入交易订单中
transcations = []
for i in range(0,data.shape[0]):
    temp = []
    for j in range(data.shape[1]):
        if str(data.values[i,j]) != 'nan':
            temp.append(str(data.values[i,j]))
    transcations.append(temp)
print(transcations)
itemsets, rules = apriori(transcations, min_support = 0.05, min_confidence = 0.2)
print('频繁项集：', itemsets)
print('关联规则：', rules)