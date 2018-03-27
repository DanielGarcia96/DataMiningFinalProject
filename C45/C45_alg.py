# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:42:43 2018

@author: Kyle
"""

import pprint
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.datasets import load_boston

data_df = pd.read_csv('adult.data', sep=',', header= None)
boston_dataset = load_boston()


#for col in data:
#    print(data[col][0])

# Run false to put all on one line, run true for default
#pd.set_option('expand_frame_repr', False)
#pd.set_option('expand_frame_repr', True)

#df = pd.DataFrame(data)
#print(df)
#print 1 row
print(data_df[0:1])
#print()
#print(data[0:1])

#print
print("break\n\n")
#print(data_df[0][0], data_df[1], data_df[5], data_df[6], data_df[8], data_df[9], data_df[12])
"""
print(data_df[0][0])
print(data_df[1][0])
print(data_df[3][0])
print(data_df[5][0])
print(data_df[6][0])
print(data_df[8][0])
print(data_df[9][0])
print(data_df[12][0])
"""

print(data_df[0][0], data_df[1][0], data_df[3][0], data_df[5][0], data_df[6][0], \
      data_df[8][0], data_df[9][0], data_df[12][0])


print("\n\n\n\n\n")
#for value in data:
#    print(value)
#    print(data[value])
#

for i in range(0, 10):
    #print(i, " hi")
    print(data_df[0][i], data_df[1][i], data_df[3][i], data_df[5][i], data_df[6][i], \
      data_df[8][i], data_df[9][i], data_df[12][i])




























