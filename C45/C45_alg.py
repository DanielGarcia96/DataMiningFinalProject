# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:42:43 2018

@author: Kyle

Using the adult data set, this will create a classifier based on
if the user makes >50K or <=50K

"""
from entropy import *

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_boston

attribute_list = ["Age", "Workclass", "fnlwgt", "Education", "Education num",
                  "Marital Status", "Occupation", "Relationship", "Race", "Sex",
                  "Capital Gain", "capital Loss", "Hours Per Week", "Native Country", "Amount"]

data_df = pd.read_csv('adult_short.data', sep= ', ', header= None, engine= 'python')
data_df.columns = attribute_list

#for col in data:
#    print(data[col][0])

# Run false to put all on one line, run true for default
#pd.set_option('expand_frame_repr', False)
#pd.set_option('expand_frame_repr', True)

#df = pd.DataFrame(data)
#print(df)
#print 1 row
#print(data_df[0:1])

#print(data_df)

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

print("here I")
#print(data_df[0][0], data_df[1][0], data_df[3][0], data_df[5][0], data_df[6][0], \
#      data_df[8][0], data_df[9][0], data_df[12][0])


print("\n\n\n\n\n")
#for value in data:
#    print(value)
#    print(data[value])
#

print("here II")
#for i in range(0, 10):
#    #print(i, " hi")
#    print(data_df[0][i], data_df[1][i], data_df[3][i], data_df[5][i], data_df[6][i], \
#      data_df[8][i], data_df[9][i], data_df[12][i], data_df[14][i])


num_rows = len(data_df.index)


print("here III")
print(data_df)
#data_df = data_df.sort_values(by=['Age'])
#print(data_df)

#broken
#print(data_df[0][0], data_df[1][0], data_df[3][0], data_df[5][0], data_df[6][0], \
#      data_df[8][0], data_df[9][0], data_df[12][0])

"""
print(data_df['Amount'][0])
print(data_df['Amount'][1])
print(data_df['Amount'][2])
print(data_df['Amount'][8])

if data_df['Amount'][8] == ">50K":
    print("TRUE")
else:
    print("FALSE")
"""

#for i in range(0, num_rows):
#    print(i)

greater = 0 # >50K
less = 0 # <= 50K
for i in range(0, num_rows):
    if data_df['Amount'][i] == ">50K":
        greater += 1
    else:
        less += 1
        
print(">50K =", greater, "<=50K =", less)

print(entropy([less, greater]))











