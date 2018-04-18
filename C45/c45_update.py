from operator import *
from entropy import *
from TreeNode import *

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_boston

'''
Returns an empty list of the list of attributes which are empty,
initializes an empty list to be used for the info gain and class counting
'''



attributeList = ["Age", "Workclass", "fnlwgt", "Education", "Education_Num",
                  "Marital_Status", "Occupation", "Relationship", "Race", "Sex",
                  "Capital_Gain", "Capital_Loss", "Hours_Per_Week", "Native_Country", "Amount"]


data_df = pd.read_csv('adult_short.data', sep= ', ', header= None, engine= 'python')
data_df.columns = attributeList

listTuples = []
exclusionList = []
for index, row in data_df.iterrows():
    listTuples.append(row)
    
decisionTree = generateDecisionTree(listTuples, attributeList, exclusionList)

print("\n\n------------------------------------------------------------\nDone")

#printTree(decisionTree)

print_tree(decisionTree)
