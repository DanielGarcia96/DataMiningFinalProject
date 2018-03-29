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
Age = []
Workclass = []
fnlwgt = []
Education = []
Education_num = []
Marital_status = []
Occupation = []
Relationship = []
Race = []
Sex = []
Capital_gain = []
Capital_loss = []
Hours = []
Country = []

attrs = []
attrs.append(Age)
attrs.append(Workclass)
attrs.append(fnlwgt)
attrs.append(Education)
attrs.append(Education_num)
attrs.append(Marital_status)
attrs.append(Occupation)
attrs.append(Relationship)
attrs.append(Race)
attrs.append(Sex)
attrs.append(Capital_gain)
attrs.append(Capital_loss)
attrs.append(Hours)
attrs.append(Country)

data_df = pd.read_csv('adult_short.data', sep= ', ', header= None, engine= 'python')
data_df.columns = attribute_list

num_rows = len(data_df.index)
num_cols = len(data_df.columns)

#print("here III")
#print(data_df)
    
def infoGain(data_df, attribute_list):
    #print(data_df)
    #print(attribute_list)
    return

def checkInList(value, class_answer, class_label):
    """
    # list structure = [attribute, num_yes, num_no]
    # so => list subscripts
    # yes for list[1]
    # no  for list[2]
    
    for handling missing data, I applied the class_answers to whichever
    attribute had the most total sum
    """
    yes = 1
    no = 2
    inlst = 0
    greater = 0 # >50K
    less = 0 # <= 50K
    
    if value == "?":
        #print(value, class_answer)
        #print(class_label)
        #print(len(class_label))
        save_ndx = -1
        maxTotal = 0
        for ndx in range(0, len(class_label)):
            if maxTotal < (class_label[ndx][1] + class_label[ndx][2]):
                maxTotal = (class_label[ndx][1] + class_label[ndx][2])
                save_ndx = ndx
        if class_answer == ">50K":
            class_label[save_ndx][yes] += 1
        else:
            class_label[save_ndx][no] += 1
        
        return
    
    for val in class_label: #check to see if this value is in the list, inc necessary value
        if value == val[0]:
            inlst = 1
            if class_answer == ">50K":
                val[yes] += 1
            else:
                val[no] += 1
        
            break

    if inlst == 0: #if not in the list Age, add it
        if class_answer == ">50K":
            class_label.append([value, 1, 0])
        else:
            class_label.append([value, 0, 1])

    return

def cntClass(data_df, attribute_list, attrs):
    num_rows = len(data_df.index)
    num_cols = len(data_df.columns)

    #print("-->")
    #print(attrs)
    #print(attrs[0])
    #print("<--")
    #print("IN cntClass\n")
    for i in range(0, num_rows):
        for j in range(0, num_cols-1): # -1 to avoid counting class: amount
            #print( data_df[attribute_list[j]][i], " ", end="" )
            #print(data_df[attribute_list[j]][i])
            checkInList(data_df[attribute_list[j]][i],
                        data_df['Amount'][i],
                        attrs[j])
            #print( data_df[attribute_list[j]][i], " ", end="" )
            #print()
            
           
        #print("|here X", data_df['Amount'][i])
        #print() #\n
        
        #if i == 20:
        #    break
        
    print("\n->out")
    return


for i in range(0, num_rows):
    if data_df['Amount'][i] == ">50K":
        greater += 1
    else:
        less += 1
       
print(">50K =", greater, "<=50K =", less)
print(entropy([less, greater]))


#attrs[0].append([39, 0, 0])

cntClass(data_df, attribute_list, attrs)

print("\n--->DEBUG<-----\n")

print("\nout")
for tup in attrs:
    print(tup)
    #print()

