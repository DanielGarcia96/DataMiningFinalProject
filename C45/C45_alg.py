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

info_data = []
entropy_age = []
entropy_workclass = []
entropy_fnlwgt = []
entropy_education = []
entropy_education_num = []
entropy_marital_status = []
entropy_occupation = []
entropy_relationship = []
entropy_race = []
entropy_sex = []
entropy_capital_gain = []
entropy_capital_loss = []
entropy_hours = []
entropy_country = []

entropy_all = []
entropy_all.append(entropy_age)
entropy_all.append(entropy_workclass)
entropy_all.append(entropy_fnlwgt)
entropy_all.append(entropy_education)
entropy_all.append(entropy_education_num)
entropy_all.append(entropy_marital_status)
entropy_all.append(entropy_occupation)
entropy_all.append(entropy_relationship)
entropy_all.append(entropy_race)
entropy_all.append(entropy_sex)
entropy_all.append(entropy_capital_gain)
entropy_all.append(entropy_capital_loss)
entropy_all.append(entropy_hours)
entropy_all.append(entropy_country)

data_df = pd.read_csv('adult_short.data', sep= ', ', header= None, engine= 'python')
data_df.columns = attribute_list

num_rows = len(data_df.index)
num_cols = len(data_df.columns)

#print("here III")
#print(data_df)
    
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
        
    #print("\n->out")
    return

def calcEntropy(info_data, attribute):
    print("CALCULATE ENTROPY")
    print(info_data)
    print(attribute)
    print("I\n")
    info_attribute = []
    for value in attribute:
        print("II")
        print(value)
        #print(value[0])
        #print(value[1], value[2])
        #print()
        info_attribute.append([value[1], value[2]])
    
    print("\nIII")
    print(info_attribute)
    
    #print("-> ", info_data)
    #print("--> ", info_attribute)
    print("IV")
    info_ofdata = info(info_data)
    #print(info_ofdata)
    print("V")
    info_ofattribute = info_a(info_data, info_attribute)
    #print(info_ofattribute)
    print("VI")
    print(info_ofdata)
    print(info_ofattribute)
    infogain = gain(info_ofdata, info_ofattribute)
    #print("---> ", infogain)
    print("VII")
    splitinfo_ofattribute = splitinfo_a(info_data, info_attribute)
    #print("----> ",splitinfo_ofattribute)
    print("VIII")
    gainratio = gainRatio(infogain, splitinfo_ofattribute)
    #print("-----> ", gainratio)
    
    print("OUT of entropy")
    return infogain, gainratio

def giveEntropy(info_data, entropy_all, attrs):
    info_gain = 0
    gain_ratio = 0
    print("\nIN ENTROPY")
    #print(info_data)
    print(entropy_all)
    #print(attrs)
    print()
    for attribute in attrs:
        print(attribute)
        print()
    
    #print(calcEntropy(info_data, attrs[13]))
    print("--->|", attrs[13])
    #ig, gr = calcEntropy(info_data, attrs[13])
    gain = 0
    gr = 0
    print("\n\ntest\n")
    print(info_data)
    print(attrs[13])
    gain, gr = calcEntropy(info_data, attrs[13])
    return


cntClass(data_df, attribute_list, attrs)
#print("out")
#print(attrs)
#print("\n\n\n")
#print(attrs[0])
#print("-> ", attrs[0][0][1], attrs[0][0][2])
#print("-> ", attrs[0][1][1], attrs[0][1][2])
print()

tst = []
print(tst)
tst.append([attrs[0][1], attrs[0][2]])
print(tst)

print("II\n")



greater = 0 # >50K
less = 0 # <= 50K    

for i in range(0, num_rows):
    if data_df['Amount'][i] == ">50K":
        greater += 1
    else:
        less += 1
       
print(">50K =", greater, "<=50K =", less)
print(info([less, greater]))
info_data.append(greater)
info_data.append(less)
print(info_data)

print("entropy")
gain, gr = calcEntropy(info_data, attrs[13])
print("entropy => ", gain, gr)

#attrs[0].append([39, 0, 0])


print("\n--->DEBUG<-----\n")

print(info_data)
print(entropy_all)
#entropy_all[0].append([1, 2])
entropy_age.append(1)
entropy_age.append(2)
print(entropy_all)
print(entropy_age)

print("GIVE")
#giveEntropy(info_data, entropy_all, attrs)

print("\nout")
#for tup in attrs:
    #print(tup)
    #print()
    

print("\n\n\n")

