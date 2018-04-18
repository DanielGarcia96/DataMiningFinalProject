# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:41:57 2018

@author: Kyle
"""

from entropy import *
from operator import itemgetter, attrgetter
import sys

class LeafNode:
    def __init__(self, partitioned_data_df):
        self.data = partitioned_data_df

class DecisionNode:
    def __init__(self, name, leftBranch, rightBranch):
        self.name = name
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch

def attribute_selection_method(data, attributeList, exclusionList):
    funcSwitch = 1 # just an identifier based on which function is calling it
    print("Selecting Attribute/Splitting Criteria")
    #list_of_classes = cntClass(data, attributeList)
    #attrClassCount = cntClass(data, attributeList, funcSwitch)
    attrClassCount = cntClass(funcSwitch, data, attributeList)
    infoAllData = getInfo(data, funcSwitch)
    #print(" Yes | No\n", infoAllData)
    entropyList = giveEntropy(infoAllData, attrClassCount)
    #print(" Info GAIN | Gain Ratio\n", entropyList)
    #print()
    #print(exclusionList)
    
    # Use gain selection to determine if you want to use information gain or gain ratio
    # default for C45 is gain ratio
    gainSelection = 1
    highestGain = 0
    saveIndex = 0
    index = 0
    
    for gains in entropyList:
        #print(index, gains, attributeList[index])
        #print(gainRatio)
        #print("--->", gains[gainSelection])
        if float(gains[gainSelection]) > float(highestGain) and attributeList[index] not in exclusionList:
            highestGain = gains[gainSelection]
            saveIndex = index
            
        index += 1
        
    
    #print(saveIndex, highestGain, attributeList[saveIndex])
    
    exclusionList.append(attributeList[saveIndex])
    #print(exclusionList)
    return(attributeList[saveIndex], saveIndex, exclusionList)

def checkIfClassesPure(data):
    yes = 0
    no = 0
    for tuples in data:
        if yes == 1 and no == 1: # not pure
            #print("NOT PURE")
            return(0)
        if tuples['Amount'] == ">50K":
            yes = 1
        if tuples['Amount'] == "<=50K":
            no = 1
    
    #print("PURE DATA")
    return(1)
    
def calcSplitPoints(continuousValues):
    #print("calcualte possible split points")
    
    candidateSplittingPoints = []
    for i in range(0, len(continuousValues)-1):
        #print(continuousValues[i])
        #splitPoint =
        candidateSplittingPoints.append(int((continuousValues[i] + continuousValues[i+1]) / 2))
    
    return(candidateSplittingPoints)

def getSplitPoint(data, splittingAttr, index):
    print("getting Splitting criterion")
    print()
    #tmpListValuesClasses = sorted(data, key=attrgetter(splittingAttr))
    tmpListValuesClasses = []
    continuousValues = []
    for tup in data:
        #print("-->", tup[splittingAttr])
        tmpListValuesClasses.append([tup[splittingAttr], tup['Amount']])
        
    #print(tmpListValuesClasses)
    for pairs in tmpListValuesClasses:
        #print(pairs)
        continuousValues.append(pairs[0])
    
    #print(continuousValues)
    continuousValues = list(set(continuousValues))
    #print(continuousValues)
    
    #print("sort it")
    continuousValues.sort()
    #print(continuousValues)
    #print()
    
    candidateSplittingPoints = calcSplitPoints(continuousValues)
    lengthUniqueValues = len(continuousValues)
    #print(candidateSplittingPoints)
    #sys.exit()
    
    splitPoint = getThreshold(tmpListValuesClasses, candidateSplittingPoints, lengthUniqueValues)
    print(splitPoint)
    #sys.exit()
    print()
    
    
    """
    print("\n")
    for tup in tmpListValuesClasses:
        print(tup[splittingAttr])
        
    list(set(tmpListValuesClasses))
    for tup in tmpListValuesClasses:
        print(tup[splittingAttr])
    """
    
    
    return(splitPoint)

def generate_decision_tree(data, attributeList, exclusionList):
    print("Building Tree")
    continuousValues = 0
    # check if all data belong to the same class
    purity = checkIfClassesPure(data)
    if purity == 1:
        return(LeafNode(data))
    
    # check if we have any more attributes to split on
    if len(exclusionList) == 14:
       return(LeafNode(data))
    
    """
    if len(exclusionList) == 14:
        print("done")
        return
    """
    
    splittingAttr, index, exclusionList = attribute_selection_method(data, attributeList, exclusionList)
    print("\tsplitting Attribute =", splittingAttr, "at index", index)#, exclusionList)
    print()
    
    if splittingAttr == "Age" or splittingAttr == "fnlwgt" or splittingAttr == "Education_Num" \
        or splittingAttr == "Capital_Gain" or splittingAttr == "Capital_Loss" or splittingAttr == "Hours_Per_Week":
            continuous = 1
            print("------->", splittingAttr)
            splittingThreshold = getSplitPoint(data, splittingAttr, index)
            print(splittingThreshold)
            
    if continuousValues == 1: #The splitting attribute contains continuous values
        dataLeftBranch, dataRightBranch = partition(data, splittingThreshold)
    else:
        return
            
            
    
    #generate_decision_tree(data, attributeList, exclusionList)
    
    
    return   
        
        
def partition(data, splittingAttribute):
    dataLeftBranch = []
    dataRightBranch = []
    
    
    
    return leftBranch, rightBranch

























