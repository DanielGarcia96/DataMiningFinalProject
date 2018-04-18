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


def distinctValues(data, splittingAttr):
    
    return

def createBranches(data, splittingAttr):
    branches = [ ["none",[]] ]
    #branches = [ ["HS-grad", []], ["Some-college", []] ]
    print("CREATING BRANCHES for discrete values")
    
    print(data)
    for val in data:
        print(val[splittingAttr])
    print("\nlength = ", len(data), "splitting attribute =", splittingAttr)
    print(branches)
    #branches.append([data[0][splittingAttr], [data[0]]])
    #branches.remove(branches[0])
    #print(branches)
    #print()
    #for val in branches:
    #    print(val)
    
    #print("---->")
    #print(branches[0]) #index
    #print(branches[0][0]) # attr name
    #print(branches[0][1]) # list that contains stuff with that attr
    
    
    #print("<----")

    inList = 0
    for tuples in data:
        for i in range(0, len(branches)):
            #print(branches[i])
            if tuples[splittingAttr] == branches[i][0]:
                #print("TRUE")
                inList = 1
                branches[i][1].append(tuples)
                break
        if inList == 0:
            branches.append([tuples[splittingAttr], [tuples]])
            #print("False")
            
        
    branches.remove(branches[0])
    
    #print(branches)
    #for val in branches:
        #print(val[0])
        #print(val[1])
        #for tup in val[1]:
        #    print(tup[splittingAttr])
        #sys.exit()
        #for obj in val:
            #print(obj)
        #print()
        
    print("_____________")
    finalBranches = []
    for val in branches:
        finalBranches.append(val[1])
        
    #print(finalBranches)
    #for branch in finalBranches:
    #    print(branch)
    #    print()
    
    print("LENGTH = ", len(finalBranches))
    
    #sys.exit()
    
    #
    #for tuples in data:
    #    print(tuples[splittingAttr])
    #    if (tuples[splittingAttr] in branches) == true:
    #        branches[splittingAttr].append
    #    else:
    #        branches.append([tuples[splittingAttr], [tuples]])
        
        #    branches.append(tuples)
    
    print()
    #print(branches[0])
    #sys.exit()
    
    return(finalBranches)

def generateDecisionTree(data, attributeList, exclusionList):
    print("Building Tree")
    #print(data)
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
            continuousValues = 1
            print("------->", splittingAttr)
            splittingThreshold = getSplitPoint(data, splittingAttr, index)
            print(splittingThreshold)
            
    if continuousValues == 1: #The splitting attribute contains continuous values
        dataLeftBranch, dataRightBranch = partition(data, splittingAttr, splittingThreshold)
        leftBranch = generateDecisionTree(dataLeftBranch, attributeList, exclusionList)
        rightBranch = generateDecisionTree(dataRightBranch, attributeList, exclusionList)
        return(DecisionNode(splittingAttr, leftBranch, rightBranch))
    else:
        print("-->", splittingAttr)
        branches = createBranches(data, splittingAttr)
        #print(branches[0])
        #print("LENGTH = ", len(branches))
        
        discBranches = emptyArray(len(branches))
        print(discBranches)
        
        for i in range(0, len(branches)):
            discBranches[i] = generateDecisionTree(branches[i], attributeList, exclusionList)
            
        
        #for index, brnData in branches:
        #    discBranches[index] = generateDecisionTree(brnData, attributeList, exclusionList)
        
        #print
        #for brnData, brn in branches, discBranches:
            #brn = generateDecisionTree(brnData, attributeList, exclusionList)
        
        #print(discBranches)
        #for obj in discBranches:
        #    print(obj)
            
    
    
    #generateDecisionTree(data, attributeList, exclusionList)
    
    return   
        
        
def partition(data, splittingAttribute, splitPoint):
    print("Parition Data on Splitting Attribute")
    
    dataLeftBranch = []
    dataRightBranch = []
    
    print(splittingAttribute,"@", splitPoint)
    #print(data)
    #for val in data:
        #print(val[0])
    for tuples in data:
        #print(tuples[splittingAttribute])
        if tuples[splittingAttribute] > splitPoint:
            dataRightBranch.append(tuples)
        else:
            dataLeftBranch.append(tuples)
        
    print("LENGTH L =", len(dataLeftBranch), "R =", len(dataRightBranch))
    
    return dataLeftBranch, dataRightBranch

def printTree(rootNode):
    if isinstance(rootNode, LeafNode):
        print("TRUE")
        #print("\t", rootNode.data)
        return
    
    print(rootNode.name)
    print("Left")
    printTree(rootNode.leftBranch)
    print("Right")
    printTree(rootNode.rightBranch)


