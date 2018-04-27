# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:01:58 2018

@author: Kyle
"""

import timeit
import time

import itertools
import sys
from math import log
from operator import *
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_boston
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt

fixedAttributeList = ["Age", "Workclass", "fnlwgt", "Education", "Education_Num",
                     "Marital_Status", "Occupation", "Relationship", "Race", "Sex",
                     "Capital_Gain", "Capital_Loss", "Hours_Per_Week", "Native_Country", "Amount"]

def leaf_class_label(partitioned_data):
    yes = 0
    no = 0
    for values in partitioned_data:
        if values['Amount'] == ">50K":
            yes += 1
        else:
            no += 1
    if yes > no:
        return(">50K")
    else:
        return("<=50K")

class LeafNode:
    def __init__(self, partitioned_data):
        self.classLabel = leaf_class_label(partitioned_data)
      
class DecisionNode:
    def __init__(self, question, leftBranch, rightBranch):
        self.question = question
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch

"""
-------------------------------------------------------------------------------
Function here was taken from google's example of a decision tree classifer using
CART.
https://github.com/random-forests/tutorials/blob/master/decision_tree.py

"""
def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

class Question:
    """A Question is used to partition a dataset.
    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val > self.value #changed from >= to >
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">"
            
        #print(self.column, self.value)
        
        return "Is %s %s %s?" % (
            fixedAttributeList[self.column], condition, str(self.value))

"""
-------------------------------------------------------------------------------
"""


'''
Returns an empty list of the list of attributes which are empty,
initializes an empty list to be used for the info gain and class counting
'''
def empty_attr_list():
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

    return(attrs)

def info(pi):
    '''
    return the Entropy of a probability distribution:
    entropy(p) = − SUM (Pi * log(Pi) )
    defintion:
            entropy is a metric to measure the uncertainty of a probability distribution.
    entropy ranges between 0 to 1
    Low entropy means the distribution varies (peaks and valleys).
    High entropy means the distribution is uniform.
    See:
            http://www.cs.csi.cuny.edu/~imberman/ai/Entropy%20and%20Information%20Gain.htm
    '''
    total = 0
    for p in pi:
        p = p / sum(pi)
        if p != 0:
            total += p * log(p, 2)
        else:
            total += 0
    total *= -1
    return total

def info_a(d, a):
    '''
    d = info(a)
    a = the values for the tuples
    '''
    total = 0
    for d_j in a:
        total += (sum(d_j) / sum(d)) * info(d_j)

    return total

def splitinfo_a(d, a):
    '''
    d = info(a)
    a = the values for the tuples
    '''
    total = 0
    for d_j in a:
        total -= (sum(d_j) / sum(d)) * log( sum(d_j)/sum(d), 2 )

    return total

def gain(info, info_a):
    '''
    return the information gain:
    gain(D, A) = entropy(D)−􏰋 SUM ( |Di| / |D| * entropy(Di) )
    '''
    #print("IN gain FUNC")
    #print("X| info = ", info)
    #print("X| info_a = ", info_a)
    
    gain = info - info_a

    return gain

def gainRatio(gain, splitInfo):
    if gain == 0 or splitInfo == 0:
        return(0)
    gainRatio = gain / splitInfo

    return gainRatio

def gini(pi):
    total = 0
    for p in pi:
        p = p / sum(pi)
        p = pow(p, 2)
        if p != 0:
            total -= p
        else:
            total += 0
    total += 1
    return(total)

def gini_attribute(sum_data, yes_no_subset):
    total = 0
    for values in yes_no_subset:
        total += (sum(values)/sum(sum_data)) * gini(values)

    return(format(total, '.5f'))

def calc_entropy(infoData, attribute):
    info_attribute = []
    for value in attribute:
        info_attribute.append([value[1], value[2]])

    info_ofdata = info(infoData)
    info_ofattribute = info_a(infoData, info_attribute)
    infogain = gain(info_ofdata, info_ofattribute)
    splitinfo_ofattribute = splitinfo_a(infoData, info_attribute)
    gainratio = gainRatio(infogain, splitinfo_ofattribute)

    return infogain, gainratio

def get_entropy(infoData, attrAnswers):
    info_gain = 0
    gain_ratio = 0
    filledEntropyList = empty_attr_list()

    i = 0
    for attribute in attrAnswers:
        info_gain, gain_ratio = calc_entropy(infoData, attribute)
        filledEntropyList[i].append(format(info_gain, '.5f'))
        filledEntropyList[i].append(format(gain_ratio, '.5f'))
        i += 1
    
    return(filledEntropyList)

def checkIfClassesPure(data):
    yes = 0
    no = 0
    
    for tuples in data:
        if yes == 1 and no == 1: # not pure
            return(0)
        if tuples['Amount'] == ">50K":
            yes = 1
        if tuples['Amount'] == "<=50K":
            no = 1

    return(1)

def get_info(data, funcSwitch, candidateSplitPoint=None):
    yes = 0
    no = 0
    for tuples in data:
        if funcSwitch == 1:
            if tuples['Amount'] == ">50K":
                yes += 1
            else:
                no += 1
        elif funcSwitch == 2:
            if tuples[0] > candidateSplitPoint:
                yes += 1
            else:
                no += 1

    return([yes, no])

def check_class_labels(value, class_answer, listClassAnswers):
    """
    # list structure = [attribute, num_yes, num_no]
    # so => list subscripts
    # yes for list[1]
    # no  for list[2]
    
    value = current attribute value
    class_answer = either <=50k or >50K for that attribute
    listClassAnswers = everything in the list already from previous calls
    
    for handling missing data, I applied the class_answers to whichever
    attribute had the most total sum
    """
    yes = 1
    no = 2
    inlst = 0

    if value == "?":
        save_ndx = 0
        maxTotal = 0
        for ndx in range(0, len(listClassAnswers)):
            if maxTotal < (listClassAnswers[ndx][1] + listClassAnswers[ndx][2]):
                maxTotal = (listClassAnswers[ndx][1] + listClassAnswers[ndx][2])
                save_ndx = ndx
                inlst = 1
        if inlst == 0:
            return
        if class_answer == ">50K":
            listClassAnswers[save_ndx][yes] += 1
        else:
            listClassAnswers[save_ndx][no] += 1

        return
    
    for val in listClassAnswers: #check to see if this value is in the list, inc necessary value
        if value == val[0]:
            inlst = 1
            if class_answer == ">50K":
                val[yes] += 1
            else:
                val[no] += 1
            break
    if inlst == 0: #if not in the list, add it
        if class_answer == ">50K":
            listClassAnswers.append([value, 1, 0])
        else:
            listClassAnswers.append([value, 0, 1])

    return

"""
Here attributeList is the list of attributes if the funcSwitch == 1
but if funcSwitch == 2, all it is is the length of the candidates getting passed in
"""
def cnt_class_values(funcSwitch, data, attributeList=None):
    attrCounts = empty_attr_list()

    if funcSwitch == 1:
        for tuples in data:
            for j in range(0, 14): # -1 to avoid counting class: amount
                check_class_labels(tuples[fixedAttributeList[j]], tuples['Amount'], attrCounts[j])
    elif funcSwitch == 2:
        for tuples in data:
            for j in range(0, attributeList):
                check_class_labels(tuples[0], tuples[1], attrCounts[j])
    elif funcSwitch == 3:
        yesAbove = 0
        noAbove = 0
        yesBelow = 0
        noBelow = 0
        for tuples in data:
            #Above the threshold, class answers == yes or >50K
            #Here attributeList is the candidate split point/threshold candidate
            if tuples[0] > attributeList:
                if tuples[1] == ">50K":
                    yesAbove += 1
                else:
                    noAbove += 1
            else:
                if tuples[1] == ">50K":
                    yesBelow += 1
                else:
                    noBelow += 1    
        classAttr = [[yesAbove, noAbove], [yesBelow, noBelow]]

        return(classAttr)

    return(attrCounts)

def check_attribute_available(index, reducedAttributeList):
    availableToUse = 0

    if fixedAttributeList[index] in reducedAttributeList:
        availableToUse = 1
    else:
        availableToUse = 0

    return(availableToUse)

def attribute_selection_method(data, reducedAttributeList):
    print("      Selecting Attribute with highest Gain Ratio")
    funcSwitch = 1
    attrClassCount = cnt_class_values(funcSwitch, data)
    infoData = get_info(data, funcSwitch)
    entropyList = get_entropy(infoData, attrClassCount)    
    # Use gain selection to determine if you want to use information gain or gain ratio
    # default for C45 is gain ratio
    gainSelection = 1
    highestGain = 0
    saveIndex = 0
    index = 0

    for gains in entropyList:
        if check_attribute_available(index, reducedAttributeList) == 1:
            if float(gains[gainSelection]) > float(highestGain):
                highestGain = gains[gainSelection]
                saveIndex = index
        index += 1
    
    if highestGain == 0:
        return(None, None, highestGain, None)
    else:
        newReducedAttributeList = reducedAttributeList
        newReducedAttributeList.remove(fixedAttributeList[saveIndex])
        return(fixedAttributeList[saveIndex], saveIndex, highestGain, newReducedAttributeList)

def testThreshold(data, candidateSplitPoint, lengthUniqueValues):
    funcSwitch = 2
    info = get_info(data, funcSwitch, candidateSplitPoint)
    funcSwitch = 3
    attrClassCount = cnt_class_values(funcSwitch, data, candidateSplitPoint)
    expectedMinInfo = info_a(info, attrClassCount)
    
    return(expectedMinInfo, candidateSplitPoint)

def getThreshold(data, candidateSplitPoints, lengthUniqueValues):
    lowestGain = 100
    splitPoint = 0
    
    for candidate in candidateSplitPoints:
        infoGainThreshold, possibleSplitPoint = testThreshold(data, candidate, lengthUniqueValues)
        
        if infoGainThreshold < lowestGain:
            lowestGain = infoGainThreshold
            splitPoint = candidate

    return(splitPoint)

def calcSplitPoints(uniqueValues):
    candidateSplittingPoints = []
    for i in range(0, len(uniqueValues)-1):
        candidateSplittingPoints.append(int((uniqueValues[i] + uniqueValues[i+1]) / 2))

    return(candidateSplittingPoints)

def getSplitPoint(data, splittingAttr, index):
    print("          Getting Splitting criterion")
    tmpListValuesClasses = []
    uniqueValues = []
    
    for tup in data:
        tmpListValuesClasses.append([tup[splittingAttr], tup['Amount']])

    for pairs in tmpListValuesClasses:
        uniqueValues.append(pairs[0])

    uniqueValues = list(set(uniqueValues))
    uniqueValues.sort()
    candidateSplittingPoints = calcSplitPoints(uniqueValues)
    lengthUniqueValues = len(uniqueValues)
    splitPoint = getThreshold(tmpListValuesClasses, candidateSplittingPoints, lengthUniqueValues)

    return(splitPoint)

def get_combination(data, splittingAttr):
    uniqueValues = []
    for tuples in data:
        uniqueValues.append(tuples[splittingAttr])

    uniqueValues = list(set(uniqueValues))

    return(uniqueValues)

def get_possible_subsets(data, uniqueValues):
    possibleSubsets = []

    for L in range(0, len(uniqueValues)+1):
        for subset in itertools.combinations(uniqueValues, L):
            sub = []
            if subset == ():
                continue
            if len(subset) == len(uniqueValues):
                continue
            for value in subset:
                sub.append(value)
            possibleSubsets.append(sub)

    return(possibleSubsets)

def calc_gini_index(allData, subsets):
    giniIndex = gini_attribute(allData, subsets)
    
    return(giniIndex)

def get_gini_index(data, splittingAttr, possibleSubsets):
    funcSwitch = 1
    allData = get_info(data, funcSwitch)
    saveCandidate = None
    minGiniIndex = 100
    
    for candidate in possibleSubsets:
        yesCandidate = 0
        noCandidate = 0
        yesNotCandidate = 0
        noNotCandidate = 0
        
        #print(candidate)

        for value in data:
            if value[splittingAttr] in candidate:
                if value['Amount'] == ">50K":
                    yesCandidate += 1
                else:
                    noCandidate += 1
            else:
                if value['Amount'] == ">50K":
                    yesNotCandidate += 1
                else:
                    noNotCandidate += 1

        giniIndex = calc_gini_index(allData, [[yesCandidate, noCandidate], [yesNotCandidate, noNotCandidate]])
        
        #print(giniIndex)
        
        if float(giniIndex) < float(minGiniIndex):
            minGiniIndex = giniIndex
            saveCandidate = candidate

    return(saveCandidate, float(minGiniIndex))

def partition_possible_subsets(possibleSubsets):
    partitionedPossibleSubsets = []
    fraction = len(possibleSubsets) * 0.15
    fraction = int(fraction)
    
    #print("PS  ->", len(possibleSubsets))
    #print("PPS ->", len(partitionedPossibleSubsets))
    
    tmp = []
    count = 0
    while len(possibleSubsets) != 0:
        #print(len(possibleSubsets), count, fraction)
        if count == fraction:
            #print("\t\t", count, fraction, len(tmp))
            partitionedPossibleSubsets.append(tmp)
            #print("\t", len(partitionedPossibleSubsets))
            count = 0
            tmp = []
            #print(tmp, count)
            #time.sleep(15)
        else:
            tmp.append(possibleSubsets.pop())
            count += 1

    #print("PS  -->", len(possibleSubsets))
    #print("PPS -->", len(partitionedPossibleSubsets))
    #print("TMP -->", len(tmp))
        #partitionedPossibleSubsets.append(tmp)
    #print()        
    #print(possibleSubsets)
    #print(len(partitionedPossibleSubsets))
    #print()
    
    #print(len(partitionedPossibleSubsets[-1]))
    #print(len(partitionedPossibleSubsets[-2]))
    #print()
    #print(partitionedPossibleSubsets)
    #print(partitionedPossibleSubsets[1])
        
    #print()
    #print("->STOP<-")
    #sys.exit()
        
        
        
    return(partitionedPossibleSubsets)

def get_splitting_subset(data, splittingAttr, index):
    print("          Getting Splitting Subset")
    uniqueValues = get_combination(data, splittingAttr)
    possibleSubsets = get_possible_subsets(data, uniqueValues)
    if splittingAttr == "Education":
        #print("EDU")
        partitionedSubsets = partition_possible_subsets(possibleSubsets)    
    
        saveSplittingSubset = None
        saveGiniIndex = 100
        
        for part in partitionedSubsets:
            splittingSubset, giniIndex = get_gini_index(data, splittingAttr, part)
            #print(giniIndex, splittingSubset)
            if giniIndex < saveGiniIndex:
                saveSplittingSubset = splittingSubset
                saveGiniIndex = giniIndex
    else:
        saveSplittingSubset, giniIndex = get_gini_index(data, splittingAttr, possibleSubsets)
            


    #print()
    #print("-|STOP|-")
    #sys.exit()    

    return(saveSplittingSubset)

def partition(funcSwitch, data, splittingAttribute, splittingCriterion=None):
    print("              Partition Data on Splitting Attribute")
    dataLeftBranch = []
    dataRightBranch = []

    if funcSwitch == 1:
        for tuples in data:
            if tuples[splittingAttribute] > splittingCriterion:
                dataRightBranch.append(tuples)
            else:
                dataLeftBranch.append(tuples)
    elif funcSwitch == 2:
        for tuples in data:
            if tuples[splittingAttribute] in splittingCriterion:
                dataRightBranch.append(tuples)
            else:
                dataLeftBranch.append(tuples)

    return (dataLeftBranch, dataRightBranch)

def generate_decision_tree(data, attributeList):
    print("    |",len(data),"|", "Attributes Available =", len(attributeList))
    
    # Return if no more data | Return data as leaf if no more attributes to split on
    if len(data) == 0:
        return()
    if len(attributeList) == 0:
        print("                  No More Attributes, Return Majority Class")
        return(LeafNode(data))
    # check if all data belong to the same class
    purity = checkIfClassesPure(data)
    if purity == 1:
        print("                  PURE DATA")
        #print(data)
        return(LeafNode(data))

    splittingAttr, index, gainRatio, reducedAttributeList = attribute_selection_method(data, attributeList)
    #print("  chose splitting Attribute =", "|->", splittingAttr, "<-|",  \
    #      "at index", index, "gain =", "|->", gainRatio, "<-|")
    print("        Splitting Attribute =", splittingAttr)


    #---------------
    
    #splittingAttr = "Education"
    #---------------



    if gainRatio == 0:
        return(LeafNode(data))

    continuousValues = 0  
    if splittingAttr == "Age" or splittingAttr == "fnlwgt" or splittingAttr == "Education_Num" \
        or splittingAttr == "Capital_Gain" or splittingAttr == "Capital_Loss" or splittingAttr == "Hours_Per_Week":
            continuousValues = 1
            #print("------->", splittingAttr)
            splittingThreshold = getSplitPoint(data, splittingAttr, index)
            print("            Split Point =", splittingThreshold)
    else:
        splittingSubset = get_splitting_subset(data, splittingAttr, index)
        print("            Spliting Subset =", splittingSubset)

    if continuousValues == 1: #The splitting attribute contains continuous values
        funcSwitch = 1
        dataLeftBranch, dataRightBranch = partition(funcSwitch, data, splittingAttr, splittingThreshold)
        question = Question(index, splittingThreshold)
    else:
        funcSwitch = 2
        dataLeftBranch, dataRightBranch = partition(funcSwitch, data, splittingAttr, splittingSubset)
        question = Question(index, splittingSubset)

    print("                LENGTH L =", len(dataLeftBranch), "R =", len(dataRightBranch))
    print("  Recursing")
    leftBranch = generate_decision_tree(dataLeftBranch, attributeList)
    print("  Recursing")
    rightBranch = generate_decision_tree(dataRightBranch, attributeList)

    return(DecisionNode(question, leftBranch, rightBranch))

"""
-------------------------------------------------------------------------------
Function(s) here/below was taken from google's example of a decision tree classifer using
CART.

"""
def print_tree(node, spacing=""):
    # Base case: we've reached a leaf
    if isinstance(node, LeafNode):
        print (spacing + "Predict", node.classLabel)
        #print(node.getData)
        #dump(node)
        return

    if node == None:
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> False:')
    print_tree(node.leftBranch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> True:')
    print_tree(node.rightBranch, spacing + "  ")

def classify(row, node):
    # Base case: we've reached a leaf
    if isinstance(node, LeafNode):
        return node.classLabel

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.rightBranch)
    else:
        return classify(row, node.leftBranch)

def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs











#Main
if __name__ == '__main__':
    start = timeit.default_timer()
    
    # Attribute List for data
    attributeList = ["Age", "Workclass", "fnlwgt", "Education", "Education_Num",
                     "Marital_Status", "Occupation", "Relationship", "Race", "Sex",
                     "Capital_Gain", "Capital_Loss", "Hours_Per_Week", "Native_Country", "Amount"]

    # load Sample data to dataframe
    sample_data = pd.read_csv('adult_small.data', sep= ', ', header= None, engine= 'python')
    sample_data.columns = attributeList
    # Split the data into training/test with a
    training_data, test_data = train_test_split(sample_data, test_size=0.2)

    print("\nStart alg...")
    print("Size of Training Data =", len(training_data), \
          "Size of Testing Data =", len(test_data))

    trainingData = []
    testData = []

    # Convert dataframe to a list
    for index, row in training_data.iterrows():
        trainingData.append(row)
    for index, row in test_data.iterrows():
        testData.append(row)

    attributeList.pop()
  
    print("\nBuilding Tree/Fit Data")
    decisionTree = generate_decision_tree(trainingData, attributeList)
    print("------------------------------------------------------------------")
    print("Finished Fitting Data")

    print_tree(decisionTree)

    print()
    print("Classify")
    #for row in testData:
        #print ("Actual: %s. Predicted: %s" %
               #(row[-1], print_leaf(classify(row, decisionTree))))


    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for row in testData:
        predict = classify(row, decisionTree)
        actual = row['Amount']
        #print("Predicted", predict, "  |Actual =", actual, )
        
        if predict == row['Amount']:
            if predict == ">50K":
                TP += 1
                #print("\tTrue Positive")
            else:
                TN += 1
                #print("\tTrue Negative")
        else:
            if predict == ">50K":
                if actual == "<=50K":
                    #print("\tFalse Positive")
                    FP += 1
                else:
                    #print("\tFalse Negative")
                    FN += 1
            if predict == "<=50K":
                if actual == ">50K":
                    #print("\tFalse Negative")
                    FN += 1
                else:
                    #print("\tFalse Positive")
                    FP += 1


    funcSwitch = 1
    classLabels = get_info(testData, funcSwitch)
    print(classLabels)
    sumPN = sum(classLabels)
    
    print()
    print("TP =", TP,
          "TN =", TN,
          "FP =", FP,
          "FN =", FN)
    errorRate = (FP + FN) / sumPN
    acc = (TP + TN) / sumPN
    tpr = TP / classLabels[0]
    tnr = TN / classLabels[1]
    prec = TP / (TP + FP)
    
    print("error rate =", errorRate,
          "accuracy =", acc,
          "TPR =", tpr,
          "TNR =", tnr,
          "Precision =", prec)

    stop = timeit.default_timer()
    print(stop - start, "seconds to complete")
