# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:41:57 2018

@author: Kyle
"""

from entropy import *
from operator import itemgetter, attrgetter
import sys

def dump(obj):
  for attr in dir(obj):
    print("obj.%s = %r" % (attr, getattr(obj, attr)))

class LeafNode:
    def __init__(self, partitioned_data_df):
        self.data = partitioned_data_df
        
    def getData(self):
        for data in self.data:
            for a in data:
                print(a)
            
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
    
    #print("\t\t\t\t\t--->||", infoAllData)
    
    #print(" Yes | No\n", infoAllData)
    entropyList = giveEntropy(infoAllData, attrClassCount)
    
    #print("\t\t\t\t\t--->||", entropyList)
    
    
    
    
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
        
    print("\t\t\t\t\t\t\t----->|", highestGain)
    
    #if highestGain == 0:
    #    print("\t\t\t\t\t\t\t----->|||", highestGain)
    #    sys.exit()
    
    
    #print(saveIndex, highestGain, attributeList[saveIndex])
    
    exclusionList.append(attributeList[saveIndex])
    #print(exclusionList)
    return(attributeList[saveIndex], saveIndex, exclusionList, highestGain)

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
    #print(splitPoint)
    #sys.exit()
    #print()
    
    
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
    #branches = [ ["none",[]] ]
    #branches = [ ["HS-grad", []], ["Some-college", []] ]
    print("CREATING BRANCHES for discrete values")
    
    
    college = ["Bachelors", "Masters", "Some-college", "Assoc-acdm", "Assoc-voc", "Doctorate", "Prof-school"]
    # private or not private workclass
    married = ["Married-spouse-absent", "Married-AF-spouse", "Married-civ-spouse"]
    office = ["Tech-support", "Adm-clerical", "Exec-Managerial", "Sales"]
    # race -> white or not?
    # sex -> male or female?
    # native country -> US or not
    
    leftBranch = []
    rightBranch = []
    
    #if splittingAttr == "Education":
    print("SPLITTING ATTR =>", splittingAttr)
    print(len(data))
    for tuples in data:
        if tuples[splittingAttr] in college or tuples[splittingAttr] in married or tuples[splittingAttr] in office:
            #print("\tTRUE", tuples)
            rightBranch.append(tuples)
        else:
            leftBranch.append(tuples)
                #print("false")
        #print(data)
        #tmp = []
        #index = 0
        #for tuples in data:
        #    tmp.append(tuples[splittingAttr])
        #    print(index, tuples[splittingAttr])
        #    index += 1
        #tmp = list(set(tmp))
        #print(tmp)
        
    #print(leftBranch)
    #sys.exit()
    
    print("L =", len(leftBranch), "R =", len(rightBranch))
    
    #print(data)
    #for val in data:
    #    print(val[splittingAttr])
    #print("\nlength = ", len(data), "splitting attribute =", splittingAttr)
    #print(branches)
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

    #inList = 0
    #for tuples in data:
    #    for i in range(0, len(branches)):
    #        #print(branches[i])
    #        if tuples[splittingAttr] == branches[i][0]:
    #            #print("TRUE")
    #            inList = 1
    #            branches[i][1].append(tuples)
    #            break
    #    if inList == 0:
    #        branches.append([tuples[splittingAttr], [tuples]])
            #print("False")
            
        
    #branches.remove(branches[0])
    
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
        
    #print("_____________")
    #finalBranches = []
    #for val in branches:
    #    finalBranches.append(val[1])
        
    #print(finalBranches)
    #for branch in finalBranches:
    #    print(branch)
    #    print()
    
    #print("LENGTH = ", len(finalBranches))
    
    #sys.exit()
    
    #
    #for tuples in data:
    #    print(tuples[splittingAttr])
    #    if (tuples[splittingAttr] in branches) == true:
    #        branches[splittingAttr].append
    #    else:
    #        branches.append([tuples[splittingAttr], [tuples]])
        
        #    branches.append(tuples)
    
    #print()
    #print(branches[0])
    #sys.exit()
    
    return(leftBranch, rightBranch)

def generateDecisionTree(data, attributeList, exclusionList):
    print("\nBuilding Tree")
    
    print("|",len(data),"|")
    print(len(exclusionList))
    
    if len(data) == 0:
        return
    
    #print(data)
    if len(exclusionList) == 14:
       return(LeafNode(data))
    
    #tmp = []
    #index = 0
    #for tuples in data:
        #tmp.append(tuples["Occupation"])
        #print(index, tuples["Education])
        #index += 1
    #tmp = list(set(tmp))
    #print(tmp)
    #sys.exit()
    
    splittingAttr, index, exclusionList, highestGain = attribute_selection_method(data, attributeList, exclusionList)
    print("\tsplitting Attribute =", splittingAttr, "at index", index, "gain =", highestGain)#, exclusionList)
    print()
    
    
    if highestGain == 0:
        return(LeafNode(data))
    
    #print(data)
    continuousValues = 0
    # check if all data belong to the same class
    purity = checkIfClassesPure(data)
    if purity == 1:
        return(LeafNode(data))
    
    # check if we have any more attributes to split on
    
    
    """
    if len(exclusionList) == 14:
        print("done")
        return
    """
    
    
    if splittingAttr == "Age" or splittingAttr == "fnlwgt" or splittingAttr == "Education_Num" \
        or splittingAttr == "Capital_Gain" or splittingAttr == "Capital_Loss" or splittingAttr == "Hours_Per_Week":
            continuousValues = 1
            print("------->", splittingAttr)
            splittingThreshold = getSplitPoint(data, splittingAttr, index)
            print(splittingThreshold)
            
    if continuousValues == 1: #The splitting attribute contains continuous values
        funcSwitch = 1
        dataLeftBranch, dataRightBranch = partition(funcSwitch, data, splittingAttr, splittingThreshold)
        leftBranch = generateDecisionTree(dataLeftBranch, attributeList, exclusionList)
        rightBranch = generateDecisionTree(dataRightBranch, attributeList, exclusionList)
        return(DecisionNode(splittingAttr, leftBranch, rightBranch))
    else:
        funcSwitch = 2
        print("-->", splittingAttr)
        #dataLeftBranch, dataRightBranch = partition(funcSwitch, data, splittingAttr)
        dataLeftBranch, dataRightBranch = createBranches(data, splittingAttr)
        leftBranch = generateDecisionTree(dataLeftBranch, attributeList, exclusionList)
        rightBranch = generateDecisionTree(dataRightBranch, attributeList, exclusionList)
        
        #print(leftBranch)
        #print(rightBranch)
        return(DecisionNode(splittingAttr, leftBranch, rightBranch))
        #branches = createBranches(data, splittingAttr)
        #print(branches[0])
        #print("LENGTH = ", len(branches))
        
        #discBranches = emptyArray(len(branches))
        #print(branches)
        
        
        
        #sys.exit()
        #for i in range(0, len(branches)):
            #discBranches[i] = generateDecisionTree(branches[i], attributeList, exclusionList)
            
        
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
        
        
def partition(funcSwitch, data, splittingAttribute, splitPoint=None):
    print("Parition Data on Splitting Attribute")
    
    
    #if funcSwitch == 2:
    #    print("-----------------HERE")
        #print(data)
    #   print(splittingAttribute)
    #    print("length =", len(data))
        #for tuples in data:
        #    print(tuples)
    #    sys.exit()
    
    
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
        print("Leaf")
        #print("\t", rootNode.data)
        #print(rootNode.getData)
        #dump(rootNode)
        print(len(rootNode.data))
        return
    
    if rootNode == None:
        return
    else:
        print(rootNode.name)
    print("going Left")
    #print("\t|left")
    printTree(rootNode.leftBranch)
    print("going Right")
    #print("\t\t|right")
    printTree(rootNode.rightBranch)

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, LeafNode):
        #print (spacing + "Predict", node.data)
        return

    if node == None:
        return

    # Print the question at this node
    print (spacing + str(node.name))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.leftBranch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.rightBranch, spacing + "  ")

