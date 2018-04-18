# -*- coding: utf-8 -*-
#entropy used from public source code: https://gist.github.com/iamaziz/02491e36490eb05a30f8

"""
Note, used the function from the web, my modication is that I changed the
function name to something more suitable
Also changed gain to move the splitinfo_a into separate function

to use functions:
    pass in a list of numbers
        for info() you pass in a list of the format [yes, no] for the class label you want
        for info_a() you pass in first the class label of [yes, no]
            and secondly you pass in all the values for the attribute you're checking
            [yes, no] for every tuple in list format
            ex: if 14 tuples total, 9 yes 5 no => info([9,5])
                and [2,3] [4,0] [3,2] for the values => info_a([9,5], [[2,3], [4,0], [3,2]])
"""
from math import log
import sys

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
    gainRatio = gain / splitInfo
    
    return gainRatio

def calcInfoGain(info, info_a):
    infoGain = 0
    
    
    return infoGain

def emptyAttrList():
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

def checkClassLabels(value, class_answer, listClassAnswers):
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
        save_ndx = -1
        maxTotal = 0
        for ndx in range(0, len(listClassAnswers)):
            if maxTotal < (listClassAnswers[ndx][1] + listClassAnswers[ndx][2]):
                maxTotal = (listClassAnswers[ndx][1] + listClassAnswers[ndx][2])
                save_ndx = ndx
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

    if inlst == 0: #if not in the list Age, add it
        if class_answer == ">50K":
            listClassAnswers.append([value, 1, 0])
        else:
            listClassAnswers.append([value, 0, 1])

    return

'''
Here attributeList is the list of attributes if the funcSwitch == 1
but if funcSwitch == 2, all it is is the length of the candidates getting passed in
'''
def cntClass(funcSwitch, data, attributeList=None):
    attrCounts = emptyAttrList()
    if funcSwitch == 1:
        for tuples in data:
            for j in range(0, 14): # -1 to avoid counting class: amount
                checkClassLabels(tuples[attributeList[j]],
                                 tuples['Amount'],
                                 attrCounts[j])
    elif funcSwitch == 2:
        #print("FUNC SWITCH == 2")
        #print(attrCounts)
        for tuples in data:
            #print(tuples)
            for j in range(0, attributeList):
                checkClassLabels(tuples[0], tuples[1], attrCounts[j])
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
            
        #print(yesAbove, noAbove, yesBelow, noBelow)
        classAttr = [[yesAbove, noAbove], [yesBelow, noBelow]]
        #print(classAttr)
        
        return(classAttr)
    #print(attrCounts)
    
    #for val in attrCounts:
    #    print(val)
    
    return(attrCounts)

def getInfo(data, funcSwitch, candidateSplitPoint=None):
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

def testThreshold(data, candidateSplitPoint, lengthUniqueValues):
    funcSwitch = 2
    #print("\nGetting info gain of threshold")
    
    #print(data)
    
    
    info = getInfo(data, funcSwitch, candidateSplitPoint)
    #print("for candidate split point", candidateSplitPoint)
    
    #print("overall info [aboveSplitPoint, belowSplitPoint] = ", info)
    
    funcSwitch = 3
    attrClassCount = cntClass(funcSwitch, data, candidateSplitPoint)
    #print(attrClassCount)
    
    expectedMinInfo = info_a(info, attrClassCount)
    #print(expectedMinInfo)    
    
    #countAttributes = cntClass(funcSwitch, data, lengthUniqueValues)
    #print(countAttributes)
    #countAttributes.pop() # removes extra list
    #for val in countAttributes:
    #    print(val)
    
    #print(countAttributes)
    
    #print()
    #print("------------------------------")
    #print(countAttributes[0])
    #print(countAttributes[1])
    
    return(expectedMinInfo, candidateSplitPoint)

def getThreshold(data, candidateSplitPoints, lengthUniqueValues):
    #print("\nGetting Threshold")
    threshold = 0
    #print(data)
    #print("SP = ", candidateSplitPoints)
    #print()
    lengthCandidates = len(candidateSplitPoints)
    #print(candidateSplitPoints)
    #print("length of candidates =", lengthCandidates,"\n")
    #print(data)
    #print("length of list =", len(data),"\n")
    
    lowestGain = 100
    splitPoint = 0
    
    test = []
    
    for candidate in candidateSplitPoints:
        #print(candidate)
        infoGainThreshold, possibleSplitPoint = testThreshold(data, candidate, lengthUniqueValues)
        test.append([infoGainThreshold, possibleSplitPoint])
        if infoGainThreshold < lowestGain:
            lowestGain = infoGainThreshold
            splitPoint = candidate
        
    
    for pairs in test:
        print(pairs)
    print(lowestGain, splitPoint)
    
    return(splitPoint)

def calcEntropy(info_data, attribute):
    #print("\nCALCULATE ENTROPY of ")
    info_attribute = []
    for value in attribute:
        info_attribute.append([value[1], value[2]])
    
    info_ofdata = info(info_data)
    info_ofattribute = info_a(info_data, info_attribute)
    #print("info(d) = ", info_ofdata)
    #print("info_a(d) = ", info_ofattribute)
    infogain = gain(info_ofdata, info_ofattribute)
    #print("gain(a) = ", infogain)
    splitinfo_ofattribute = splitinfo_a(info_data, info_attribute)
    #print("splitinfo_a(d) = ",splitinfo_ofattribute)
    gainratio = gainRatio(infogain, splitinfo_ofattribute)
    #print("gainratio(a) = ", gainratio)
    
    #print()
    return infogain, gainratio

def giveEntropy(info_data, attrAnswers):
    info_gain = 0
    gain_ratio = 0
    filledEntropyList = emptyAttrList()
    #print("\nIN ENTROPY")
    #print(emptyAttrList)
    #print()
    i = 0
    for attribute in attrAnswers:
        info_gain, gain_ratio = calcEntropy(info_data, attribute)
        filledEntropyList[i].append(format(info_gain, '.5f'))
        filledEntropyList[i].append(format(gain_ratio, '.5f'))
        i += 1
    
    #print(filledEntropyList)
    return(filledEntropyList)


