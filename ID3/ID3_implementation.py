#!/usr/bin/python3

##########################################################################################
# File Name          : ID3_implementation.py                                             #
#                                                                                        #
# Author             : Daniel Garcia                                                     #
#                                                                                        #
# Creation Date      : Mon Mar 26 14:08:59 CDT 2018                                      #
#                                                                                        #
# Last Modified      : Mon Mar 27 23:01:59 CDT 2018                                      #
#                                                                                        #
# Purpose            : An implementation of ID3 for the adult data                       #
#                                                                                        #
# Command Parameters :                                                                   #
#                                                                                        #
# Input              :                                                                   #
#                                                                                        #
# Results            :                                                                   #
#                                                                                        #
# Returns            :                                                                   #
#                                                                                        #
# Notes              :                                                                   #
#                      Only implemeted for a certain dataset.                            #
#                                                                                        #
##########################################################################################

from pandas import DataFrame
import numpy as np

df = DataFrame.from_csv('adult.data.csv', sep=' ', index_col=None)

split_points = {
    "age" : (df['age'].max() / df['age'].min()) / 2,
    "fnlwgt" : (df['fnlwgt'].max() / df['fnlwgt'].min()) / 2,
    "capital-gain" : (df['capital-gain'].max() / df['capital-gain'].min()) / 2,
    "capital-loss": (df['capital-loss'].max() / df['capital-loss'].min()) / 2,
    "hours" : (df['hours'].max() / df['hours'].min()) / 2
}

# calculate entropy
above = df.loc[df['greater-than-50k'] == '>50K'].shape[0]
below = df.loc[df['greater-than-50k'] == '<=50K'].shape[0]

information = -((above / df.shape[0]) * math.log(above / df.shape[0], 2)) - \
                              ((below / df.shape[0]) * math.log(below / df.shape[0], 2))

def information_gain_selector(attributes, tuples):
    # calculate information gain
    info_d = dict(zip(attributes, np.zeros(len(attributes)))) 

    for i in info_d.keys():
        tmp = 0.0

        # Do a simple binary split
        if i in split_points:
            above_split = tuples.loc[tuples[i] >  split_points[i]]
            below_split = tuples.loc[tuples[i] <= split_points[i]]

            p_above = above_split.loc[above_split.['greater-than-50k'] == '>50K']
            n_above = above_split.loc[above_split.['greater-than-50k'] == '<=50K']
            p_below = below_split.loc[below_split.['greater-than-50k'] == '>50K']
            n_below = below_split.loc[below_split.['greater-than-50k'] == '<=50K']

            tmp =   ((above_split.shape[0] / tuples.shape[0]) *             \
                    ((-(p_above.shape[0] / above_split.shape[0]) *      \
                    log(p_above.shape[0] / above_split.shape[0], 2)) -  \
                    (n_above.shape[0] / above_split.shape[0]) *         \
                    log(n_above.shape[0] / above_split.shape[0], 2))) + \
                    ((below_split.shape[0] / tuples.shape[0]) *             \
                    ((-(p_below.shape[0] / below_split.shape[0]) *      \
                    log(p_below.shape[0] / below_split.shape[0], 2)) -  \
                    (n_below.shape[0] / below_split.shape[0]) *         \
                    log(n_below.shape[0] / below_split.shape[0], 2))) 
                    

            info_d.put(i, info_d.get(i) + tmp)
            info_d.put(i, information - info_d.get(i))
        # Do a split on each unique value
        else:
            tmp = 0.0
            for k in i.unique():
                sp = tuples.loc[tuples[i] == k]

                p = sp.loc[sp.['greater-than-50k'] == '>50K']
                n = sp.loc[sp.['greater-than-50k'] == '<=50K']

                tmp +=  ((sp.shape[0] / tuples.shape[0]) *       \
                        ((-(p.shape[0] / sp.shape[0]) *      \
                        log(p.shape[0] / sp.shape[0], 2)) -  \
                        (n.shape[0] / sp.shape[0]) *         \
                        log(n.shape[0] / sp.shape[0], 2)))

            info_d.put(i, info_d.get(i) + tmp)
            info_d.put(i, information - info_d.get(i))


    # select the attribute with the most information gain
    greatest = info_d.keys()[0]
    for i in info_d.keys():
        if info_d.get(i) > info_d.get(greatest):
            greatest = i

    return greatest
        

def ID3():
    N = []
    # (1) create a node N ;
    # (2) if tuples in D are all of the same class, C, then
    # (3) return N as a leaf node labeled with the class C;
    # (4) if attribute list is empty then
    # (5) return N as a leaf node labeled with the majority class in D; // majority voting
    # (6) apply Attribute selection method(D, attribute list) to find the “best” splitting criterion;
    # (7) label node N with splitting criterion;
    # (8) if splitting attribute is discrete-valued and multiway splits allowed then
    #       not restricted to binary trees
    # (9) attribute list ← attribute list − splitting attribute; // remove splitting attribute
    # (10) for each outcome j of splitting criterion
    #        partition the tuples and grow subtrees for each partition
    # (11) let Dj be the set of data tuples in D satisfying outcome j; // a partition
    # (12) if Dj is empty then
    # (13) attach a leaf labeled with the majority class in D to node N ;
    # (14) else attach the node returned by Generate decision tree(Dj , attribute list) to node N ;
    #       endfor
    # (15) return N ;
