#!/usr/bin/python3

##########################################################################################
# File Name          : tree.py                                                           #
#                                                                                        #
# Author             : Daniel Garcia                                                     #
#                                                                                        #
# Creation Date      : Wed Mar 28 18:18:45 CDT 2018                                      #
#                                                                                        #
# Last Modified      : Wed Mar 28 18:18:45 CDT 2018                                      #
#                                                                                        #
# Purpose            : Provide a class for a tree structure                              #
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
#                                                                                        #
##########################################################################################

class Tree(object):
    def __init__(self, tuples = None, attributes = None, majority = 0.9):
        self.tuples = tuples                     # Tuples we are considering
        self.b_count = 0                         # Number of branches
        self.attributes = attributes             # Attributes we can test on
        self.branches = dict()                   # Branch paths arranged by dictionary
        self.majority = majority                 # Majority needed to be a class
        self.identifier = determine_identifier() # Two value tuple, (class/test, value)

    # Determine what kind of node this is
    def determine_identifier():
        # Find the percentage of tuples >50k
        above = self.tuples.loc[self.tuples['greater-than-50K'] == ">50k"].shape[0]
        above_percent /= self.tuples.shape[0]

        # Find the percentage of tuples <=50k
        below = self.tuples.loc[self.tuples['greater-than-50K'] == "<=50k"].shape[0]
        below_percent /= self.tuples.shape[0]

        # If there is a majority, this node is a class
        if above_percent >= self.majority:
            return ("class", ">50k")
        elif below_percent >= self.majority:
            return ("class", "<=50k")

        # If we've run out of attributes to test on, basic majority wins
        if len(attributes) <= 0 or attributes == None:
            if above > below:
                return ("class", ">50k")
            else
                return ("class", "<=50k")

        # Otherwise, this is a test node and the attribute to test on will be selected
        # by the attribute selector
        return ("test", attribute_selector(self.tuples))

    # Train the tree
    def train():
        # If this is a class, it's a terminating condition
        if self.identifier[0] == "class":
            return

        # For each unique value of the selected attribute,
        # make a branch with only those unique values and train it
        for i in self.tuples[self.identifier[1]].unique():
            self.branches[i] = 
                Tree(tuples = self.tuples.loc[tuples[self.identifier] == i],
                     attributes = [x for i, x in enumerate(self.attributes) if i != self.identifier[1]],
                     majority = self.majority)
            self.b_count += 1
            self.branches[i].train()

    # Classify a particular record
    def classify(datum):
        # If this is a class, we've reached the end
        if self.identifier[0] == "class":
            return self.identifier[1]
        # Otherwise, look at the next level down
        else:
            self.branches[datum[self.identifier[1]]].classify(datum)
