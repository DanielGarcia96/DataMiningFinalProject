#!/usr/bin/python3

##########################################################################################
# File Name          : tree.py                                                           #
#                                                                                        #
# Author             : Daniel Garcia                                                     #
#                                                                                        #
# Creation Date      : Wed Mar 28 18:18:45 CDT 2018                                      #
#                                                                                        #
# Last Modified      : Wed Apr 18 00:29:33 CDT 2018                                      #
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

import pdb
import math
import random
import logging
import pandas as pd

logging.basicConfig(filename='tree.log', level=logging.DEBUG)

class ID3_Tree(object):
    def __init__(self, data, classes=(1, 2), majority = 0.80, numeric_cutoff = 25, attrs=()):
        self.data = data                                # Tuples we are considering
        self.b_count = 0                                # Number of branches
        self.branches = dict()                          # Branch paths arranged by dictionary
        self.majority = majority                        # Majority needed to be a class
        self.classes = classes                          # Tuple determining the classes 
        self.numeric_cutoff = numeric_cutoff            # Point at which number of unique values indicates a numeric value
        self.attrs = dict()

        if(attrs == ()):
            self.determine_attributes()   # Find out which attributes are numeric
        else:
            self.attrs = attrs   # Find out which attributes are numeric

        self.identifier = self.determine_identifier()   # Two value tuple, (class/test, value)

    def fraction(self, fract=0.1):
        self.data = self.data.sample(frac=fract, replace=True)

    def determine_attributes(self):
        for i in self.data.columns:
            if self.data[i].unique().shape[0] >= self.numeric_cutoff and is_number(self.data.iloc[0][i]):
                self.attrs[i] = 'numeric'
            else:
                self.attrs[i] = 'non-numeric'

    # Determine what kind of node this is
    def determine_identifier(self):
        # If there is no data, pick a class at random
        if self.data.shape[0] == 0:
            return ("class", random.choice(self.classes))
        class_candidates = dict()

        # Check for a majority
        for i in self.classes:
            num = self.data[self.data.iloc[:,-1] == i].shape[0]
            class_candidates[i] = num / self.data.shape[0]

        # If there is a proper majority, this node is a class
        best = max(class_candidates, key=class_candidates.get, default=self.classes[0])
        if(class_candidates[best] >= self.majority):
            return ("class", best)


        logging.debug("Length of self.data.columns: " +
                        '('+str(self.data.columns.shape[0])+')\n')

        # If we've run out of attributes to test on, basic majority wins
        if(self.data.columns.shape[0] <= 1):
            return ("class", best)

        # Otherwise, this is a test node and the attribute to test on will be selected
        # by the attribute selector
        return ("test", self.information_gain_selector(self.data.columns, self.data, self.classes))

    def __str__(self):
        data_string = "Data:\n\t" + ' '.join(self.data) + "\n"
        attribute_string = "Attributes:\n\t" + ' '.join(self.data.columns) + "\n"
        branch_string = '\nBranches:\n'
        for k in self.branches.keys():
            branch_string += "\tKey: " + str(k) + "\n";
        branch_string += "\n"

        result_string = data_string + attribute_string + branch_string
        result_string += "b_count: " + str(self.b_count) + "\n"
        result_string += "majority: " + str(self.majority) + "\n"
        result_string += "classes: " + ' '.join(self.classes) + "\n"
        result_string += "numeric_cutoff: " + str(self.numeric_cutoff) + "\n"
        result_string += "identifier: " + ', '.join(self.identifier) + "\n\n"
        
        return result_string

    def __repr__(self):
        data_string = "Data:\n\t" + ' '.join(self.data) + "\n"
        attribute_string = "Attributes:\n\t" + ' '.join(self.data.columns) + "\n"
        branch_string = '\nBranches:\n'
        for k in self.branches.keys():
            branch_string += "\tKey: " + str(k) + "\n";
        branch_string += "\n"

        result_string = data_string + attribute_string + branch_string
        result_string += "b_count: " + str(self.b_count) + "\n"
        result_string += "majority: " + str(self.majority) + "\n"
        result_string += "classes: " + ' '.join(self.classes) + "\n"
        result_string += "numeric_cutoff: " + str(self.numeric_cutoff) + "\n"
        result_string += "identifier: " + ', '.join(self.identifier) + "\n\n"
        
        return result_string

    def is_numeric(self, data, attr):
        logging.debug("Testing: " + attr)
        logging.debug("Accessing value: " + str(data.iloc[0][attr]))
        return data[attr].unique().shape[0] >= self.numeric_cutoff and is_number(data.iloc[0][attr])

    def information_gain_selector(self, attr, data, classes):
        # calculate expected information to classify a tuple
        expected_info = 0.0
        d_size = data.shape[0]

        for i in classes:
            size_ci = data.loc[data.iloc[:,-1] == i].shape[0]
            pi = size_ci / d_size
            expected_info += pi * math.log(pi, 2)

        expected_info = -expected_info

        # calculate the information gain for each class
        attribute_gain_dict = dict()
        for i in attr:
            # Screen out the class column
            if i == data.columns[-1]:
                break

            attribute_gain_dict[i] = 0.0

            # Use heuristic to see if attribute is numeric
            # If numeric, use median as split point
            if self.attrs[i] == 'numeric':
                median = data[i].median()
                splits = (data.loc[data[i] <= median], data.loc[data[i] > median])

                for j in splits:
                    tmp = 0.0
                    j_size = j.shape[0]
                    for k in classes:
                        k_size = j.loc[j.iloc[:,-1] == k].shape[0]
                        if k_size == 0:
                            continue
                        pk = k_size / j_size
                        tmp -= pk * math.log(pk, 2)

                    attribute_gain_dict[i] += (j_size / d_size) * tmp
                
            # Otherwise just use unique attribute values as split points
            else:
                for g in data[i].unique():
                    tmp = 0.0
                    j = data.loc[data[i] == g]
                    j_size = j.shape[0]
                    for k in classes:
                        k_size = j.loc[j.iloc[:,-1] == k].shape[0]
                        if k_size == 0:
                            continue
                        pk = k_size / j_size
                        tmp -= pk * math.log(pk, 2)

                    attribute_gain_dict[i] += (j_size / d_size) * tmp

            attribute_gain_dict[i] = expected_info - attribute_gain_dict[i]

        logging.debug(attribute_gain_dict)

        # return the greatest information gain
        m = max(attribute_gain_dict, key=attribute_gain_dict.get)
        logging.debug("Returning: " + m)
        return m

    def print_tree(self, depth=0):
        logging.debug("Iterating")
        if self.identifier[0] == "class":
            print("\t" * depth, end=" ")
            print(self.identifier)
        else:
            for val in self.branches.keys():
                print("\t" * depth, end=' ')
                print(self.identifier)
                self.branches[val].print_tree(depth+1)

    # Train the tree
    def fit(self):
        # If this is a class, it's a terminating condition
        if self.identifier[0] == "class":
            return
        # Else it's a test node
        else:
            # Check for a numeric test
            if self.attrs[self.identifier[1]] == 'numeric':
                # Create a branch for all values below the median
                self.branches['below'] =  ID3_Tree (
                    self.data.loc[self.data[self.identifier[1]] <=
                        self.data[self.identifier[1]].median()]
                        .drop(self.identifier[1], axis=1),
                    classes=self.classes,
                    majority=self.majority,
                    numeric_cutoff=self.numeric_cutoff,
                    attrs=self.attrs
                )
                self.b_count += 1
                self.branches['below'].fit()

                # Create a branch for all values above the median
                self.branches['above'] = ID3_Tree (
                    self.data.loc[self.data[self.identifier[1]] >
                        self.data[self.identifier[1]].median()].
                        drop(self.identifier[1], axis=1),
                    classes=self.classes,
                    majority=self.majority,
                    numeric_cutoff=self.numeric_cutoff,
                    attrs=self.attrs
                )
                self.b_count += 1
                self.branches['above'].fit()
            # Must be non-numeric test
            else:
                for i in self.data[self.identifier[1]].unique():
                    self.branches[i] = ID3_Tree (
                        self.data.loc[self.data[self.identifier[1]] == i].
                            drop(self.identifier[1], axis=1),
                        classes=self.classes,
                        majority=self.majority,
                        numeric_cutoff=self.numeric_cutoff,
                        attrs=self.attrs
                    )
                    self.b_count += 1
                    self.branches[i].fit()
            

    # Classify a particular record
    def classify(self, datum):
        # If this is a class, we've reached the end
        if self.identifier[0] == "class":
            return self.identifier[1]
        # Otherwise, look at the next level down
        elif self.attrs[self.identifier[1]] == 'numeric':
            if datum[self.identifier[1]] <= self.data[self.identifier[1]].median():
                return self.branches['below'].classify(datum)
            else:
                return self.branches['above'].classify(datum)
        else:
            if datum[self.identifier[1]] in self.branches.keys():
                return self.branches[datum[self.identifier[1]]].classify(datum)
            else:
                return random.choice(self.classes)

# Helper function to determine if parameter is a number
def is_number(s):
    logging.debug("Checking :" + str(s))
    try:
        float(s)
        return True
    except ValueError:
        try:
            int(s)
            return True
        except:
            return False
        return False


df = pd.read_csv('adult.data.csv', sep=',', header=0)

t1 = ID3_Tree(df, classes=("<=50K", ">50K"), majority=0.6, numeric_cutoff=15)
t1.fraction(0.5)

print("Fitting adult dataset with 0.6 threshold, this could take a minute or two...")
t1.fit()

print("Classifying adult test data: ")
pass_count = 0
fail_count = 0
for i in range(df.shape[0]):
    result = t1.classify(df.iloc[i])

    if df.iloc[i][-1] == result:
        pass_count += 1
    else:
        fail_count += 1

print("Properly identified records: %d" % pass_count)
print("Improperly identified records: %d" % fail_count)
print("Error Rate: %0.2f%%\n\n" % ((fail_count / (pass_count + fail_count)) * 100))

t1 = ID3_Tree(df, classes=("<=50K", ">50K"), majority=0.8, numeric_cutoff=15)
t1.fraction(0.5)

print("Fitting adult dataset with 0.8 threshold, this could take a minute or two...")
t1.fit()

print("Classifying adult test data: ")
pass_count = 0
fail_count = 0
for i in range(df.shape[0]):
    result = t1.classify(df.iloc[i])

    if df.iloc[i][-1] == result:
        pass_count += 1
    else:
        fail_count += 1

print("Properly identified records: %d" % pass_count)
print("Improperly identified records: %d" % fail_count)
print("Error Rate: %0.2f%%\n\n" % ((fail_count / (pass_count + fail_count)) * 100))

df = pd.read_csv('bupa.data.csv', sep=',', header=0)

t2 = ID3_Tree(df)

print("Fitting bupa dataset with 0.9 threshold, this could take a minute or two...")
t2.fit()

print("Classifying bupa test data: ")
pass_count = 0
fail_count = 0
for i in range(df.shape[0]):
    result = t2.classify(df.iloc[i])

    if df.iloc[i][-1] == result:
        pass_count += 1
    else:
        fail_count += 1

print("Properly identified records: %d" % pass_count)
print("Improperly identified records: %d" % fail_count)
print("Error Rate: %0.2f%%\n\n" % ((fail_count / (pass_count + fail_count)) * 100))
