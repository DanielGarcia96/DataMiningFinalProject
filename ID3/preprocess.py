#!/usr/bin/python3

##########################################################################################
# File Name          : preprocess.py                                                     #
#                                                                                        #
# Author             : Daniel Garcia                                                     #
#                                                                                        #
# Creation Date      : Fri Apr 27 12:47:23 CDT 2018                                      #
#                                                                                        #
# Last Modified      : Fri Apr 27 12:47:23 CDT 2018                                      #
#                                                                                        #
# Purpose            : Remove null and empty data values                                 #
#                                                                                        #
# Command Parameters :                                                                   #
#                                                                                        #
# Input              :                                                                   #
#                                                                                        #
# Results            :                                                                   #
#                                                                                        #
# Returns            : A DataFrame cleaned of all null values.                           #
#                                                                                        #
# Notes              :                                                                   #
#                                                                                        #
#                                                                                        #
##########################################################################################

import pandas as pd

def create_and_process_data():
    df = pd.read_csv('adult.data.csv', sep=',', header=0)

    # Clean null values out of discrete attributes
    df.loc[df['workclass'] == '?', 'workclass'] = df['workclass'].mode()[0]
    df.loc[df['marital-status'] == '?', 'marital-status'] = df['marital-status'].mode()[0]
    df.loc[df['occupation'] == '?', 'occupation'] = df['occupation'].mode()[0]
    df.loc[df['relationship'] == '?', 'relationship'] = df['relationship'].mode()[0]
    df.loc[df['race'] == '?', 'race'] = df['race'].mode()[0]
    df.loc[df['sex'] == '?', 'sex'] = df['sex'].mode()[0]
    df.loc[df['education'] == '?', 'education'] = df['education'].mode()[0]
    df.loc[df['nativecountry'] == '?', 'nativecountry'] = df['nativecountry'].mode()[0]

    return df
