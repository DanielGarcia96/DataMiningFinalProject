#!/usr/bin/python3

##########################################################################################
# File Name          : stats.py                                                          #
#                                                                                        #
# Author             : Daniel Garcia                                                     #
#                                                                                        #
# Creation Date      : Wed Apr 25 21:51:29 CDT 2018                                      #
#                                                                                        #
# Last Modified      : Wed Apr 25 21:51:29 CDT 2018                                      #
#                                                                                        #
# Purpose            : Export the relevant statistics for the adult and bupa datasets.   #
#                                                                                        #
# Command Parameters : None                                                              #
#                                                                                        #
# Input              : None                                                              #
#                                                                                        #
# Results            : Relevant statistics for the adult and bupa datasets               #
#                                                                                        #
# Returns            : Relevant statistics for the adult and bupa datasets               #
#                                                                                        #
# Notes              :                                                                   #
#     - Depends on the adult.data.csv and bupa.data.csv files in ID3                     #
#     - Depends on the Pandas library                                                    #
#                                                                                        #
##########################################################################################

import pandas as pd

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

adult_df = pd.read_csv('ID3/adult.data.csv')
print("Adult numeric attributes")
print(adult_df.select_dtypes(include=numerics).drop(labels=['education-num'], axis=1).describe(include='all'))
print(" ")

print("Adult nominal attributes")
print(adult_df.select_dtypes(exclude=numerics).drop(labels=['education'], axis=1).describe(include='all'))
print(" ")

print("Adult ordinal attributes")
print(adult_df[['education', 'education-num']].describe(include='all'))
print(" ")

bupa_df = pd.read_csv('ID3/bupa.data.csv')
print(bupa_df.describe(include='all'))
