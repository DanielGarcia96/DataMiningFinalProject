# -*- coding: utf-8 -*-
#entropy used from public source code: https://gist.github.com/iamaziz/02491e36490eb05a30f8

"""
Note, used the function from the web, my modication is that I changed the
function name to something more suitable
Also changed gain to move the splitinfo_a into separate function
"""
from math import log

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
    
    gain = info - info_a
    return gain

def gainRatio(gain, splitInfo):
    gainRatio = gain / splitInfo
    
    return gainRatio
