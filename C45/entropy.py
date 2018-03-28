# -*- coding: utf-8 -*-
#entropy used from public source code: https://gist.github.com/iamaziz/02491e36490eb05a30f8

from math import log

def entropy(pi):
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


def gain(d, a):
    '''
    return the information gain:
    gain(D, A) = entropy(D)−􏰋 SUM ( |Di| / |D| * entropy(Di) )
    '''

    total = 0
    for v in a:
        total += sum(v) / sum(d) * entropy(v)

    gain = entropy(d) - total
    return gain
