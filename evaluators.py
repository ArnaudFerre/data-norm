# Author: Arnaud Ferré
# RALI, Montreal University
#
# Description :



#######################################################################################################
# Imports:
#######################################################################################################


from os import listdir
from os.path import isfile, join, splitext, basename
import re
import sys
from numpy import std, median



#######################################################################################################
# Functions:
#######################################################################################################

def accuracy(dd_pred, dd_resp):
    totalScore = 0.0

    for id in dd_resp.keys():
        score = 0.0
        l_cuiPred = dd_pred[id]["pred_cui"]
        l_cuiResp = dd_resp[id]["cui"]
        if len(l_cuiPred) > 0:
            for cuiPred in l_cuiPred:
                if cuiPred in l_cuiResp:
                    score+=1
            score=score/len(l_cuiResp) #multi-norm

        totalScore += score

    totalScore = totalScore/len(dd_resp.keys())

    return totalScore




#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':

    print("No test...")