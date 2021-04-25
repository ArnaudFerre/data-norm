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
import time




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


################################################

from loaders import is_desc


# Wang score:
def S_Aconcept_value(Aconcept, l_TA, concept, w, dd_ref):

    value = None

    if Aconcept == concept:
        value = 1.0

    else:
        tps1 = time.clock()
        l_childrenOfConcept = list() #concept.children
        for cui in dd_ref.keys():
            if is_desc(dd_ref, cui, concept):
                l_childrenOfConcept.append(cui)


        # If a children is not in TA, it's not useful to explore this path (not defined in Wang et al. 2007...):
        l_TA_inter_childrenOfConcept = list(set(l_childrenOfConcept) & set(l_TA))

        l_childrenValues = list()
        for children in l_TA_inter_childrenOfConcept:
            print("\t",end="")
            childrenValue =  S_Aconcept_value(Aconcept, l_TA, children, w, dd_ref)
            l_childrenValues.append(w * childrenValue)

        if len(l_childrenValues) > 0:
            value = max(l_childrenValues)
        else:
            value = 0.0

        tps2 = time.clock()
        print("\t", dd_ref[Aconcept]["label"], dd_ref[concept]["label"], tps2 - tps1)

    return value


def get_TA(Aconcept, dd_ref):
    l_TA = [Aconcept]

    l_ancestorsOfA = list() #Aconcept.rparents(level=-1, intermediate=True)
    for cui in dd_ref.keys():
        if is_desc(dd_ref, Aconcept, cui):
            l_ancestorsOfA.append(cui)

    if len(l_ancestorsOfA) > 0:
        for ancestorOfA in l_ancestorsOfA:
            l_TA.append(ancestorOfA)
    return l_TA



def semantic_value(Aconcept, w, dd_ref):
    l_TA = get_TA(Aconcept, dd_ref)

    SV = 0.0
    for concept in l_TA:
        SV += S_Aconcept_value(Aconcept, l_TA, concept, w, dd_ref)

    return SV



def wang_score(pred, ref, w, dd_ref):
    score = 0.0

    # Find intersection of TA and TB:
    l_Tpred = get_TA(pred, dd_ref)
    l_Tref = get_TA(ref, dd_ref)
    l_inter = list(set(l_Tref) & set(l_Tpred))

    # Sum S-values on intersection:
    for concept in l_inter:
        tps1 = time.clock()
        score += S_Aconcept_value(pred, l_Tpred, concept, w, dd_ref) + S_Aconcept_value(ref, l_Tref, concept, w, dd_ref)
        tps2 = time.clock()
        print("\t", dd_ref[concept]["label"], tps2 - tps1, "score += S_A_pred +  S_A_soluce\n")

    # Divide by SV(A)+SV(B):
    score = score / (semantic_value(pred, w, dd_ref) + semantic_value(ref, w, dd_ref))

    sys.exit(0)

    return score




def global_wang_score(dd_pred, dd_resp, dd_ref, w):
    totalScore = 0.0

    progresssion = -1
    for i, id in enumerate(dd_resp.keys()):

        score = 0.0
        l_cuiPred = dd_pred[id]["pred_cui"]
        l_cuiResp = dd_resp[id]["cui"]

        if len(l_cuiPred) == 1:
            cuiPred = dd_pred[id]["pred_cui"][0]

            print("mention", dd_resp[id]["mention"], " -> ", dd_ref[cuiPred]["label"], " - soluce:", dd_ref[dd_resp[id]["cui"][0]]["label"])

            l_scoreForThisRep = list()
            for cuiResp in l_cuiResp:
                l_scoreForThisRep.append(wang_score(cuiPred, cuiResp, w, dd_ref))

            score = max(l_scoreForThisRep)
            score=score/len(l_cuiResp) #multi-norm expected cases

        elif len(l_cuiPred) > 1:
            print("WARNING: Multi-norm predictions not take into account in this Wang score...")


        totalScore += score


        #Print progression:
        currentProgression = round(100*(i/len(dd_resp.keys()) ))
        if currentProgression > progresssion:
            print(currentProgression, "%")
            progresssion = currentProgression



    totalScore = totalScore/len(dd_resp.keys())

    return totalScore






#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':

    print("No test...")