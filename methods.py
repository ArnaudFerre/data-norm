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

def exact_matcher(dd_mentions, dd_ref):
    """
    Description: Lowercase labels and mentions ;
    Find the first exact match between each lowercased mention and a label
    :param dd_mentions:
    :param dd_ref:
    :return:
    """
    dd_predictions = dict()

    for id in dd_mentions.keys():
        for cui in dd_ref.keys():

            mention = dd_mentions[id]["mention"].lower()
            label = dd_ref[cui]["label"].lower()

            if mention == label:
                dd_predictions[id] = dict()
                dd_predictions[id]["mention"] = dd_mentions[id]["mention"]
                dd_predictions[id]["pred_cui"] = cui
                dd_predictions[id]["label"] = dd_ref[cui]["label"]

    return dd_predictions







#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':

    print("No test...")