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

def by_heart_matcher(dd_mentions, dd_lesson):
    """
   Description: If a mention has been seen in dd_lesson, then predict same
   """
    dd_predictions = dict()
    nbMentions = len(dd_mentions.keys())
    progresssion = -1

    for i, id in enumerate(dd_mentions.keys()):
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []

    for i, id in enumerate(dd_mentions.keys()):
        mention = dd_mentions[id]["mention"]

        for idByHeart in dd_lesson.keys():
            if dd_lesson[idByHeart]["mention"] == mention:
                dd_predictions[id]["mention"] = mention
                dd_predictions[id]["pred_cui"] = dd_lesson[idByHeart]["cui"]
                break

        # Print progression:
        currentProgression = round(100 * (i / nbMentions))
        if currentProgression > progresssion:
            print(str(currentProgression)+"%", end=" ")
            progresssion = currentProgression

    return dd_predictions



def exact_matcher(dd_mentions, dd_ref):
    """
    Description: Find the first exact match between each mention and label
    :param dd_mentions:
    :param dd_ref:
    :return:
    """
    dd_predictions = dict()
    nbMentions = len(dd_mentions.keys())
    progresssion = -1


    for i, id in enumerate(dd_mentions.keys()):
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []


    for i, id in enumerate(dd_mentions.keys()):
        mention = dd_mentions[id]["mention"]

        for cui in dd_ref.keys():
            l_tags = list()
            l_tags.append(dd_ref[cui]["label"])
            if "tags" in dd_ref[cui].keys():
                for tag in dd_ref[cui]["tags"]:
                    l_tags.append(tag)

            if mention in l_tags:
                dd_predictions[id]["mention"] = dd_mentions[id]["mention"]
                dd_predictions[id]["pred_cui"].append(cui)
                dd_predictions[id]["label"] = dd_ref[cui]["label"]


        #Print progression:
        currentProgression = round(100*(i/nbMentions))
        if currentProgression > progresssion:
            print(currentProgression, "%",)
            progresssion = currentProgression

    return dd_predictions


###
# Maybe improve all rule-based methods by adding training mentions in tags of concepts.
###



def lemmes_exact_matcher():
    return None


def roots_exact_matcher():
    return None


def sieve():
    return None

def pyDNorm():
    return None



###################################################
# Tools:
###################################################

def lowercaser_mentions(dd_mentions):
    for id in dd_mentions.keys():
        dd_mentions[id]["mention"] = dd_mentions[id]["mention"].lower()
    return dd_mentions


def lowercaser_ref(dd_ref):
    for cui in dd_ref.keys():
        dd_ref[cui]["label"] = dd_ref[cui]["label"].lower()
        if "tags" in dd_ref[cui].keys():
            l_lowercasedTags = list()
            for tag in dd_ref[cui]["tags"]:
                l_lowercasedTags.append(tag.lower())
            dd_ref[cui]["tags"] = l_lowercasedTags
    return dd_ref


#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':

    from loaders import loader_snomed_ct_au, loader_amt, loader_all_cadec_folds, extract_one_cadec_fold, loader_all_random_cadec_folds

    ddd_data = loader_all_random_cadec_folds("../CADEC/1_Random_folds_AskAPatient/")
    print("\nddd_data built.")

    dd_amt = loader_amt("../CADEC/AMT_v2.56/Uuid_sct_concepts_au.gov.nehta.amt.standalone_2.56.txt")
    dd_ref = loader_snomed_ct_au("../CADEC/SNOMED_CT_AU_20140531/SnomedCT_Release_AU1000036_20140531/RF2 Release\Snapshot/Terminology/sct2_Description_Snapshot-en-AU_AU1000036_20140531.txt")
    for cui in dd_amt.keys():
        dd_ref[cui] = dd_amt[cui]

    dd_train0 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-0.train")
    dd_train1 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-1.train")
    dd_train2 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-2.train")
    dd_train3 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-3.train")
    dd_train4 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-4.train")
    dd_train5 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-5.train")
    dd_train6 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-6.train")
    dd_train7 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-7.train")
    dd_train8 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-8.train")
    dd_train9 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-9.train")

    dd_test0 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-0.test")
    dd_test1 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-1.test")
    dd_test2 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-2.test")
    dd_test3 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-3.test")
    dd_test4 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-4.test")
    dd_test5 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-5.test")
    dd_test6 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-6.test")
    dd_test7 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-7.test")
    dd_test8 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-8.test")
    dd_test9 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-9.test")


    print("\nLowercase mentions dataset...")
    dd_mentions_test0 = lowercaser_mentions(dd_test0)
    print(dd_mentions_test0)
    print("Done.")

    print("\nLowercase reference...")
    dd_ref = lowercaser_ref(dd_ref)
    print("Done.")

    print("\npredicting...")
    #dd_predictions = exact_matcher(dd_mentions_test0, dd_ref)
    print("\nEM predicted.")
    dd_predictions2 = by_heart_matcher(dd_mentions_test0, dd_train0)
    print(dd_predictions2)
    print("\nEM predicted.")

    print("\nscoring...")
    from evaluators import accuracy
    #score = accuracy(dd_predictions, dd_test0)
    score2 = accuracy(dd_predictions2, dd_test0)
    #print("AccuracyEM=",score)
    print("AccuracyBH=", score2)