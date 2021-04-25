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
import copy

from numpy import std, median, any
from pronto import Ontology

from writers import write_ref

from loaders import loader_ontobiotope, get_tags_in_ref, select_subpart_hierarchy, loader_one_bb4_fold, extract_data
import json
from evaluators import accuracy




#######################################################################################################
# Functions:
#######################################################################################################

def BioSyn_train_dictionary_adaptater(newFile, dd_ref):

    with open(newFile, 'w', encoding="utf8") as newFoldFile:

        for cui in dd_ref.keys():

            line = cui+'||'+dd_ref[cui]["label"].lower()+"\n"
            newFoldFile.write(line)

            if "tags" in dd_ref[cui].keys():
                for tag in dd_ref[cui]["tags"]:
                    if tag != "":
                        line = cui + '||' + tag.lower()+"\n"
                        newFoldFile.write(line)

    print("\nSaved in", newFile, ".")



def BioSyn_training_files(outputDir, ddd_data, l_selectedTypes=[]):

    for file in ddd_data.keys():
        with open(outputDir+'/'+file+".concept", 'w', encoding="utf8") as newFoldFile:

            for id in ddd_data[file].keys():

                if "type" in ddd_data[file][id].keys():
                    typeInfo = ddd_data[file][id]["type"]
                    if len(l_selectedTypes) > 0:
                        if typeInfo not in l_selectedTypes:
                            continue
                else:
                    typeInfo = "UnknownType"


                line = file + "||" + "na" + '|' + "na" + "||" + typeInfo + "||" + ddd_data[file][id]["mention"].lower() + "||"
                for i, cui in enumerate(ddd_data[file][id]["cui"]):
                    if i < (len(ddd_data[file][id]["cui"])-1):
                        line += cui + "|" # WARNING: NOT SURE THAT BioSy TAKE INTO ACCOUNT MULTI-NORM...
                    else:
                        line += cui + "\n"
                newFoldFile.write(line)


        print("\nSaved in", outputDir+'/'+file+".concept", ".")



def BioSyn_json_predictions_adapter(jsonFilePath, dd_mentionsToNormalized):
    dd_pred = dict()
    for id in dd_mentionsToNormalized.keys():
        dd_pred[id] = dict()
        dd_pred[id]["mention"] = dd_mentionsToNormalized[id]["mention"]
        dd_pred[id]["pred_cui"] = []

    with open(jsonFilePath) as f:
        ddd_BioSyn_pred = json.load(f)

    for dld_mention in ddd_BioSyn_pred["queries"]:

        if len(dld_mention["mentions"]) == 1:

            surfaceForm = dld_mention["mentions"][0]["mention"]
            pred_cui = dld_mention["mentions"][0]["candidates"][0]["cui"] #top-ranked candidate

            for id in dd_mentionsToNormalized.keys():
                if dd_mentionsToNormalized[id]["mention"].lower() == surfaceForm.strip():
                    dd_pred[id]["pred_cui"] = [pred_cui]

        else:
            print("2 mentions? meaning?")
            for truc in dld_mention["mentions"]:
                print(truc)

    return dd_pred





def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])


def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i + 1]  # to get acc@(i+1)
                mention_hit += any([candidate['label'] for candidate in candidates])

            # When all mentions in a query are predicted correctly,
            # we consider it as a hit
            if mention_hit == len(mentions):
                hit += 1

        data['acc{}'.format(i + 1)] = hit / len(queries)

    return data



def evaluate_top1_acc(data, dd_dataset, dd_ref):
    queries = data['queries']
    print(len(queries), len(dd_dataset.keys()))

    hit = 0
    for query in queries:
        mentions = query['mentions'] # list containing in general 1 mention, and their candidates CUIs (top20).

        mention_hit = 0
        for mention in mentions:
            candidate = mention['candidates'][0]  # to get acc@(i+1)
            if candidate['label'] == 1:
                hit += 1

                ###
                surfaceForm = mentions[0]["mention"]
                pred_cui = candidate["cui"]  # top-ranked candidate

                for id in dd_dataset.keys():
                    if dd_dataset[id]["mention"].lower() == surfaceForm.strip():
                        dd_pred[id]["pred_cui"] = [pred_cui]

                        if pred_cui != dd_dataset[id]["cui"][0]:
                            print(dd_ref[pred_cui]["label"], end=" --- ")
                            for cui in dd_dataset[id]["cui"]:
                                print(dd_ref[cui]["label"], end=" - ")
                            print("\n")
                # ##



            break

    data['acc{}'.format(1)] = hit / len(queries)

    return data

#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':



    print("loading OntoBiotope...")
    dd_obt = loader_ontobiotope("../BB4/OntoBiotope_BioNLP-OST-2019.obo")
    print("loaded. (Nb of concepts in SCT =", len(dd_obt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_obt)), ")")
    print("\nExtracting Bacterial Habitat hierarchy:")
    dd_habObt = select_subpart_hierarchy(dd_obt, 'OBT:000001')
    print("Done. (Nb of concepts in this subpart of OBT =", len(dd_habObt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_habObt)), ")")

    #BioSyn_train_dictionary_adaptater("obt_train_dictionary.txt", dd_habObt)


    print("\n\n")

    """
    print("\nLoading BB4 dev corpora...")
    ddd_dataAll = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_dev"])
    dd_habAll = extract_data(ddd_dataAll, l_type=["Habitat"])
    print("loaded.(Nb of mentions in whole corpus =", len(dd_habAll.keys()), ")")
    BioSyn_training_files("processed_dev", ddd_dataAll, l_selectedTypes=["Habitat"])

    print("\nLoading BB4 train corpora...")
    ddd_dataAll = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_train"])
    dd_habAll = extract_data(ddd_dataAll, l_type=["Habitat"])
    print("loaded.(Nb of mentions in whole corpus =", len(dd_habAll.keys()), ")")
    BioSyn_training_files("processed_train", ddd_dataAll, l_selectedTypes=["Habitat"])
    """

    print("\n\n")

    ddd_dataDev = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_dev"])
    dd_habDev = extract_data(ddd_dataDev, l_type=["Habitat"])
    print("loaded.(Nb of mentions in dev =", len(dd_habDev.keys()), ")")

    dd_pred = BioSyn_json_predictions_adapter("predictions_eval.json", dd_habDev)

    BioSyn_score_BB4_trainOnVal = accuracy(dd_pred, dd_habDev)
    print("\n\nBioSyn_score_BB4_trainOnVal:", BioSyn_score_BB4_trainOnVal)



    print("\n\n")


    with open("predictions_eval.json") as f:
        ddd_BioSyn_pred = json.load(f)
    data2 = evaluate_top1_acc(ddd_BioSyn_pred, dd_habDev, dd_habObt)
    for key in data2.keys():
        if key != "queries":
            print(key, data2[key])
            break



    # Voir diff entre dd_pred et un autre utilisant les labels=0/1 du ddd_BioSyn_pred:



    sys.exit(0)