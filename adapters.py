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

from numpy import std, median
from pronto import Ontology

from writers import write_ref




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


#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':

    from loaders import loader_ontobiotope, get_tags_in_ref, select_subpart_hierarchy, loader_one_bb4_fold, extract_data

    print("loading OntoBiotope...")
    dd_obt = loader_ontobiotope("../BB4/OntoBiotope_BioNLP-OST-2019.obo")
    print("loaded. (Nb of concepts in SCT =", len(dd_obt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_obt)), ")")
    print("\nExtracting Bacterial Habitat hierarchy:")
    dd_habObt = select_subpart_hierarchy(dd_obt, 'OBT:000001')
    print("Done. (Nb of concepts in this subpart of OBT =", len(dd_habObt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_habObt)), ")")

    BioSyn_train_dictionary_adaptater("obt_train_dictionary.txt", dd_habObt)


    print("\n\n")
    

    print("\nLoading BB4 dev corpora...")
    ddd_dataAll = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_dev"])
    dd_habAll = extract_data(ddd_dataAll, l_type=["Habitat"])
    print("loaded.(Nb of mentions in whole corpus =", len(dd_habAll.keys()), ")")
    BioSyn_training_files("processed_dev", ddd_dataAll, l_selectedTypes=["Habitat"])

    print("\nLoading BB4 train corpora...")
    ddd_dataAll = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_train"])
    dd_habAll = extract_data(ddd_dataAll, l_type=["Habitat"])
    print("loaded.(Nb of mentions in whole corpus =", len(dd_habAll.keys()), ")")
    BioSyn_training_files("processed_dev", ddd_dataAll, l_selectedTypes=["Habitat"])