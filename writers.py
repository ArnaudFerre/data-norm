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




#######################################################################################################
# Functions:
#######################################################################################################


def ref_dict_to_obo(dd_ref, filePath):

    with open(filePath, 'w', encoding="utf8") as newFoldFile:

        newFoldFile.write("format - version: 1.2\n\n")

        for cui in dd_ref.keys():

            parents = ""
            if "parents" in dd_ref[cui].keys():
                for parentCui in dd_ref[cui]["parents"]:
                    if parentCui in dd_ref.keys():
                        parents+="is_a: "+parentCui+" ! "+dd_ref[parentCui]["label"]+"\n"

            synonyms = ""
            if "tags" in dd_ref[cui].keys():
                for tag in dd_ref[cui]["tags"]:
                    if tag!="":
                        synonyms+='synonym: "'+tag+'" EXACT []\n'

            conceptInfo = "[Term]\nid: "+cui+"\nname: "+dd_ref[cui]["label"]+"\n"+synonyms+parents+"\n"

            newFoldFile.write(conceptInfo)

    print("Saved in", filePath,".")



def write_ref(dd_ref, filePath):

    with open(filePath, 'w', encoding="utf8") as file:

        line = "CUI" + "\t" + "label" + "\t" + "ParentsCUIs" + "\t" + "tags"  + "\n"
        file.write(line)

        for cui in dd_ref.keys():

            lineLabel = ""
            lineParents = ""
            lineTags = ""

            if "label" in dd_ref[cui].keys():
                lineLabel = dd_ref[cui]["label"]

            if "parents" in dd_ref[cui].keys():
                for i, parentCui in enumerate(dd_ref[cui]["parents"]):
                    if i == len(dd_ref[cui]["parents"])-1 :
                        lineParents += parentCui
                    else:
                        lineParents += parentCui + " | "

            if "tags" in dd_ref[cui].keys():
                for i, tag in enumerate(dd_ref[cui]["tags"]):
                    if i == len(dd_ref[cui]["tags"]) - 1:
                        lineTags += tag
                    else:
                        lineTags += tag + " | "

            line = cui + "\t" + lineLabel + "\t" + lineParents + "\t" + lineTags + "\n"

            file.write(line)

    print("Reference saved in", filePath)



#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':

    from loaders import loader_clinical_finding_file
    dd_subSct = loader_clinical_finding_file("../CADEC/clinicalFindingSubPart.csv")
    ref_dict_to_obo(dd_subSct, "../CADEC/clinicalFindingSubPart.obo")

    from loaders import loader_medic
    dd_medic = loader_medic("../NCBI/CTD_diseases_DNorm_v2012_07_6.tsv")
    ref_dict_to_obo(dd_medic, "../NCBI/CTD_diseases_DNorm_v2012_07_6.obo")


    from pronto import Ontology
    obo_medic = Ontology("../NCBI/CTD_diseases_DNorm_v2012_07_6.obo")

