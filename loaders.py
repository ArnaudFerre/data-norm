# Author: Arnaud Ferré
# RALI, Montreal University
#
# Description :
# Reference = dict() -> {CUI_1: {"label": X1}, {"tags": [Y11, Y12, ..., Y1N]}, {"parents": [CUI_11, ..., CUI_1M]}, {"alt_cui": [alt_CUI_11, ..., alt_CUI_1P]}}
# Minimal attribute: label
# corpus = dict() -> (mentionId_1: {"mention": Z1}, {"cui": [C11, ..., C1Q]}, {"T": "T*"}, {"type": "****"}}
# Minimal attributes: mention and cui



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

###################################################
# Reference loaders:
###################################################

def loader_snomed_ct_au(descriptionFilePath, languageFilePath, relationFilePath, l_select=['900000000000207008', '900000000000012004', '32506021000036107', '32570491000036106', '161771000036108']):

    # Extract IsA from relationships files.
    # It seems that 116680003 is the typeId of IsA relation (src: https://confluence.ihtsdotools.org/display/DOCGLOSS/relationship+type)
    # 280844000	71737002
    # moduleId= '900000000000012004', '32506021000036107', '32570491000036106' & 161771000036108 seem to be NOT relevant to CADEC normalization task...

    dd_sct = dict()

    dd_description = dict()
    with open(descriptionFilePath, encoding="utf8") as file:
        i=0
        for line in file:

            if i > 0: #Not taking into account the first line of the file
                l_line = line.split('\t')

                if l_line[3] in l_select:
                    if l_line[2] == '1':

                        id = l_line[0]
                        dd_description[id] = dict()
                        dd_description[id]["term"] = l_line[7]
                        cui = l_line[4]
                        dd_description[id]["cui"] = cui

                        dd_sct[cui] = dict()
                        dd_sct[cui]["tags"] = list()
                        dd_sct[cui]["parents"] = list()


            i+=1


    with open(languageFilePath, encoding="utf8") as file:
        i = 0
        for line in file:

            if i > 0:  # Not taking into account the first line of the file
                l_line = line.split('\t')

                if l_line[2] == "1":

                    referencedComponentId = l_line[5]
                    acceptabilityId = l_line[6].rstrip()
                    cui = dd_description[referencedComponentId]["cui"]

                    if acceptabilityId == "900000000000548007": #preferred term/label id

                        dd_sct[cui]["label"] = dd_description[referencedComponentId]["term"]

                    elif acceptabilityId == "900000000000549004": #Acceptable term (i.e. tags/synonyms)
                        dd_sct[cui]["tags"].append(dd_description[referencedComponentId]["term"])

            i += 1


    # Some modifications to extract correct data from the SCT-AU v20140531 files:

    # There is also no label for the concept 401103008, and no tags in the language file.
    # But this concept is not in the "Clinical finding" hierarchy, thus, it is just erased:
    del dd_sct['401103008']

    # It seems that there are 62 concepts without preferred term/label in this SCT-AU version (with only active value).
    # Here, a tag is just taken to be the label, but in pratice, no one of these 62 concepts are conserved in the "Clinical finding" hierarchy.
    """
    l_conceptsWithoutLabel=["141461000119106", "141471000119100", "284831000119107", "285801000119101", "288451000119108", "33771000119101", "115176002", "206030004", "268808004", "206064000", "18001006", "52055002", "68983007", "60949004", "53035006", "66215008", "88990005", "42226006", "75291008", "82372009", "73890002", "111463007", "38405004", "65522009", "82587000", "83818003", "67743008", "25932007", "63254003", "20962004", "68122004", "17317007", "69366004", "50968003", "31902002", "80951007", "70165003", "55730009", "22747004", "19691005", "65599008", "28627008", "61362004", "76984009", "81804001", "38425000", "69693005", "63434001", "68213008", "16505001", "30409006", "56192002", "36336001", "111462002", "39918005", "27979008", "79187002", "53516004", "63654005", "62048002", "38942004", "53634006"]
    """
    i = 0
    for cui in dd_sct.keys():
        if "label" not in dd_sct[cui].keys():
            l_tags = dd_sct[cui]["tags"]
            dd_sct[cui]["label"] = l_tags[0]
            dd_sct[cui]["tags"] = l_tags[1:]


    with open(relationFilePath, encoding="utf8") as file:
        i = 0
        for line in file:

            if i > 0:  # Not taking into account the first line of the file
                l_line = line.split('\t')

                if l_line[2] == "1":

                    relationshipGroup = l_line[7]
                    if relationshipGroup == "116680003": # is_a relationship id

                        sourceId = l_line[4]
                        parentId = l_line[5]
                        dd_sct[sourceId]["parents"].append(parentId)

            i += 1


    return dd_sct




def loader_amt(filePath):

    dd_amt = dict()
    if isfile(filePath):
        with open(filePath, encoding="utf8") as file:
            i=0
            for line in file:
                l_line = line.split('\t')
                if i > 0: #Not taking into account the first line of the file

                    if l_line[1] == "0":

                        request = re.compile('.*concept\).*|.*type\).*|.*\(relationship details\).*|.*\(AU qualifier\).*') # metadata concepts
                        if request.match(l_line[2]): # metadata concepts
                            pass
                        else:
                            cui = l_line[0]
                            dd_amt[cui] = dict()
                            dd_amt[cui]["label"] = l_line[2]

                i+=1

    return dd_amt


#########################

def loader_ontobiotope(filePath):
    dd_obt = dict()
    onto = Ontology(filePath)
    for o_concept in onto:
        dd_obt[o_concept.id]=dict()

        dd_obt[o_concept.id]["label"]= o_concept.name

        dd_obt[o_concept.id]["tags"] = list()
        for o_tag in o_concept.synonyms:
            dd_obt[o_concept.id]["tags"].append(o_tag.desc)

        dd_obt[o_concept.id]["parents"] = list()
        for o_parent in o_concept.parents:
            dd_obt[o_concept.id]["parents"].append(o_parent.id)

    return dd_obt


#########################


def loader_medic(filePath):

    dd_medic = dict()

    if isfile(filePath):
        with open(filePath, encoding="utf8") as file:

            for line in file:

                l_line = line.split('\t')

                try:
                    cui = l_line[1]
                    if cui == "DiseaseID":
                        pass
                    else:
                        dd_medic[cui]=dict()

                        dd_medic[cui]["label"] = l_line[0]
                        if len(l_line[2]) > 0:
                            dd_medic[cui]["alt_cui"] = l_line[2].split('|')

                        dd_medic[cui]["tags"] = l_line[7].rstrip().split('|')
                        dd_medic[cui]["parents"] = l_line[4].split('|')

                except:
                    pass


    return dd_medic



###################################################
# Corpus loaders:
###################################################

def loader_all_custom_cadec_folds(repPath):
    """
    Description:
    :param repPath:
    :return:
    """

    ddd_data = dict()

    i = 0
    for foldFileName in listdir(repPath):
        foldFilePath = join(repPath, foldFileName)

        if isfile(foldFilePath):
            with open(foldFilePath) as foldFile:

                foldFileNameWithoutExt = splitext(foldFileName)[0]
                ddd_data[foldFileNameWithoutExt] = dict()

                for line in foldFile:
                    exampleId = "cadec_" + "{number:06}".format(number=i)
                    ddd_data[foldFileNameWithoutExt][exampleId] = dict()

                    mention, cui = line.split('\t')
                    ddd_data[foldFileNameWithoutExt][exampleId]["mention"] = mention
                    ddd_data[foldFileNameWithoutExt][exampleId]["cui"] = [cui.rstrip()] #No multi-norm in CADEC Custom

                    i += 1

    return ddd_data


def extract_one_cadec_fold(ddd_data, foldName):
    dd_data = dict()

    for fold in ddd_data.keys():
        if fold == foldName:
            dd_data = ddd_data[fold] #WARNING: not a deepcopy

    return dd_data



def loader_all_random_cadec_folds(repPath):

    ddd_data = dict()

    i = 0
    for foldFileName in listdir(repPath):
        foldFilePath = join(repPath, foldFileName)

        if isfile(foldFilePath):
            with open(foldFilePath) as foldFile:

                foldFileNameWithoutExt = splitext(foldFileName)[0]
                ddd_data[foldFileNameWithoutExt] = dict()

                for line in foldFile:
                    exampleId = "cadec_" + "{number:06}".format(number=i)
                    ddd_data[foldFileNameWithoutExt][exampleId] = dict()

                    cui, label, mention = line.split('\t')
                    ddd_data[foldFileNameWithoutExt][exampleId]["mention"] = mention.rstrip()
                    ddd_data[foldFileNameWithoutExt][exampleId]["cui"] = [cui.rstrip()] #No multi-norm in CADEC Random
                    ddd_data[foldFileNameWithoutExt][exampleId]["label"] = label

                    i += 1

    return ddd_data



def loader_all_initial_cadec_folds(repPath):
    """
    If alternative AND multi CUIs, it doesn't work (but seems it never happens).
    Idem with CONCEPT_LESS AND a CUI.
    """
    ddd_data = dict()
    k=0
    i = 0
    for foldFileName in listdir(repPath):
        foldFilePath = join(repPath, foldFileName)

        with open(foldFilePath) as foldFile:

            foldFileNameWithoutExt = splitext(foldFileName)[0]
            ddd_data[foldFileNameWithoutExt] = dict()

            for line in foldFile:

                exampleId = "initial_cadec_" + "{number:06}".format(number=i)
                ddd_data[foldFileNameWithoutExt][exampleId] = dict()

                l_line = line.split('\t')

                # NIL cases:
                request1 = re.compile('^CONCEPT\_LESS')
                if request1.match(l_line[1]):
                    l_cui = ["CONCEPT_LESS"]
                    l_label = [None]
                    mention = l_line[2].rstrip()

                else:
                    # Cases where separators '|' are presents:
                    request2 = re.compile('.*[ ]?\|[ ]?.*')
                    if request2.match(l_line[1]):

                        # Multi-norm cases:
                        request3 = re.compile('.*[ ]?\+ [0-9]+.*')
                        if request3.match(l_line[1]):
                            l_cui = list()
                            l_label = list()
                            l_concepts = l_line[1].split('+') # max 3 concepts it seems
                            for j, concept in enumerate(l_concepts):
                                l_cui.append(concept.split('|')[0].strip())
                                l_label.append(concept.split('|')[1].strip())
                            mention = l_line[2].rstrip()

                        else:
                            # Alternative CUI cases:
                            request4 = re.compile('.*\| or [0-9]+.*')
                            if request4.match(l_line[1]):
                                l_cui = [l_line[1].split('|')[0].strip()]
                                l_label = [l_line[1].split('|')[1].strip()]
                                altCui = l_line[1].split('|')[2].strip()
                                altCui = altCui.split("or")[1].strip()
                                l_altCui = [altCui]
                                ddd_data[foldFileNameWithoutExt][exampleId]["alt_cui"] = l_altCui
                                mention = l_line[2].rstrip()

                            # Single-norm cases:
                            else:
                                l_cui = [l_line[1].split('|')[0].strip()]
                                l_label = [l_line[1].split('|')[1].strip()]
                                mention = l_line[2].rstrip()

                    # Cases where separators '|' are NOT presents:
                    # (seem to happen just one time in ARTHROTEC.91.ann)
                    else:
                        l_cui = [l_line[1].split()[0]]
                        l_label = [l_line[1].split()[1]]
                        mention = l_line[2].rstrip()

                ddd_data[foldFileNameWithoutExt][exampleId]["mention"] = mention
                ddd_data[foldFileNameWithoutExt][exampleId]["cui"] = l_cui
                ddd_data[foldFileNameWithoutExt][exampleId]["label"] = l_label

                i += 1


    return ddd_data





# From the list gave by :
def get_cui_list(filePath):
    l_cuis = list()

    if isfile(filePath):
        with open(filePath, encoding="utf8") as file:

            i = 0
            for line in file:
                l_cuis.append(line.rstrip())

    return l_cuis


#########################


def loader_one_bb4_fold(l_repPath):
    """
    Description: WARNING: OK only if A1 file is read before its A2 file.
    :param repPath:
    :return:
    """

    ddd_data = dict()

    i = 0
    for repPath in l_repPath:

        for fileName in listdir(repPath):
            filePath = join(repPath, fileName)

            if isfile(filePath):

                fileNameWithoutExt, ext = splitext(fileName)

                if ext == ".a1":

                    with open(filePath, encoding="utf8") as file:

                        if fileNameWithoutExt not in ddd_data.keys():

                            ddd_data[fileNameWithoutExt] = dict()
                            for line in file:

                                l_line = line.split('\t')

                                if l_line[1].split(' ')[0] == "Title" or l_line[1].split(' ')[0] == "Paragraph":
                                    pass
                                else:
                                    exampleId = "bb4_" + "{number:06}".format(number=i)

                                    ddd_data[fileNameWithoutExt][exampleId] = dict()

                                    ddd_data[fileNameWithoutExt][exampleId]["T"] = l_line[0]
                                    ddd_data[fileNameWithoutExt][exampleId]["type"] = l_line[1].split(' ')[0]
                                    ddd_data[fileNameWithoutExt][exampleId]["mention"] = l_line[2].rstrip()

                                    if "cui" not in ddd_data[fileNameWithoutExt][exampleId].keys():
                                        ddd_data[fileNameWithoutExt][exampleId]["cui"] = list()

                                    i += 1


                elif ext == ".a2":

                    with open(filePath, encoding="utf8") as file:

                        if fileNameWithoutExt in ddd_data.keys():

                            for line in file:
                                l_line = line.split('\t')

                                l_info = l_line[1].split(' ')
                                Tvalue = l_info[1].split(':')[1]

                                for id in ddd_data[fileNameWithoutExt].keys():
                                    if ddd_data[fileNameWithoutExt][id]["T"] == Tvalue :
                                        if ddd_data[fileNameWithoutExt][id]["type"] == "Habitat" or ddd_data[fileNameWithoutExt][id]["type"] == "Phenotype":
                                            cui = "OBT:"+l_info[2].split(':')[2].rstrip()
                                            ddd_data[fileNameWithoutExt][id]["cui"].append(cui)
                                        elif ddd_data[fileNameWithoutExt][id]["type"] == "Microorganism":
                                            cui = l_info[2].split(':')[1].rstrip()
                                            ddd_data[fileNameWithoutExt][id]["cui"] = [cui] #No multi-normalization for microorganisms


    return ddd_data



#########################


def loader_one_ncbi_fold(l_foldPath):
    ddd_data = dict()

    i = 0
    for foldPath in l_foldPath:

        if isfile(foldPath):
            with open(foldPath, encoding="utf8") as file:

                fileNameWithoutExt, ext = splitext(basename(foldPath))
                ddd_data[fileNameWithoutExt] = dict()

                notInDoc = True
                nextAreMentions = False
                for line in file:

                    if line == '\n' and nextAreMentions == True and notInDoc == False:
                        notInDoc = True
                        nextAreMentions = False


                    if nextAreMentions == True and notInDoc == False:
                        l_line = line.split('\t')

                        exampleId = "ncbi_" + "{number:06}".format(number=i)
                        ddd_data[fileNameWithoutExt][exampleId] = dict()

                        ddd_data[fileNameWithoutExt][exampleId]["mention"] = l_line[3]
                        ddd_data[fileNameWithoutExt][exampleId]["type"] = l_line[4]

                        #Parfois une liste de CUIs (des '|' ou des '+'):
                        cuis = l_line[5].rstrip()
                        request11 = re.compile('.*\|.*')
                        request12 = re.compile('.*\+.*')
                        if request11.match(cuis):
                            l_cuis = cuis.split('|')

                            # The first CUI is choose;
                            ddd_data[fileNameWithoutExt][exampleId]["cui"] = [l_cuis[0]]

                            # If supplementary CUIs are alternative CUIs (imply a kind of ambiguity...):
                            ddd_data[fileNameWithoutExt][exampleId]["alt_cui"] = l_cuis[1:]

                        elif request12.match(cuis):
                            l_cuis = cuis.split('+') #multi-normalization
                            ddd_data[fileNameWithoutExt][exampleId]["cui"] = l_cuis
                        else:
                            ddd_data[fileNameWithoutExt][exampleId]["cui"] = [cuis]

                        i+=1


                    request2 = re.compile('^\d*\|a\|')
                    if nextAreMentions==False and request2.match(line) is not None:
                        nextAreMentions = True

                    request3 = re.compile('^\d*\|t\|')
                    if notInDoc==True and request3.match(line) is not None:
                        notInDoc = False


    return ddd_data


###################################################
# Tools:
###################################################

def fusion_ref(dd_ref1, dd_ref2):
    dd_fullRef = dict()
    for cui in dd_ref1.keys():
        dd_fullRef[cui] = dd_ref1[cui]
    for cui in dd_ref2.keys():
        if cui in dd_ref1.keys():
            print("WARNING: same CUIs in both reference! ["+cui+" - "+dd_ref1[cui]["label"]+" / "+dd_ref2[cui]["label"]+")")
        dd_fullRef[cui] = dd_ref2[cui]
    return dd_fullRef


def get_tags_in_ref(dd_ref):
    s_tags = set()
    for cui in dd_ref.keys():
        s_tags.add(dd_ref[cui]["label"])
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                s_tags.add(tag)
    return s_tags


def get_cuis_set_from_corpus(dd_corpus):
    s_cuis = set()
    for id in dd_corpus.keys():
        for cui in dd_corpus[id]["cui"]:
            s_cuis.add(cui)
    return s_cuis

##########################

"""
dd_ref = {'A': {"parents": ['X']}, 'B': {"parents": ['A']}, 'C': {"parents": ['A']}, 'D': {"parents": ['B']},
          'E': {"parents": ['B', 'F']}, 'F': {"parents": ['C']}, 'G': {"parents": ['E'], "label": "G", "tags": ["g", "jay"]}, 'H': {"parents": ['G']},
          'X': {"parents": []}}
"""
def is_desc(dd_ref, cui, cuiParent):
    result = False

    if "parents" in dd_ref[cui].keys():
        if len(dd_ref[cui]["parents"]) > 0:

            # Normal case if no infinite is_a loop:
            if cuiParent in dd_ref[cui]["parents"]:
                result = True
            else:
                for parentCui in dd_ref[cui]["parents"]:
                    result = is_desc(dd_ref, parentCui, cuiParent)
                    if result == True:
                        break

    return result


# 99814 concepts in Clinical Findings hierarchy?
def select_subpart_hierarchy(dd_ref, newRootCui):
    dd_subpart = dict()

    dd_subpart[newRootCui] = dd_ref[newRootCui]
    dd_subpart[newRootCui]["parents"] = []

    for cui in dd_ref.keys():
        if is_desc(dd_ref, cui, newRootCui)== True:

            dd_subpart[cui] = copy.deepcopy(dd_ref[cui])


    # Clear parents which are not in the descendants of the new root:
    for cui in dd_subpart.keys():
        dd_subpart[cui]["parents"] = list()
        for parentCui in dd_ref[cui]["parents"]:
            if is_desc(dd_ref, parentCui, newRootCui) or parentCui==newRootCui:
                dd_subpart[cui]["parents"].append(parentCui)


    return dd_subpart



def select_subpart_with_patterns_in_label(dd_ref, l_patterns=["(trade product)", "(medicinal product)"]):
    dd_subpart = dict()

    """
    l_metacarac = ['.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '\\', '|', '(', ')']
    print(l_metacarac)
    request = re.compile('')
    """

    # 100%: “(medicinal product)”, “(AU substance)”, “(trade product unit of use)” or “(medicinal product unit of use)”.
    request1 = re.compile('.*\(trade product\).*')
    request2 = re.compile('.*\(medicinal product\).*')
    request3 = re.compile('.*\(AU substance\).*')
    request4 = re.compile('.*\(trade product unit of use\).*')
    request5 = re.compile('.*\(medicinal product unit of use\).*')
    for cui in dd_ref.keys():
        if request1.match(dd_ref[cui]["label"]) or request2.match(dd_ref[cui]["label"]) or request3.match(dd_ref[cui]["label"]) or request4.match(dd_ref[cui]["label"]) or request5.match(dd_ref[cui]["label"]):
            dd_subpart[cui] = copy.deepcopy(dd_ref[cui])

    # Clear the labels from its pattern:



    return dd_subpart








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




##########################

def fusion_folds(l_dd_folds):
    dd_data = dict()

    for dd_fold in l_dd_folds:

        for id in dd_fold.keys():
            dd_data[id] = dict()

        for id in dd_fold.keys():
            dd_data[id]["mention"] = dd_fold[id]["mention"]
            dd_data[id]["cui"] = dd_fold[id]["cui"]

    return dd_data



def extract_data(ddd_data, l_type=[]):
    dd_data = dict()

    for fileName in ddd_data.keys():
        for id in ddd_data[fileName].keys():
            if ddd_data[fileName][id]["type"] in l_type:
                dd_data[id] = ddd_data[fileName][id]

    return dd_data



def extract_data_without_file(ddd_data):
    dd_data = dict()
    for file in ddd_data.keys():
        for id in ddd_data[file].keys():
            dd_data[id] = ddd_data[file][id]
    return dd_data







###################################################
# Checking:
###################################################
def check_if_cuis_arent_in_ref(s_cuis, dd_ref):
    s_unknowCuis = set()
    for cui in s_cuis:
        if cui not in dd_ref.keys():
            s_unknowCuis.add(cui)
    return s_unknowCuis



###################################################
# Printers:
###################################################


#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':


    ################################################
    print("\n\n\nCADEC (3 datasets):\n")
    ################################################

    print("loading SCT-AUv20140531...")
    dd_sct = loader_snomed_ct_au("../CADEC/SNOMED_CT_AU_20140531/SnomedCT_Release_AU1000036_20140531/RF2 Release/Snapshot/Terminology/sct2_Description_Snapshot-en-AU_AU1000036_20140531.txt",
                                 "../CADEC/SNOMED_CT_AU_20140531/SnomedCT_Release_AU1000036_20140531/RF2 Release/Snapshot/Refset/Language/der2_cRefset_LanguageSnapshot-en-AU_AU1000036_20140531.txt",
                                 "../CADEC/SNOMED_CT_AU_20140531/SnomedCT_Release_AU1000036_20140531/RF2 Release/Snapshot/Terminology/sct2_Relationship_Snapshot_AU1000036_20140531.txt")
    print("loaded. (Nb of concepts in SCT =", len(dd_sct.keys()),", Nb of tags =", len(get_tags_in_ref(dd_sct)), ")")

    print("\nExtracting subpart Clinical Finding hierarchy:")
    dd_subSct = select_subpart_hierarchy(dd_sct, '404684003')
    print("Done. (Nb of concepts in this subpart of SCT =", len(dd_subSct.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_subSct)), ")")

    write_ref(dd_subSct, "../CADEC/clinicalFindingSubPart.csv")

    sys.exit(0)



    print("loading AMTv2.56...")
    dd_amt = loader_amt("../CADEC/AMT_v2.56/Uuid_sct_concepts_au.gov.nehta.amt.standalone_2.56.txt")
    print("loaded. (Nb of concepts in AMT =", len(dd_amt.keys()),", Nb of tags =", len(get_tags_in_ref(dd_amt)), ")")

    print("\nExtracting all Trade Product from AMT:")
    l_pat = ["(trade product)"] #["(trade product)", "(medicinal product)"]
    dd_subAmt = select_subpart_with_patterns_in_label(dd_amt, l_patterns=l_pat)
    print("Done. (Nb of concepts in this subpart AMT (", l_pat, ") =", len(dd_subAmt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_subAmt)), ")")


    request1 = re.compile('.+(\(.+\))$')
    s_types = set()
    for i, cui in enumerate(dd_amt.keys()):
        m = request1.match(dd_amt[cui]["label"])
        if m:
            s_types.add(m.group(1))

    print(len(s_types), s_types)









    sys.exit(0)



    print("\nFusion SCT & AMT in one reference...")
    dd_ref = fusion_ref(dd_sct, dd_amt)
    print("done. (Nb of concepts in SCT+AMT =", len(dd_ref.keys()),", Nb of tags =", len(get_tags_in_ref(dd_ref)), ")")

    print("\nFusion subSCT & AMT in one reference...")
    dd_subRef = fusion_ref(dd_subSct, dd_amt)
    print("done. (Nb of concepts in subSCT+AMT =", len(dd_subRef.keys()),", Nb of tags =", len(get_tags_in_ref(dd_subRef)), ")")

    print("\nFusion subSCT & subAMT in one reference...")
    dd_subsubRef = fusion_ref(dd_subSct, dd_subAmt)
    print("done. (Nb of concepts in subSCT+subAMT =", len(dd_subsubRef.keys()),", Nb of tags =", len(get_tags_in_ref(dd_subsubRef)), ")")

    print("\nLoading CUIs list used by [Miftahutdinov et al. 2019]...")
    l_sctFromCadec = get_cui_list("../CADEC/custom_CUI_list.txt")
    print("loaded.(Nb of concepts in the list (with CONCEPT_LESS =", len(l_sctFromCadec),")")


    print("\n\nLoading initial CADEC corpus...")
    ddd_data = loader_all_initial_cadec_folds("../CADEC/0_Original_CADEC/AMT-SCT/")
    dd_initCadec = extract_data_without_file(ddd_data)
    print("loaded.(Nb of mentions in initial CADEC =", len(dd_initCadec.keys()),")")

    print("\nLoading random CADEC corpus...")
    ddd_randData = loader_all_random_cadec_folds("../CADEC/1_Random_folds_AskAPatient/")
    dd_randCadec = extract_data_without_file(ddd_randData)
    print("loaded.(Nb of mentions in ALL folds for random CADEC =", len(dd_randCadec.keys()),")")

    print("\nLoading custom CADEC corpus...")
    ddd_customData = loader_all_custom_cadec_folds("../CADEC/2_Custom_folds/")
    dd_customCadec = extract_data_without_file(ddd_customData)
    print("loaded.(Nb of mentions in ALL folds for custom CADEC =", len(dd_customCadec.keys()),")")


    print("\n\nLoading cuis set in corpus...")
    s_cuisInInitCadec = get_cuis_set_from_corpus(dd_initCadec)
    s_cuisInRandCadec = get_cuis_set_from_corpus(dd_randCadec)
    s_cuisInCustomCadec = get_cuis_set_from_corpus(dd_customCadec)
    print("Loaded.(Nb of distinct used concepts in init/rand/custom =", len(s_cuisInInitCadec), len(s_cuisInRandCadec), len(s_cuisInCustomCadec),")")


    print("\n\nChecking:")

    s_unknownCuisFromInit = check_if_cuis_arent_in_ref(s_cuisInInitCadec, dd_ref)
    print("\nUnknown concepts in initial CADEC:", len(s_unknownCuisFromInit))
    s_unknownCuisFromRand = check_if_cuis_arent_in_ref(s_cuisInRandCadec, dd_ref)
    print("Unknown concepts in random CADEC:", len(s_unknownCuisFromRand))
    s_unknownCuisFromCustom = check_if_cuis_arent_in_ref(s_cuisInCustomCadec, dd_ref)
    print("Unknown concepts in custom CADEC:", len(s_unknownCuisFromCustom))
    s_unknownCuisFromList = check_if_cuis_arent_in_ref(l_sctFromCadec, dd_ref)
    print("Unknown concepts from [Miftahutdinov et al. 2019] list:", len(s_unknownCuisFromList))

    print("s_unknownCuisFromInit:",s_unknownCuisFromInit)


    print("\n\nChecking on subSCT and full AMT:")

    s_unknownCuisFromInitSub = check_if_cuis_arent_in_ref(s_cuisInInitCadec, dd_subRef)
    print("\nUnknown concepts in initial CADEC:", len(s_unknownCuisFromInitSub))
    s_unknownCuisFromRandSub = check_if_cuis_arent_in_ref(s_cuisInRandCadec, dd_subRef)
    print("Unknown concepts in random CADEC:", len(s_unknownCuisFromRandSub))
    s_unknownCuisFromCustomSub = check_if_cuis_arent_in_ref(s_cuisInCustomCadec, dd_subRef)
    print("Unknown concepts in custom CADEC:", len(s_unknownCuisFromCustomSub))
    s_unknownCuisFromListSub = check_if_cuis_arent_in_ref(l_sctFromCadec, dd_subRef)
    print("Unknown concepts from [Miftahutdinov et al. 2019] list:", len(s_unknownCuisFromListSub))

    print("s_unknownCuisFromInitSub:", s_unknownCuisFromInitSub)


    print("dif subSCT//SCT:")
    for cui in s_unknownCuisFromInitSub:
        if cui not in s_unknownCuisFromInit:
            print(cui)


    print("\n\nChecking on subSCT and subAMT:")

    s_unknownCuisFromInitSubSub = check_if_cuis_arent_in_ref(s_cuisInInitCadec, dd_subsubRef)
    print("\nUnknown concepts in initial CADEC:", len(s_unknownCuisFromInitSub))
    s_unknownCuisFromRandSubSub = check_if_cuis_arent_in_ref(s_cuisInRandCadec, dd_subsubRef)
    print("Unknown concepts in random CADEC:", len(s_unknownCuisFromRandSub))
    s_unknownCuisFromCustomSubSub = check_if_cuis_arent_in_ref(s_cuisInCustomCadec, dd_subsubRef)
    print("Unknown concepts in custom CADEC:", len(s_unknownCuisFromCustomSub))
    s_unknownCuisFromListSubSub = check_if_cuis_arent_in_ref(l_sctFromCadec, dd_subsubRef)
    print("Unknown concepts from [Miftahutdinov et al. 2019] list:", len(s_unknownCuisFromListSub))


    print("dif subSubSCTAMT//full:")
    i=0
    for cui in s_unknownCuisFromInitSubSub:
        if cui not in s_unknownCuisFromInit:
            try:
                print("-", cui, dd_ref[cui]["label"])
            except:
                print(cui)
            i+=1
    print(i)




    sys.exit(0)

    print(s_unknownCuisFromInit)
    print(s_unknownCuisFromRand)
    print("---------------------")
    for cui in s_unknownCuisFromRand:
        if cui not in s_unknownCuisFromInit:
            print(cui)

    for file in ddd_randData.keys():
        for id in ddd_randData[file].keys():
            for cui in ddd_randData[file][id]["cui"]:
                if cui == "21499005" or cui == "81680008":
                    print(file, id, dd_randCadec[id])



    ################################################
    print("\n\n\n\nBB4:\n")
    ################################################

    print("loading OntoBiotope...")
    dd_obt = loader_ontobiotope("../BB4/OntoBiotope_BioNLP-OST-2019.obo")
    print("loaded. (Nb of concepts in SCT =", len(dd_obt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_obt)), ")")


    print("\nLoading BB4 corpora...")
    ddd_dataAll = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_train", "../BB4/BioNLP-OST-2019_BB-norm_dev", "../BB4/BioNLP-OST-2019_BB-norm_test"])
    dd_habAll = extract_data(ddd_dataAll, l_type=["Habitat"])
    print("loaded.(Nb of mentions in whole corpus =", len(dd_habAll.keys()), ")")

    ddd_dataTrain = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_train"])
    dd_habTrain = extract_data(ddd_dataTrain, l_type=["Habitat"]) # ["Habitat", "Phenotype", "Microorganism"]
    print("loaded.(Nb of mentions in train =", len(dd_habTrain.keys()), ")")

    ddd_dataDev = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_dev"])
    dd_habDev = extract_data(ddd_dataDev, l_type=["Habitat"])
    print("loaded.(Nb of mentions in dev =", len(dd_habDev.keys()), ")")

    ddd_dataTrainDev = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_train", "../BB4/BioNLP-OST-2019_BB-norm_dev"])
    dd_habTrainDev = extract_data(ddd_dataTrainDev, l_type=["Habitat"])
    print("loaded.(Nb of mentions in train+dev =", len(dd_habTrainDev.keys()), ")")

    ddd_dataTest = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_test"])
    dd_habTest = extract_data(ddd_dataTest, l_type=["Habitat"])
    print("loaded.(Nb of mentions in test =", len(dd_habTest.keys()), ")")


    print("\nLoading cuis set in corpus...")
    s_cuisHabTrain = get_cuis_set_from_corpus(dd_habTrain)
    s_cuisHabDev = get_cuis_set_from_corpus(dd_habDev)
    s_cuisHabTrainDev = get_cuis_set_from_corpus(dd_habTrainDev)
    print("Loaded.(Nb of distinct used concepts in train/dev/train+dev hab corpora =", len(s_cuisHabTrain),len(s_cuisHabDev),len(s_cuisHabTrainDev),")")


    print("\nChecking:")
    s_unknownHabCuisTrain = check_if_cuis_arent_in_ref(s_cuisHabTrain, dd_obt)
    s_unknownHabCuisDev = check_if_cuis_arent_in_ref(s_cuisHabDev, dd_obt)
    s_unknownHabCuisTrainDev = check_if_cuis_arent_in_ref(s_cuisHabTrainDev, dd_obt)
    print("\nUnknown concepts in train/dev/train+dev hab corpora:", len(s_unknownHabCuisTrain),len(s_unknownHabCuisDev),len(s_unknownHabCuisTrainDev))



    ################################################
    print("\n\n\n\nNCBI:\n")
    ################################################

    print("loading MEDIC...")
    dd_medic = loader_medic("../NCBI/FixedVersion/CTD_diseases_DNorm_v2012_07_6_fixed.tsv")
    print("loaded. (Nb of concepts in SCT =", len(dd_medic.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_medic)), ")")


    print("\nLoading NCBI corpora...")
    ddd_dataFull = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItrainset_corpus_fixed.txt", "../NCBI/FixedVersion/NCBIdevelopset_corpus.txt", "../NCBI/FixedVersion/NCBItestset_corpus_fixed.txt"])
    dd_Full = extract_data(ddd_dataFull, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in full corpus =", len(dd_Full.keys()), ")")

    ddd_dataTrain = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItrainset_corpus_fixed.txt"])
    dd_Train = extract_data(ddd_dataTrain, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in train corpus =", len(dd_Train.keys()), ")")

    ddd_dataDev = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBIdevelopset_corpus.txt"])
    dd_Dev = extract_data(ddd_dataDev, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in dev corpus =", len(dd_Dev.keys()), ")")

    ddd_dataTrainDev = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItrainset_corpus_fixed.txt", "../NCBI/FixedVersion/NCBIdevelopset_corpus.txt"])
    dd_TrainDev = extract_data(ddd_dataTrainDev, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in train+dev corpus =", len(dd_TrainDev.keys()), ")")

    ddd_dataTest = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItestset_corpus_fixed.txt"])
    dd_Test = extract_data(ddd_dataTest, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in test corpus =", len(dd_Test.keys()), ")")


    print("\nLoading cuis set in corpus...")
    s_cuis = get_cuis_set_from_corpus(dd_Full)
    print("Loaded.(Nb of distinct used concepts in full corpus =", len(s_cuis), ")")


    print("\nChecking:")
    s_unknownCuis = check_if_cuis_arent_in_ref(s_cuisInInitCadec, dd_medic)
    print("\nUnknown concepts in full NCBI corpus:", len(s_unknownCuis))



