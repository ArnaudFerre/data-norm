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

def loader_snomed_ct_au(descriptionFilePath, languageFilePath, relationFilePath):

    # Extract IsA from relationships files.
    # It seems that 116680003 is the typeId of IsA relation (src: https://confluence.ihtsdotools.org/display/DOCGLOSS/relationship+type)
    # 280844000	71737002
    # moduleId= '900000000000012004', '32506021000036107', '32570491000036106' & 161771000036108 seem to be NOT relevant to CADEC normalization task...
    # Meaning of l_select=['900000000000207008', '900000000000012004', '32506021000036107', '32570491000036106', '161771000036108']

    dd_sct = dict()

    dd_description = dict()
    with open(descriptionFilePath, encoding="utf8") as file:
        i=0
        for line in file:

            if i > 0: #Not taking into account the first line of the file
                l_line = line.split('\t')

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

    with open(filePath, encoding="utf8") as file:

        requestMESH = re.compile('MESH:(.+)$')
        requestOMIM = re.compile('OMIM:(.+)$')

        for line in file:

            if line[0] != '#': #commentary lines

                l_line = line.split('\t')

                cui = l_line[1]
                mMESH = requestMESH.match(cui)
                if mMESH:
                    shortCui = mMESH.group(1)
                else:  # OMIM
                    shortCui = cui

                dd_medic[shortCui]=dict()

                dd_medic[shortCui]["label"] = l_line[0]

                if len(l_line[2]) > 0:
                    dd_medic[shortCui]["alt_cui"] = list()
                    l_altCuis = l_line[2].split('|')
                    for altCui in l_altCuis:
                        mMESH = requestMESH.match(altCui)
                        if mMESH:
                            shortAltCui = mMESH.group(1)
                            dd_medic[shortCui]["alt_cui"].append(shortAltCui)
                        else: #OMIM
                            dd_medic[shortCui]["alt_cui"].append(altCui)

                dd_medic[shortCui]["tags"] = l_line[7].rstrip().split('|')

                if len(l_line[4]) > 0:
                    l_parents = l_line[4].split('|')
                    dd_medic[shortCui]["parents"] = list()
                    for parentCui in l_parents:
                        mMESH = requestMESH.match(parentCui)
                        if mMESH:
                            shortParentCui = mMESH.group(1)
                            dd_medic[shortCui]["parents"].append(shortParentCui)
                        else: #OMIM
                            dd_medic[shortCui]["alt_cui"].append(parentCui)


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




def correct_initial_cadec_2016(inputRep, outputRep):

    for foldFileName in listdir(inputRep):
        foldFilePath = join(inputRep, foldFileName)
        with open(foldFilePath, 'r') as foldFile:

            newFoldFilePath = join(outputRep, foldFileName)
            with open(newFoldFilePath, 'w', encoding="utf8") as newFoldFile:

                for line in foldFile:

                    # 2 errors in the DICLOFENAC-SODIUM.7.ann file ('\t' replaced by many spaces...) in 2016 version:
                    if foldFileName=="DICLOFENAC-SODIUM.7.ann":
                        if line=="TT7     42399005 | Renal failure | 411 415;432 440     renal failure\n":
                            newLine="TT7\t42399005 | Renal failure | 411 415;432 440\trenal failure\n"
                            newFoldFile.write(newLine)
                        elif line=="TT8     409622000 | Respiratory failure | 421 440     renal and respiratory failure \n":
                            newLine="TT8\t409622000 | Respiratory failure | 421 440\trenal and respiratory failure\n"
                            newFoldFile.write(newLine)
                        else:
                            newFoldFile.write(line)

                    # 1 error in LIPITOR.484.ann (missing separator symbol...) in 2016 version:
                    elif foldFileName == "LIPITOR.484.ann" and line == "TT3	76948002 | Severe pain | + 45326000 | Shoulder pain 78 119	severe intense left arm and shoulder pain\n":
                        newFoldFile.write("TT3	76948002 | Severe pain | + 45326000 | Shoulder pain | 78 119	severe intense left arm and shoulder pain\n")

                    # 1 error in LIPITOR.691.ann (missing separator symbol...) in 2016 version:
                    elif foldFileName=="LIPITOR.691.ann" and line=="TT1	76948002 | Severe pain | + 57676002 Arthralgia |  0 18	extreme joint pain\n":
                        newFoldFile.write("TT1	76948002 | Severe pain | + 57676002 | Arthralgia |  0 18	 extreme joint pain\n")

                    # 1 error in LIPITOR.698.ann (missing separator symbol...) in 2016 version:
                    elif foldFileName=="LIPITOR.698.ann" and line=="TT2	76948002 | Severe pain | + 57676002 Arthralgia |  43 72	extreme pain in all my joints\n":
                        newFoldFile.write("TT2	76948002 | Severe pain | + 57676002 | Arthralgia |  43 72	extreme pain in all my joints\n")

                    # 1 error in LIPITOR.772.ann (missing separator symbol...) in 2016 version:
                    elif foldFileName == "LIPITOR.772.ann" and line == "TT1	76948002 | Severe pain |+ 288226003 | Myalgia/myositis - shoulder 0 34	severe muscle pain in my shoulders\n":
                        newFoldFile.write("TT1	76948002 | Severe pain |+ 288226003 | Myalgia/myositis - shoulder | 0 34	severe muscle pain in my shoulders\n")

                    # 3 errors in LIPITOR.6.ann (wrong CUI and missing separator...) in 2016 version:
                    elif foldFileName=="LIPITOR.6.ann":
                        if line=="TT10	81680008 | Neck pain | 514 518;539 543	pain neck\n":
                            newLine="TT10	81680005 | Neck pain | 514 518;539 543	pain neck\n"
                            newFoldFile.write(newLine)
                        elif line=="TT14	81680008 | Neck pain | 838 842;866 870	pain neck\n":
                            newLine="TT14	81680005 | Neck pain | 838 842;866 870	pain neck\n"
                            newFoldFile.write(newLine)
                        elif line=="TT7	76948002 | Severe pain |+ 288226003 | Myalgia/myositis - shoulder 440 458;459 487	severe pain in the muscles in the shoulder area\n":
                            newLine="TT7	76948002 | Severe pain |+ 288226003 | Myalgia/myositis - shoulder | 440 458;459 487	severe pain in the muscles in the shoulder area\n"
                            newFoldFile.write(newLine)
                        else:
                            newFoldFile.write(line)

                    # 1 error in LIPITOR.511.ann (wrong CUI...) in 2016 version:
                    elif foldFileName=="LIPITOR.511.ann" and line=="TT3	67849003 | Excruciating pain | + 20070731 | pain in lower limb |   38 66	excruciating pain in my legs\n":
                        newFoldFile.write("TT3	67849003 | Excruciating pain | + 10601006 | pain in lower limb |   38 66	excruciating pain in my leg\n")

                    # 2 errors in LIPITOR.496.ann (wrong CUI and missing separator symbol...) in 2016 version:
                    elif foldFileName=="LIPITOR.496.ann" and line=="TT9	21499005|Feeling agitated 232 249	Severe aggitation\n":
                        newFoldFile.write("TT9	24199005|Feeling agitated | 232 249	Severe aggitation\n")


                    # Erasing of 3 ambiguous annotations:
                    elif foldFileName == "ARTHROTEC.1.ann" and line == "TT6	102498003 | Agony | or 76948002|Severe pain| 260 265	agony\n":
                        continue
                    elif foldFileName == "ARTHROTEC.140.ann" and line == "TT3	102498003 | Agony | or 76948002|Severe pain| 261 266	agony\n":
                        continue
                    elif foldFileName == "ARTHROTEC.16.ann" and line == "TT8	102498003 | Agony | or 76948002|Severe pain| 73 78	agony\n":
                        continue


                    # 1 error in LIPITOR.574.ann (SCT-AU concept not in the "Clinical finding" <404684003> hierarchy as expected...) in 2016 version:
                    # Applied solution: erase this exemple.
                    elif foldFileName == "LIPITOR.574.ann" and line=="TT10	1806006 | Eruption | 431 445	Abdominal rash\n":
                        continue

                    # 1 error in LIPITOR.72.ann (SCT-AU concept not in the "Clinical finding" <404684003> hierarchy as expected...) in 2016 version:
                    # Applied solution: erase this exemple.
                    elif foldFileName == "LIPITOR.72.ann" and line=="TT9	183202003 | Shin splint | 168 190	felt like shin splints\n":
                        continue

                    # 3 errors in VOLTAREN-XR.3.ann (SCT-AU concept not in the "Clinical finding" <404684003> hierarchy as expected...) in 2016 version:
                    # Applied solution: erase this exemple.
                    # Including 2 wrong CUIs.
                    elif foldFileName == "VOLTAREN-XR.3.ann":
                        if line=="TT4	251377007 | Abdominal pressure | 428 446	abdominal pressure\n":
                            continue
                        elif line == "TT1	28551000168108 | Voltaren | 50 58	Voltaren\n":
                            newFoldFile.write("TT1	3768011000036106 | Voltaren | 50 58	Voltaren\n")
                        elif line == "TT7	28551000168108 | Voltaren | 896 904	Voltaren\n":
                            newFoldFile.write("TT7	3768011000036106 | Voltaren | 896 904	Voltaren\n")
                        else:
                            newFoldFile.write(line)


                    # 1 error in VOLTAREN.5.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.5.ann" and line=="TT2	13481000168104 | Panadeine | 130 139	panadeine\n":
                        newFoldFile.write("TT2	3272011000036106 | Panadeine | 130 139	panadeine\n")

                    # 1 error in LIPITOR.669.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "LIPITOR.669.ann" and line == "TT20	26171000168109 | Lasix | 418 423	lasix\n":
                        newFoldFile.write("TT20	3215011000036108 | Lasix | 418 423	lasix\n")

                    # 1 error in LIPITOR.779.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "LIPITOR.779.ann" and line == "TT8	26171000168109 | Lasix | 395 401	lasiks\n":
                        newFoldFile.write("TT8	3215011000036108 | Lasix | 395 401	lasiks\n")

                    # 1 error in LIPITOR.357.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "LIPITOR.357.ann" and line == "TT8	47401000168109 | Lescol | 658 664	Lescol\n":
                        newFoldFile.write("TT8	3228011000036108 | Lescol | 658 664	Lescol\n")

                    # 1 error in LIPITOR.592.ann (unknown CUI, seem to be wrong CUI for AMT, and wrong label):
                    elif foldFileName == "LIPITOR.592.ann" and line == "TT16	45291000168107 | Efexor-XR | 289 296	Effexor\n":
                        newFoldFile.write("TT16	3288011000036104 | Efexor-XR | 289 296	Efexor\n")

                    # 1 error in DICLOFENAC-SODIUM.4.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "DICLOFENAC-SODIUM.4.ann" and line == "TT1	28551000168108 | Voltaren | 152 160	VOLTAREN\n":
                        newFoldFile.write("TT1	3768011000036106 | Voltaren | 152 160	VOLTAREN\n")

                    # 1 error in PENNSAID.1.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "PENNSAID.1.ann" and line == "TT1	28551000168108 | Voltaren | 397 405	voltaren\n":
                        newFoldFile.write("TT1	3768011000036106 | Voltaren | 397 405	voltaren\n")

                    # 1 error in VOLTAREN-XR.16.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN-XR.16.ann" and line == "TT1	28551000168108 | Voltaren | 202 210	voltaren\n":
                        newFoldFile.write("TT1	3768011000036106 | Voltaren | 202 210	voltaren\n")

                    # 1 error in VOLTAREN-XR.20.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN-XR.20.ann" and line == "TT4	28551000168108 | Voltaren | 17 28	Voltaren-XR\n":
                        newFoldFile.write("TT4	3768011000036106 | Voltaren | 17 28	Voltaren-XR\n")

                    # 2 errors in VOLTAREN-XR.5.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN-XR.5.ann":
                        if line == "TT3	28551000168108 | Voltaren | 49 57	Voltaren\n":
                            newFoldFile.write("TT3	3768011000036106 | Voltaren | 49 57	Voltaren\n")
                        elif line == "TT5	28551000168108 | Voltaren | 301 309	voltaren\n":
                            newFoldFile.write("TT5	3768011000036106 | Voltaren | 301 309	voltaren\n")
                        else:
                            newFoldFile.write(line)

                    # 1 error in VOLTAREN.20.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.20.ann" and line == "TT1	28551000168108 | Voltaren | 84 92	Voltaren\n":
                        newFoldFile.write("TT1	3768011000036106 | Voltaren | 84 92	Voltaren\n")

                    # 3 errors in VOLTAREN-XR.21.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN-XR.21.ann":
                        if line == "TT2	28551000168108 | Voltaren | 9 17	Voltaren\n":
                            newFoldFile.write("TT2	3768011000036106 | Voltaren | 9 17	Voltaren\n")
                        elif line == "TT8	28551000168108 | Voltaren | 386 394	Voltaren\n":
                            newFoldFile.write("TT8	3768011000036106 | Voltaren | 386 394	Voltaren\n")
                        elif line == "TT6	14351000168102 | Seroquel | 394 402	seroquel\n":
                            newFoldFile.write("TT6	3110011000036101 | Seroquel | 394 402	seroquel\n")
                        else:
                            newFoldFile.write(line)

                    # 2 errors in VOLTAREN.25.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.25.ann":
                        if line == "TT7	28551000168108 | Voltaren | 20 28	voltaren\n":
                            newFoldFile.write("TT7	3768011000036106 | Voltaren | 20 28	voltaren\n")
                        elif line=="TT4	59781000168101 | Epipen Auto-Injector | 217 223	epipen\n":
                            newFoldFile.write("TT4	32981011000036102 | Epipen Auto-Injector | 217 223	epipen\n")
                        else:
                            newFoldFile.write(line)

                    # 1 error in VOLTAREN.26.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.26.ann" and line == "TT1	28551000168108 | Voltaren | 45 53	voltaren\n":
                        newFoldFile.write("TT1	3768011000036106 | Voltaren | 45 53	voltaren\n")

                    # 1 error in VOLTAREN.34.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.34.ann" and line == "TT7	28551000168108 | Voltaren | 375 383	Volteren\n":
                        newFoldFile.write("TT7	3768011000036106 | Voltaren | 375 383	Volteren\n")

                    # 1 error in VOLTAREN.37.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.37.ann" and line == "TT3	28551000168108 | Voltaren | 439 447	Voltaren\n":
                        newFoldFile.write("TT3	3768011000036106 | Voltaren | 439 447	Voltaren\n")

                    # 1 error in VOLTAREN.4.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.4.ann" and line == "TT2	28551000168108 | Voltaren | 52 60	voltaren\n":
                        newFoldFile.write("TT2	3768011000036106 | Voltaren | 52 60	voltaren\n")

                    # 4 errors in VOLTAREN.40.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.40.ann":
                        if line == "TT4	28551000168108 | Voltaren | 63 71	voltaren\n":
                            newFoldFile.write("TT4	3768011000036106 | Voltaren | 63 71	voltaren\n")
                        elif line == "TT1	28551000168108 | Voltaren | 239 247	voltaren\n":
                            newFoldFile.write("TT1	3768011000036106 | Voltaren | 239 247	voltaren\n")
                        elif line == "TT14	28551000168108 | Voltaren | 336 344	Voltaren\n":
                            newFoldFile.write("TT14	3768011000036106 | Voltaren | 336 344	Voltaren\n")
                        elif line == "TT2	28551000168108 | Voltaren | 552 560	voltaren\n":
                            newFoldFile.write("TT2	3768011000036106 | Voltaren | 552 560	voltaren\n")
                        else:
                            newFoldFile.write(line)

                    # 1 error in VOLTAREN.46.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.46.ann" and line == "TT5	28551000168108 | Voltaren | 226 234	Voltaren\n":
                        newFoldFile.write("TT5	3768011000036106 | Voltaren | 226 234	Voltaren\n")

                    # 1 error in VOLTAREN.6.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.6.ann" and line == "TT1	28551000168108 | Voltaren | 327 335	voltaren\n":
                        newFoldFile.write("TT1	3768011000036106 | Voltaren | 327 335	voltaren\n")

                    # 1 error in VOLTAREN.7.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.7.ann" and line == "TT7	28551000168108 | Voltaren | 357 365	Voltaren\n":
                        newFoldFile.write("TT7	3768011000036106 | Voltaren | 357 365	Voltaren\n")

                    # 2 errors in VOLTAREN.8.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.8.ann":
                        if line == "TT15	28551000168108 | Voltaren | 659 667	Voltaren\n":
                            newFoldFile.write("TT15	3768011000036106 | Voltaren | 659 667	Voltaren\n")
                        elif line == "TT16	28551000168108 | Voltaren | 735 743	Voltaren\n":
                            newFoldFile.write("TT16	3768011000036106 | Voltaren | 735 743	Voltaren\n")
                        else:
                            newFoldFile.write(line)

                    # 1 error in VOLTAREN.9.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.9.ann" and line == "TT2	28551000168108 | Voltaren | 98 106	Voltaren\n":
                        newFoldFile.write("TT2	3768011000036106 | Voltaren | 98 106	Voltaren\n")

                    # 1 error in VOLTAREN.19.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.19.ann" and line == "TT1	28551000168108 | Voltaren | 34 42	voltaren\n":
                        newFoldFile.write("TT1	3768011000036106 | Voltaren | 34 42	voltaren\n")

                    # 2 errors in VOLTAREN.21.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "VOLTAREN.21.ann":
                        if line == "TT2	28551000168108 | Voltaren | 9 17	Voltaren\n":
                            newFoldFile.write("TT2	3768011000036106 | Voltaren | 9 17	Voltaren\n")
                        elif line == "TT8	28551000168108 | Voltaren | 386 394	Voltaren\n":
                            newFoldFile.write("TT8	3768011000036106 | Voltaren | 386 394	Voltaren\n")
                        else:
                            newFoldFile.write(line)

                    # 1 error in LIPITOR.331.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "LIPITOR.331.ann" and line == "TT6	25701000168107 | Enbrel | 156 173	Enbrel injections\n":
                        newFoldFile.write("TT6	4368011000036108 | Enbrel | 156 173	Enbrel injections\n")

                    # 1 error in ARTHROTEC.68.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "ARTHROTEC.68.ann" and line == "TT7	15611000168108 | Naprosyn | 562 570	Naprosyn\n":
                        newFoldFile.write("TT7	2949011000036106 | Naprosyn | 562 570	Naprosyn\n")

                    # 2 errors in LIPITOR.12.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "LIPITOR.12.ann":
                        if line == "TT12	21290011000036100 | Testosterone | 419 431	testosterone\n":
                            newFoldFile.write("TT12	21290011000036109 | Testosterone | 419 431	testosterone\n")
                        elif line == "TT16	21290011000036100 | Testosterone | 550 562	testosterone\n":
                            newFoldFile.write("TT16	21290011000036109 | Testosterone | 550 562	testosterone\n")
                        else:
                            newFoldFile.write(line)

                    #  1 error in ARTHROTEC.129.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "ARTHROTEC.129.ann" and line == "TT3	2531000168109 | Andrews Tums Antacid | 30 34	Tums\n":
                        newFoldFile.write("TT3	75947011000036105 | Andrews Tums Antacid | 30 34	Tums\n")

                    # 1 error in ARTHROTEC.133.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "ARTHROTEC.133.ann" and line == "TT14	2531000168109 | Andrews Tums Antacid | 751 755	Tums\n":
                        newFoldFile.write("TT14	75947011000036105 | Andrews Tums Antacid | 751 755	Tums\n")

                    # 1 error in LIPITOR.546.ann (unknown CUI, seem to be wrong CUI for AMT):
                    elif foldFileName == "LIPITOR.546.ann" and line == "TT29	31911000168108 | Rhinocort Aqueous | 1580 1589	Rhinocort\n":
                        newFoldFile.write("TT29	39673011000036101 | Rhinocort Aqueous | 1580 1589	Rhinocort\n")

                    # 1 error in VOLTAREN.10.ann (unknown CUI, not sure if it should be voltaren or a more specific product):
                    # (+ modification of label in corpus)
                    elif foldFileName == "VOLTAREN.10.ann" and line == "TT2	38181000168109 | Voltaren Emulgel | 76 88	Voltaren Gel\n":
                        newFoldFile.write("TT2	3768011000036106 | Voltaren | 76 88	Voltaren Gel\n")

                    # 1 error in LIPITOR.721.ann (unknown CUI, not sure if it should be voltaren or a more specific product):
                    elif foldFileName == "LIPITOR.721.ann" and line == "TT4	3481000168101 | Nasonex Aqueous | 140 147	Nasonex\n":
                        newFoldFile.write("TT4	39710011000036107 | Nasonex Aqueous | 140 147	Nasonex\n")

                    # 1 error in VOLTAREN.25.ann (unknown CUI, not sure if it should be voltaren or a more specific product):
                    elif foldFileName == "VOLTAREN.25.ann" and line == "TT12	21290011000036100 | Testosterone | 419 431	testosterone\n":
                        continue

                    # Next are wrong label errors in corpus (not a big problem, use the labels from the reference).
                    # There are others (elements in list inverted, synonym rather that preferred term, label simplification, ...).
                    # Here, it just corrects the really bad labelling (typos, failed parsing, ...).

                    # 1 error in ARTHROTEC.137.ann (typo in label...) in 2016 version:
                    elif foldFileName == "ARTHROTEC.137.ann" and line=="TT6	225014007 | Feeling empty |+ 271681002 | Stomch ache |  249 275	stomach emptiness and pain\n":
                        newFoldFile.write("TT6	225014007 | Feeling empty |+ 271681002 | Stomach ache |  249 275	stomach emptiness and pain\n")

                    # 1 error in ARTHROTEC.91.ann (wrong label...) in 2016 version:
                    elif foldFileName == "ARTHROTEC.91.ann" and line=="TT2	31460011000036109  (substance) 34 46	acidophilous\n":
                        newFoldFile.write("TT2	31460011000036109  lactobacillus acidophilus 34 46	acidophilous\n")

                    else:
                        newFoldFile.write(line)

    print("Fixed Initial CADEC Corpus saved in", outputRep)



def loader_all_initial_cadec_folds(repPath, originalRepPath):
    """
    # If alternative AND multi CUIs, it doesn't work (but seems it never happens).
    Idem with CONCEPT_LESS AND a CUI.
    """
    ddd_data = dict()
    k=0
    i = 0
    for foldFileName in listdir(repPath):
        foldFilePath = join(repPath, foldFileName)

        with open(foldFilePath, 'r', encoding="utf8") as foldFile:

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


                # Find the type of mention in "original" directory of CADEC corpus:
                foldFilePath = join(originalRepPath, foldFileName)
                requestType = re.compile('^([a-zA-Z]+) .+')
                with open(foldFilePath, 'r', encoding="utf8") as originalFile:
                    for originalLine in originalFile:
                        tid = 'T' + originalLine.split("\t")[0]
                        currentTid= l_line[0]
                        if tid == currentTid:
                            mType = requestType.match(originalLine.split("\t")[1])
                            if mType:
                                type = mType.group(1)
                                ddd_data[foldFileNameWithoutExt][exampleId]["type"] = type

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
                    if request11.match(cuis): #Maybe directly delete theses mentions...
                        l_cuis = cuis.split('|')

                        # The first CUI is choose;
                        ddd_data[fileNameWithoutExt][exampleId]["cui"] = [l_cuis[0].strip()]

                        # If supplementary CUIs are alternative CUIs (imply a kind of ambiguity...):
                        ddd_data[fileNameWithoutExt][exampleId]["alt_cui"] = l_cuis[1:]

                    elif request12.match(cuis):
                        l_cuis = cuis.split('+') #multi-normalization
                        ddd_data[fileNameWithoutExt][exampleId]["cui"] = l_cuis
                    else:
                        if cuis.strip()=="MESH:C535662": # FORMATING ERROR in the initial testfold file of the NCBI Disease Corpus...
                            ddd_data[fileNameWithoutExt][exampleId]["cui"] = ["C535662"]
                        else:
                            ddd_data[fileNameWithoutExt][exampleId]["cui"] = [cuis.strip()]

                    i+=1


                request2 = re.compile('^\d*\|a\|')
                if nextAreMentions==False and request2.match(line) is not None:
                    nextAreMentions = True

                request3 = re.compile('^\d*\|t\|')
                if notInDoc==True and request3.match(line) is not None:
                    notInDoc = False


    return ddd_data


def replace_altCui_by_main_cui(s_unknownCuis, dd_medic, txtFileNCBI, fixedTxtFileNCBI):
    # 1/ Search unknown CUIs which are only present in alternative CUIs in MEDIC,
    # And find the main associated CUI for them in MEDIC:
    lc_targetCuisAndToReplaceCuis = set()
    for unknownCui in s_unknownCuis:
        if unknownCui not in dd_medic.keys():

            for cui in dd_medic.keys():
                if "alt_cui" in dd_medic[cui].keys():
                    for altCui in dd_medic[cui]["alt_cui"]:

                        if altCui == unknownCui:
                            lc_targetCuisAndToReplaceCuis.add((cui, unknownCui))

    # 2/ Replace in a new NCBI corpus fold file:
    with open(fixedTxtFileNCBI, 'w', encoding="utf8") as outputFile:
        with open(txtFileNCBI, encoding="utf8") as fileNCBI:

            nbCorrections=0
            nbDeletions=0

            notInDoc = True
            nextAreMentions = False

            for line in fileNCBI:

                newLine = ""

                if line == '\n' and nextAreMentions == True and notInDoc == False:
                    notInDoc = True
                    nextAreMentions = False
                    newLine = line

                if nextAreMentions == True and notInDoc == False:
                    l_line = line.split('\t')

                    # Parfois une liste de CUIs (des '|' ou des '+'):
                    cuis = l_line[5].rstrip()
                    request11 = re.compile('.*\|.*')
                    request12 = re.compile('.*\+.*')


                    # Some CUIs are possible for correct prediction here (ambiguities... so erased):
                    if request11.match(cuis):
                        nbDeletions+=1
                        continue


                    # Multi-norm cases:
                    elif request12.match(cuis):
                        l_cuis = cuis.split('+')  # multi-normalization
                        toModify = False
                        for k, cui in enumerate(l_cuis):
                            for couple in lc_targetCuisAndToReplaceCuis:
                                if couple[1] == cui:
                                    l_cuis[k] = couple[0]
                                    toModify = True

                        if toModify == True:
                            newCuisInLine = ""
                            for k, cui in enumerate(l_cuis):
                                if k == (len(l_cuis) - 1):
                                    newCuisInLine = newCuisInLine + l_cuis[k]
                                else:
                                    newCuisInLine = newCuisInLine + l_cuis[k] + '+'
                            l_line[5] = newCuisInLine + '\n'

                            for i, bloc in enumerate(l_line):
                                if i == (len(l_line) - 1):
                                    newLine = newLine + bloc + '\n'
                                else:
                                    newLine = newLine + bloc + '\t'


                    else:
                        cui = cuis
                        toModify = False
                        for couple in lc_targetCuisAndToReplaceCuis:
                            if couple[1] == cui:
                                l_line[5] = couple[0]
                                toModify = True
                        if toModify == True:
                            for i, bloc in enumerate(l_line):
                                if i == (len(l_line) - 1):
                                    newLine = newLine + bloc + '\n'
                                else:
                                    newLine = newLine + bloc + '\t'

                    if newLine=="":
                        newLine=line
                    else:
                        nbCorrections+=1
                        #print("newLine", newLine, "------ line:", line, "$")


                request2 = re.compile('^\d*\|a\|')
                if nextAreMentions == False and request2.match(line) is not None:
                    nextAreMentions = True
                    newLine = line

                request3 = re.compile('^\d*\|t\|')
                if notInDoc == True and request3.match(line) is not None:
                    notInDoc = False
                    newLine = line


                # New file writing:
                outputFile.write(newLine)

    print("Fixed CBI Disease Corpus saved (", nbCorrections, "corrections and", nbDeletions, "deletions).")


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



def select_subpart_with_patterns_in_label(dd_ref):
    dd_subpart = dict()

    """
    l_metacarac = ['.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '\\', '|', '(', ')']
    """

    # 100% in corpus: "(trade product)", “(medicinal product)”, “(AU substance)”, “(trade product unit of use)” or “(medicinal product unit of use)”.
    request1 = re.compile('(.*) \(trade product\).*')
    request2 = re.compile('(.*) \(medicinal product\).*')
    request3 = re.compile('(.*) \(AU substance\).*')
    request4 = re.compile('(.*) \(trade product unit of use\).*')
    request5 = re.compile('(.*) \(medicinal product unit of use\).*')

    for cui in dd_ref.keys():

        m1 = request1.match(dd_ref[cui]["label"])
        m2 = request2.match(dd_ref[cui]["label"])
        m3 = request3.match(dd_ref[cui]["label"])
        m4 = request4.match(dd_ref[cui]["label"])
        m5 = request5.match(dd_ref[cui]["label"])

        if m1 or m2 or m3 or m4 or m5:
            dd_subpart[cui] = copy.deepcopy(dd_ref[cui])

            if m1:
                tags = m1.group(1)
                dd_subpart[cui]["type"] = "trade product"
            elif m2:
                tags = m2.group(1)
                dd_subpart[cui]["type"] = "medicinal product"
            elif m3:
                tags = m3.group(1)
                dd_subpart[cui]["type"] = "AU substance"
            elif m4:
                tags = m4.group(1)
                dd_subpart[cui]["type"] = "trade product unit of use"
            elif m5:
                tags = m5.group(1)
                dd_subpart[cui]["type"] = "medicinal product unit of use"


            # In the file Uuid_sct_concepts_au.gov.nehta.amt.standalone_2.56.txt, there are '|' separators between tags (need confirmation on this...):
            l_labelAndTags = tags.split(' | ')
            if len(l_labelAndTags) > 1:
                dd_subpart[cui]["tag"] = list()

            for i, tag in enumerate(l_labelAndTags):
                if i == 0:
                    dd_subpart[cui]["label"] = tag
                else:
                    dd_subpart[cui]["tag"].append(tag)

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


    print("loading AMTv2.56...")
    dd_amt = loader_amt("../CADEC/AMT_v2.56/Uuid_sct_concepts_au.gov.nehta.amt.standalone_2.56.txt")
    print("loaded. (Nb of concepts in AMT =", len(dd_amt.keys()),", Nb of tags =", len(get_tags_in_ref(dd_amt)), ")")

    print("\nExtracting sub-AMT:")
    dd_subAmt = select_subpart_with_patterns_in_label(dd_amt)
    print("Done. (Nb of concepts in this subpart AMT =", len(dd_subAmt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_subAmt)), ")")

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
    correct_initial_cadec_2016("../CADEC/0_Initial_CADEC/CADEC_2_2016/sct/", "../CADEC/4_MyCADEC/")
    ddd_data = loader_all_initial_cadec_folds("../CADEC/4_MyCADEC/", "../CADEC/0_Initial_CADEC/CADEC_2_2016/original/")
    dd_initCadecDrugs = extract_data(ddd_data, l_type=["Drug"])
    dd_initCadecClinicalFindings = extract_data(ddd_data, l_type=["ADR", "Finding", "Disease", "Symptom"])
    dd_initCadec = extract_data_without_file(ddd_data)
    print("loaded.(Nb of mentions in initial CADEC =", len(dd_initCadec.keys()),")")
    print("(Nb of drug mentions:", len(dd_initCadecDrugs.keys()), " ; Nb of clinical finding mentions:", len(dd_initCadecClinicalFindings.keys()), ")")

    # NIL
    nbDrugUC=0
    nbClinicalFindingUC=0
    for id in dd_initCadecDrugs.keys():
        if dd_initCadecDrugs[id]["cui"] == ['CONCEPT_LESS']:
            nbDrugUC+=1
    for id in dd_initCadecClinicalFindings.keys():
        if dd_initCadecClinicalFindings[id]["cui"] == ['CONCEPT_LESS']:
            nbClinicalFindingUC+=1
    print("(Nb of CONCEPT_LESS drug mentions:", nbDrugUC, " ; Nb of CONCEPT_LESS clinical finding mentions:", nbClinicalFindingUC, ")")

    # Multi-normalization
    nbDrugMul=0
    nbClinicalFindingMul=0
    for id in dd_initCadecDrugs.keys():
        if len(dd_initCadecDrugs[id]["cui"]) > 1:
            nbDrugMul+=1
    for id in dd_initCadecClinicalFindings.keys():
        if len(dd_initCadecClinicalFindings[id]["cui"]) > 1:
            nbClinicalFindingMul+=1
    print("(Nb of multi-normalized drug mentions:", nbDrugMul, " ; Nb of multi-normalized clinical finding mentions:", nbClinicalFindingMul, ")")

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
    s_cuisInInitCadecDrugs = get_cuis_set_from_corpus(dd_initCadecDrugs)
    s_cuisInInitCadecClinicalFindings = get_cuis_set_from_corpus(dd_initCadecClinicalFindings)
    s_cuisInRandCadec = get_cuis_set_from_corpus(dd_randCadec)
    s_cuisInCustomCadec = get_cuis_set_from_corpus(dd_customCadec)
    print("Loaded.(Nb of distinct used concepts in init/rand/custom =", len(s_cuisInInitCadec), len(s_cuisInRandCadec), len(s_cuisInCustomCadec),")")
    print("(Nb of distinct used concepts in Drug type/CF type =", len(s_cuisInInitCadecDrugs), len(s_cuisInInitCadecClinicalFindings), ")")


    print("\n\nChecking:")

    print("\n\nChecking on full SCT and full AMT:")
    s_unknownCuisFromInit = check_if_cuis_arent_in_ref(s_cuisInInitCadec, dd_ref)
    print("Unknown concepts in initial CADEC:", len(s_unknownCuisFromInit), s_unknownCuisFromInit)
    s_unknownCuisFromRand = check_if_cuis_arent_in_ref(s_cuisInRandCadec, dd_ref)
    print("Unknown concepts in random CADEC:", len(s_unknownCuisFromRand))
    s_unknownCuisFromCustom = check_if_cuis_arent_in_ref(s_cuisInCustomCadec, dd_ref)
    print("Unknown concepts in custom CADEC:", len(s_unknownCuisFromCustom))
    s_unknownCuisFromList = check_if_cuis_arent_in_ref(l_sctFromCadec, dd_ref)
    print("Unknown concepts from [Miftahutdinov et al. 2019] list:", len(s_unknownCuisFromList))

    s_unknownCuisFromInitDrugs = check_if_cuis_arent_in_ref(s_cuisInInitCadecDrugs, dd_subAmt)
    print("Unknown Drug concepts in initial CADEC and subAMT:", len(s_unknownCuisFromInitDrugs), s_unknownCuisFromInitDrugs)
    s_unknownCuisFromInitCF = check_if_cuis_arent_in_ref(s_cuisInInitCadecClinicalFindings, dd_subSct)
    print("Unknown CF concepts in initial CADEC and subSCT:", len(s_unknownCuisFromInitCF), s_unknownCuisFromInitCF)


    print("\n\nChecking on subSCT and full AMT:")
    s_unknownCuisFromInitSub = check_if_cuis_arent_in_ref(s_cuisInInitCadec, dd_subRef)
    print("Unknown concepts in initial CADEC:", len(s_unknownCuisFromInitSub))
    s_unknownCuisFromRandSub = check_if_cuis_arent_in_ref(s_cuisInRandCadec, dd_subRef)
    print("Unknown concepts in random CADEC:", len(s_unknownCuisFromRandSub))
    s_unknownCuisFromCustomSub = check_if_cuis_arent_in_ref(s_cuisInCustomCadec, dd_subRef)
    print("Unknown concepts in custom CADEC:", len(s_unknownCuisFromCustomSub))
    s_unknownCuisFromListSub = check_if_cuis_arent_in_ref(l_sctFromCadec, dd_subRef)
    print("Unknown concepts from [Miftahutdinov et al. 2019] list:", len(s_unknownCuisFromListSub))

    print("Concepts in initial CADEC corpus which are not in the Clinical Finding hierarchy of SCT-AU:")
    for cui in s_unknownCuisFromInitSub:
        if cui not in s_unknownCuisFromInit:
            print(cui, dd_ref[cui]["label"], end=", ")


    print("\n\nChecking on subSCT and subAMT:")
    s_unknownCuisFromInitSubSub = check_if_cuis_arent_in_ref(s_cuisInInitCadec, dd_subsubRef)
    print("Unknown concepts in initial CADEC:", len(s_unknownCuisFromInitSub))
    s_unknownCuisFromRandSubSub = check_if_cuis_arent_in_ref(s_cuisInRandCadec, dd_subsubRef)
    print("Unknown concepts in random CADEC:", len(s_unknownCuisFromRandSub))
    s_unknownCuisFromCustomSubSub = check_if_cuis_arent_in_ref(s_cuisInCustomCadec, dd_subsubRef)
    print("Unknown concepts in custom CADEC:", len(s_unknownCuisFromCustomSub))
    s_unknownCuisFromListSubSub = check_if_cuis_arent_in_ref(l_sctFromCadec, dd_subsubRef)
    print("Unknown concepts from [Miftahutdinov et al. 2019] list:", len(s_unknownCuisFromListSub))

    print("Concepts in initial CADEC corpus which are not in the selected subpart of AMT (see select_subpart_with_patterns_in_label function):")
    for cui in s_unknownCuisFromInitSubSub:
        if cui not in s_unknownCuisFromInitSub:
            print(cui, dd_ref[cui]["label"], end=", ")



    ################################################
    print("\n\n\n\nBB4:\n")
    ################################################

    print("loading OntoBiotope...")
    dd_obt = loader_ontobiotope("../BB4/OntoBiotope_BioNLP-OST-2019.obo")
    print("loaded. (Nb of concepts in SCT =", len(dd_obt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_obt)), ")")

    print("\nExtracting Bacterial Habitat hierarchy:")
    dd_habObt = select_subpart_hierarchy(dd_obt, 'OBT:000001')
    print("Done. (Nb of concepts in this subpart of OBT =", len(dd_habObt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_habObt)), ")")


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
    s_unknownHabCuisTrain = check_if_cuis_arent_in_ref(s_cuisHabTrain, dd_habObt)
    s_unknownHabCuisDev = check_if_cuis_arent_in_ref(s_cuisHabDev, dd_habObt)
    s_unknownHabCuisTrainDev = check_if_cuis_arent_in_ref(s_cuisHabTrainDev, dd_habObt)
    print("\nUnknown concepts in train/dev/train+dev hab corpora:", len(s_unknownHabCuisTrain),len(s_unknownHabCuisDev),len(s_unknownHabCuisTrainDev))



    ################################################
    print("\n\n\n\nNCBI:\n")
    ################################################

    print("\nLoading MEDIC...")
    dd_medic = loader_medic("../NCBI/CTD_diseases_DNorm_v2012_07_6.tsv")
    print("loaded. (Nb of concepts in MEDIC =", len(dd_medic.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_medic)), ")")
    print("Note: Inconsistence errors in MEDIC -> MESH:D006938 = OMIM:144010 and MESH:C537710 = OMIM:153400.")


    print("\n\nInitial NCBI Disease Corpus:\n")

    print("Loading NCBI corpora...")
    ddd_dataFull = loader_one_ncbi_fold(["../NCBI/Voff/NCBItrainset_corpus.txt", "../NCBI/Voff/NCBIdevelopset_corpus.txt", "../NCBI/Voff/NCBItestset_corpus.txt"])
    dd_Full = extract_data(ddd_dataFull, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in full corpus =", len(dd_Full.keys()), ")")

    ddd_dataTrain = loader_one_ncbi_fold(["../NCBI/Voff/NCBItrainset_corpus.txt"])
    dd_Train = extract_data(ddd_dataTrain, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in train corpus =", len(dd_Train.keys()), ")")

    ddd_dataDev = loader_one_ncbi_fold(["../NCBI/Voff/NCBIdevelopset_corpus.txt"])
    dd_Dev = extract_data(ddd_dataDev, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in dev corpus =", len(dd_Dev.keys()), ")")

    ddd_dataTrainDev = loader_one_ncbi_fold(["../NCBI/Voff/NCBItrainset_corpus.txt", "../NCBI/Voff/NCBIdevelopset_corpus.txt"])
    dd_TrainDev = extract_data(ddd_dataTrainDev, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in train+dev corpus =", len(dd_TrainDev.keys()), ")")

    ddd_dataTest = loader_one_ncbi_fold(["../NCBI/Voff/NCBItestset_corpus.txt"])
    dd_Test = extract_data(ddd_dataTest, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in test corpus =", len(dd_Test.keys()), ")")


    print("\nLoading cuis set in corpus...")
    s_cuisNCBIFull = get_cuis_set_from_corpus(dd_Full)
    s_cuisNCBITrain = get_cuis_set_from_corpus(dd_Train)
    s_cuisNCBIDev = get_cuis_set_from_corpus(dd_Dev)
    s_cuisNCBITrainDev = get_cuis_set_from_corpus(dd_TrainDev)
    s_cuisNCBITest = get_cuis_set_from_corpus(dd_Test)
    print("Loaded.(Nb of distinct used concepts in Full/train/dev/train+dev/test NCBI folds =", len(s_cuisNCBIFull),len(s_cuisNCBITrain),len(s_cuisNCBIDev),len(s_cuisNCBITrainDev),len(s_cuisNCBITest),")")


    print("\nChecking:")
    s_unknownNCBICuisFull = check_if_cuis_arent_in_ref(s_cuisNCBIFull, dd_medic)
    s_unknownNCBICuisTrain = check_if_cuis_arent_in_ref(s_cuisNCBITrain, dd_medic)
    s_unknownNCBICuisDev = check_if_cuis_arent_in_ref(s_cuisNCBIDev, dd_medic)
    s_unknownNCBICuisTrainDev = check_if_cuis_arent_in_ref(s_cuisNCBITrainDev, dd_medic)
    s_unknownNCBICuisTest = check_if_cuis_arent_in_ref(s_cuisNCBITest, dd_medic)
    print("Unknown concepts in Full/train/dev/train+dev/test NCBI folds:", len(s_unknownNCBICuisFull),len(s_unknownNCBICuisTrain),len(s_unknownNCBICuisDev),len(s_unknownNCBICuisTrainDev), len(s_unknownNCBICuisTest))
    print("All Unknown concepts:", s_unknownNCBICuisFull)


    print("\n\n")


    print("Fixed NCBI Disease Corpus:\n")


    print("Generate fixed versions of the 3 folds...")
    replace_altCui_by_main_cui(s_unknownNCBICuisTrain, dd_medic, "../NCBI/Voff/NCBItrainset_corpus.txt", "../NCBI/FixedVersion/NCBItrainset_corpus_fixed.txt")
    replace_altCui_by_main_cui(s_unknownNCBICuisDev, dd_medic, "../NCBI/Voff/NCBIdevelopset_corpus.txt", "../NCBI/FixedVersion/NCBIdevelopset_corpus_fixed.txt")
    replace_altCui_by_main_cui(s_unknownNCBICuisTest, dd_medic, "../NCBI/Voff/NCBItestset_corpus.txt", "../NCBI/FixedVersion/NCBItestset_corpus_fixed.txt")


    print("\nLoading Fixed NCBI corpora...")
    ddd_dataFullFixed = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItrainset_corpus_fixed.txt", "../NCBI/FixedVersion/NCBIdevelopset_corpus_fixed.txt", "../NCBI/FixedVersion/NCBItestset_corpus_fixed.txt"])

    dd_FullFixed = extract_data(ddd_dataFullFixed, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in full corpus =", len(dd_FullFixed.keys()), ")")

    ddd_dataTrainFixed = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItrainset_corpus_fixed.txt"])
    dd_TrainFixed = extract_data(ddd_dataTrainFixed, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in train corpus =", len(dd_TrainFixed.keys()), ")")

    ddd_dataDevFixed = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBIdevelopset_corpus_fixed.txt"])
    dd_DevFixed = extract_data(ddd_dataDevFixed, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in dev corpus =", len(dd_DevFixed.keys()), ")")

    ddd_dataTrainDevFixed = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItrainset_corpus_fixed.txt", "../NCBI/FixedVersion/NCBIdevelopset_corpus_fixed.txt"])
    dd_TrainDevFixed = extract_data(ddd_dataTrainDevFixed, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in train+dev corpus =", len(dd_TrainDevFixed.keys()), ")")

    ddd_dataTestFixed = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItestset_corpus_fixed.txt"])
    dd_TestFixed = extract_data(ddd_dataTestFixed, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in test corpus =", len(dd_TestFixed.keys()), ")")


    print("\nLoading cuis set in corpus...")
    s_cuisNCBIFullFixed = get_cuis_set_from_corpus(dd_FullFixed)
    s_cuisNCBITrainFixed = get_cuis_set_from_corpus(dd_TrainFixed)
    s_cuisNCBIDevFixed = get_cuis_set_from_corpus(dd_DevFixed)
    s_cuisNCBITrainDevFixed = get_cuis_set_from_corpus(dd_TrainDevFixed)
    s_cuisNCBITestFixed = get_cuis_set_from_corpus(dd_TestFixed)
    print("Loaded.(Nb of distinct used concepts in Full/train/dev/train+dev/test NCBI folds =", len(s_cuisNCBIFullFixed),len(s_cuisNCBITrainFixed),len(s_cuisNCBIDevFixed),len(s_cuisNCBITrainDevFixed),len(s_cuisNCBITestFixed),")")


    print("\nChecking:")
    s_unknownCuisFixed = check_if_cuis_arent_in_ref(s_cuisNCBIFullFixed, dd_medic)
    print("Unknown concepts in full NCBI corpus:", len(s_unknownCuisFixed))
    print(s_unknownCuisFixed)




