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

###################################################
# Reference loaders:
###################################################

def loader_snomed_ct_au(descriptionFilePath, languageFilePath, l_select=['900000000000207008', '900000000000012004', '32506021000036107', '32570491000036106', '161771000036108'], l_keptActivity=["0","1"]):

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
                    if l_line[2] in l_keptActivity:

                        id = l_line[0]
                        dd_description[id] = dict()
                        dd_description[id]["term"] = l_line[7]
                        cui = l_line[4]
                        dd_description[id]["cui"] = cui

                        dd_sct[cui] = dict()
                        dd_sct[cui]["tags"] = list()

            i+=1


    with open(languageFilePath, encoding="utf8") as file:
        i = 0
        for line in file:

            if i > 0:  # Not taking into account the first line of the file
                l_line = line.split('\t')

                if l_line[2] in l_keptActivity:

                    referencedComponentId = l_line[5]
                    acceptabilityId = l_line[6].rstrip()
                    cui = dd_description[referencedComponentId]["cui"]

                    if acceptabilityId == "900000000000548007": #preferred term/label id
                        dd_sct[cui]["label"] = dd_description[referencedComponentId]["term"]

                    elif acceptabilityId == "900000000000549004": #Acceptable term (i.e. tags/synonyms)
                        dd_sct[cui]["tags"].append(dd_description[referencedComponentId]["term"])

            i += 1

    # It seems that there are 6 concepts without preferred term/label in this SCT-AU version.
    # All these 6 have only one tag, so we just swap the label and the tag.
    for cui in dd_sct.keys():
        if "label" not in dd_sct[cui].keys():
            try:
                dd_sct[cui]["label"] = dd_sct[cui]["tags"][0]
                dd_sct[cui]["tags"] = []
            except:
                print(cui, dd_sct[cui])

    return dd_sct




def loader_amt(filePath):

    s_set = set()

    dd_sct = dict()
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
                            dd_sct[cui] = dict()
                            dd_sct[cui]["label"] = l_line[2]

                i+=1

    return dd_sct



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
                                            cui = l_info[2].split(':')[2].rstrip()
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
    print("\n\n\n\nCADEC (3 datasets):\n")
    ################################################

    print("loading SCT-AUv20140531...")
    dd_sct = loader_snomed_ct_au("../CADEC/SNOMED_CT_AU_20140531/SnomedCT_Release_AU1000036_20140531/RF2 Release/Snapshot/Terminology/sct2_Description_Snapshot-en-AU_AU1000036_20140531.txt",
                                 "../CADEC/SNOMED_CT_AU_20140531/SnomedCT_Release_AU1000036_20140531/RF2 Release/Snapshot/Refset/Language/der2_cRefset_LanguageSnapshot-en-AU_AU1000036_20140531.txt",
                                 l_keptActivity=["1"])#
    print("loaded. (Nb of concepts in SCT =", len(dd_sct.keys()),", Nb of tags =", len(get_tags_in_ref(dd_sct)), ")")

    print("loading AMTv2.56...")
    dd_amt = loader_amt("../CADEC/AMT_v2.56/Uuid_sct_concepts_au.gov.nehta.amt.standalone_2.56.txt")
    print("loaded. (Nb of concepts in AMT =", len(dd_amt.keys()),", Nb of tags =", len(get_tags_in_ref(dd_amt)), ")")

    print("\nFusion SCT & AMT in one reference...")
    dd_ref = fusion_ref(dd_sct, dd_amt)
    print("done. (Nb of concepts in SCT+AMT =", len(dd_ref.keys()),", Nb of tags =", len(get_tags_in_ref(dd_ref)), ")")

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
    ddd_data = loader_all_custom_cadec_folds("../CADEC/2_Custom_folds/")
    dd_customCadec = extract_data_without_file(ddd_data)
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












    sys.exit(0)


    print("len(dd_sct):", len(dd_sct), "\nlen(l_sctFromCadec):", len(l_sctFromCadec))


    s_nil = set()
    l_cuis = dd_sct.keys()
    for cuiCadec in l_sctFromCadec:
        if cuiCadec not in l_cuis:
            s_nil.add(cuiCadec)

    print(len(s_nil), s_nil)




    ################################################
    print("\n\n\n\nRandom CADEC\n")
    ################################################

    ddd_data = loader_all_random_cadec_folds("../CADEC/1_Random_folds_AskAPatient/")
    print("\nddd_data built.")

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

    dd_dev0 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-0.validation")
    dd_dev1 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-1.validation")
    dd_dev2 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-2.validation")
    dd_dev3 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-3.validation")
    dd_dev4 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-4.validation")
    dd_dev5 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-5.validation")
    dd_dev6 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-6.validation")
    dd_dev7 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-7.validation")
    dd_dev8 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-8.validation")
    dd_dev9 = extract_one_cadec_fold(ddd_data, "AskAPatient.fold-9.validation")

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

    print("Unitary folds built, ex: dd_test0:", dd_test0)


    s_cuisInRandomCadec = set()
    for file in ddd_data.keys():
        for id in ddd_data[file]:
            for cui in ddd_data[file][id]["cui"]:
                s_cuisInRandomCadec.add(cui)

    print("len(s_cuisInRandomCadec):", len(s_cuisInRandomCadec))



    ################################################
    print("\n\n\n\nInitial CADEC\n")
    ################################################

    ddd_data = loader_all_initial_cadec_folds("../CADEC/0_Original_CADEC/AMT-SCT/")
    dd_cadec = extract_data_without_file(ddd_data)

    print("test ddd_data['ARTHROTEC.32']:", ddd_data["ARTHROTEC.32"])

    s_cuis = set()
    l_altCuis = list()
    compt=0
    cuiLess = 0
    for file in ddd_data.keys():
        for id in ddd_data[file].keys():
            for cui in ddd_data[file][id]["cui"]:
                s_cuis.add(cui)
            if "alt_cui" in ddd_data[file][id].keys():
                for altCui in ddd_data[file][id]["alt_cui"]:
                    s_cuis.add(altCui)
                    l_altCuis.append(altCui)
            if len(ddd_data[file][id]["cui"]) > 1:
                compt+=1
            if ddd_data[file][id]["cui"] == ["CONCEPT_LESS"]:
                cuiLess+=1

    print("len(s_cuis):", len(s_cuis))
    print("len(l_altCuis):", len(l_altCuis), l_altCuis)
    print("multi", compt, "cui LSS:", cuiLess)






    #Load AMT:
    dd_amt = loader_amt("../CADEC/AMT_v2.56/Uuid_sct_concepts_au.gov.nehta.amt.standalone_2.56.txt")

    s_amt = set()
    for cui in dd_amt.keys():
        s_amt.add(cui)

    print(len(s_amt), len(dd_amt.keys()))


    print("\n\n")

    cmpInSCT=0
    cmpInAMT = 0
    conceptLess = 0
    Nbmentions = 0
    wtf = 0
    s1 = set()
    s2 = set()
    s3 = set()
    for file in ddd_data.keys():
        for id in ddd_data[file].keys():
            Nbmentions+=1
            for cui in ddd_data[file][id]["cui"]:
                if cui in dd_sct.keys() and cui!="CONCEPT_LESS":
                    cmpInSCT +=1
                    s1.add(cui)
                elif cui in dd_amt.keys() and cui!="CONCEPT_LESS":
                    cmpInAMT +=1
                    s2.add(cui)
                elif cui=="CONCEPT_LESS":
                    conceptLess += 1
                else:
                    wtf+=1
                    s3.add(cui)
                    print(file, ddd_data[file][id])




    print(cmpInSCT, cmpInAMT, conceptLess, Nbmentions, wtf)
    print(len(s1), len(s2), len(s3))




    get_log(dd_cadec, dd_cadec, tag1="Full", tag2="Full")





    sys.exit(0)

    ddd_data = loader_all_cadec_folds("../CADEC/2_Custom_folds/")
    print("\nddd_data built.")

    dd_train0 = extract_one_cadec_fold(ddd_data, "train_0")
    dd_train1 = extract_one_cadec_fold(ddd_data, "train_1")
    dd_train2 = extract_one_cadec_fold(ddd_data, "train_2")
    dd_train3 = extract_one_cadec_fold(ddd_data, "train_3")
    dd_train4 = extract_one_cadec_fold(ddd_data, "train_4")

    dd_test0 = extract_one_cadec_fold(ddd_data, "test_0")
    dd_test1 = extract_one_cadec_fold(ddd_data, "test_1")
    dd_test2 = extract_one_cadec_fold(ddd_data, "test_2")
    dd_test3 = extract_one_cadec_fold(ddd_data, "test_3")
    dd_test4 = extract_one_cadec_fold(ddd_data, "test_4")

    print("Unitary folds built, ex: dd_test0:", dd_test0)

    dd_train_data = fusion_folds([dd_train0, dd_train1, dd_train2, dd_train3, dd_train4])
    dd_test_data = fusion_folds([dd_test0, dd_test1, dd_test2, dd_test3, dd_test4])

    print("Full train fold, and full test fold, built.")

    dd_train_test_0 = fusion_folds([dd_train0, dd_test0])
    dd_train_test_1 = fusion_folds([dd_train1, dd_test1])
    dd_train_test_2 = fusion_folds([dd_train2, dd_test2])
    dd_train_test_3 = fusion_folds([dd_train3, dd_test3])
    dd_train_test_4 = fusion_folds([dd_train4, dd_test4])

    print("Train/test folds built, ex: dd_train_test_0:", dd_train_test_0)

    dd_full = fusion_folds([dd_train_test_0, dd_train_test_1, dd_train_test_2, dd_train_test_3, dd_train_test_4])


    get_log(dd_full, dd_test_data, tag1="Full", tag2="all_test")
    print("\n\n")
    get_log(dd_train_data, dd_test_data, tag1="all_train", tag2="all_test")
    print("\n\n")
    get_log(dd_train0, dd_test0, tag1="train0", tag2="test0")
    print("\n\n")
    get_log(dd_train1, dd_test1, tag1="train1", tag2="test1")
    print("\n\n")
    get_log(dd_train2, dd_test2, tag1="train2", tag2="test2")
    print("\n\n")
    get_log(dd_train3, dd_test3, tag1="train3", tag2="test3")
    print("\n\n")
    get_log(dd_train4, dd_test4, tag1="train4", tag2="test4")



    ################################################
    print("\n\n\n\nBB4")
    ################################################

    from pronto import Ontology
    onto = Ontology("../BB4/OntoBiotope_BioNLP-OST-2019.obo")

    ddd_dataTrain = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_train"])
    ddd_dataDev = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_dev"])
    ddd_dataTrainDev = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_train", "../BB4/BioNLP-OST-2019_BB-norm_dev"])
    ddd_dataTest = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_test"])
    ddd_dataAll = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_train", "../BB4/BioNLP-OST-2019_BB-norm_dev", "../BB4/BioNLP-OST-2019_BB-norm_test"])

    dd_habTrain = extract_data(ddd_dataTrain, l_type=["Habitat"])
    dd_habDev = extract_data(ddd_dataDev, l_type=["Habitat"])
    dd_habTrainDev = extract_data(ddd_dataTrainDev, l_type=["Habitat"])
    dd_habTest = extract_data(ddd_dataTest, l_type=["Habitat"])
    dd_habAll = extract_data(ddd_dataAll, l_type=["Habitat"])
    print(dd_habTrain)
    print(dd_habTest)

    get_log(dd_habTrain, dd_habTest, tag1="train_hab", tag2="test_hab")
    print("\n\n")
    get_log(dd_habTrain, dd_habDev, tag1="train_hab", tag2="dev_hab")
    print("\n\n")
    get_log(dd_habTrainDev, dd_habTest, tag1="train+dev_hab", tag2="test_hab")



    ################################################
    print("\n\n\n\nNCBI")
    ################################################

    dd_medic = loader_medic("../NCBI/FixedVersion/CTD_diseases_DNorm_v2012_07_6_fixed.tsv")
    nbConcepts = len(dd_medic.keys())
    print("\nNumber of concepts in MEDIC:", nbConcepts)

    # Fields in MEDIC file:
    # DiseaseName	DiseaseID	AltDiseaseIDs	Definition	ParentIDs	TreeNumbers	ParentTreeNumbers	Synonyms
    # In trainset: 2792129	158	178	recurrent meningitis	SpecificDisease	D008581+D012008

    # For initial MEDIC/datasets, some alternative CUIs are used in the corpus...
    s_medic = set(dd_medic.keys())
    s_alt = set()
    s_inter = set()
    s_both = set(dd_medic.keys())
    for cui in dd_medic.keys():
        if "alt_cui" in dd_medic[cui].keys():
            for altCui in dd_medic[cui]["alt_cui"]:
                s_both.add(altCui)
                if altCui not in s_medic:
                    s_alt.add(altCui)
                else:
                    s_inter.add(cui+"/"+altCui)

    print("Number of alternative CUIs in MEDIC:", len(s_alt))
    print("Number of CUIs in MEDIC (main+alternative)", len(s_both))
    print("Sibling concepts in MEDIC used in the NCBI dataset:", s_inter)


    print("\n\n")


    ddd_dataFull = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItrainset_corpus_fixed.txt", "../NCBI/FixedVersion/NCBIdevelopset_corpus.txt", "../NCBI/FixedVersion/NCBItestset_corpus_fixed.txt"])
    dd_Full = extract_data(ddd_dataFull, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("Number of examples/mentions in Full dataset: ", len(dd_Full.keys()))

    ddd_dataTrain = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItrainset_corpus_fixed.txt"])
    dd_Train = extract_data(ddd_dataTrain, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print(id, len(dd_Train.keys()))

    ddd_dataDev = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBIdevelopset_corpus.txt"])
    dd_Dev = extract_data(ddd_dataDev, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print(len(dd_Dev.keys()))

    ddd_dataTrainDev = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItrainset_corpus_fixed.txt", "../NCBI/FixedVersion/NCBIdevelopset_corpus.txt"])
    dd_TrainDev = extract_data(ddd_dataTrainDev, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print(len(dd_TrainDev.keys()))

    ddd_dataTest = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItestset_corpus_fixed.txt"])
    dd_Test = extract_data(ddd_dataTest, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print(len(dd_Test.keys()))


    get_log(dd_Train, dd_Dev, tag1="train", tag2="dev")
    print("\n\n")
    get_log(dd_TrainDev, dd_Test, tag1="train+dev", tag2="test")
    print("\n\n")
    get_log(dd_Train, dd_Test, tag1="train", tag2="test")
    print("\n\n")
    get_log(dd_Full, dd_Test, tag1="Full", tag2="test")


    sys.exit(0)

