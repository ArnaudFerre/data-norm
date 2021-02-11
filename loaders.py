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
# Loaders:
###################################################

def loader_all_cadec_folds(repPath):
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

            print("Load ", foldFileName)
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
# Intrinsic analysers:
###################################################

def get_all_used_cui_in_fold(dd_data):
    s_cui = set()
    for id in dd_data.keys():
        for cui in dd_data[id]["cui"]:
            s_cui.add(cui)
    return s_cui

def get_all_used_cui(ddd_data):
    s_cui = set()
    for fold in ddd_data.keys():
        dd_fold = ddd_data[fold]
        s_cui = s_cui.union(get_all_used_cui_in_fold(dd_fold))
    return s_cui

#########################

def get_freq_examples(dd_data):
    dd_freq_example = dict()

    for id in dd_data.keys():
        surfaceForm = dd_data[id]["mention"]
        dd_freq_example[surfaceForm] = dict()

    for id in dd_data.keys():
        surfaceForm = dd_data[id]["mention"]
        l_cuis = dd_data[id]["cui"]
        for cui in l_cuis:
            dd_freq_example[surfaceForm][cui] = 0

    for id in dd_data.keys():
        l_cuis = dd_data[id]["cui"]
        for cui in l_cuis:
            dd_freq_example[ dd_data[id]["mention"] ][ cui ] += 1

    return dd_freq_example

def get_freq_examples_in_whole(ddd_data):
    dd_freq_example = dict()

    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            surfaceForm = ddd_data[foldName][id]["mention"]
            dd_freq_example[surfaceForm] = dict()

    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            surfaceForm = ddd_data[foldName][id]["mention"]
            cui = ddd_data[foldName][id]["cui"]
            dd_freq_example[surfaceForm][cui] = 0

    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            surfaceForm = ddd_data[foldName][id]["mention"]
            cui = ddd_data[foldName][id]["cui"]
            dd_freq_example[surfaceForm][cui] += 1

    return dd_freq_example


def get_unique_example(dd_freq_example):
    nb = 0
    for surfaceForm in dd_freq_example.keys():
        for cui in dd_freq_example[surfaceForm].keys():
            freq = dd_freq_example[surfaceForm][cui]
            if freq == 1:
                nb+=1
    return nb


def get_distinct_examples(dd_data):
    l_distinctExamples = list()
    dd_freq_example = get_freq_examples(dd_data)
    for surfaceForm in dd_freq_example.keys():
        for cui in dd_freq_example[surfaceForm].keys():
            l_distinctExamples.append([surfaceForm, cui])
    return l_distinctExamples

def get_distinct_examples_in_whole(ddd_data):
    l_distinctExamples = list()
    dd_freq_example = get_freq_examples_in_whole(ddd_data)
    for surfaceForm in dd_freq_example.keys():
        for cui in dd_freq_example[surfaceForm].keys():
            l_distinctExamples.append([surfaceForm, cui])
    return l_distinctExamples


#########################

def get_all_surface_forms_in_fold(dd_data):
    s_surfaceForms = set()
    for id in dd_data.keys():
        s_surfaceForms.add(dd_data[id]["mention"])
    return s_surfaceForms

def get_all_surface_forms(ddd_data):
    s_surfaceForms = set()

    for fold in ddd_data.keys():
        for id in ddd_data[fold].keys():
            s_surfaceForms.add(ddd_data[fold][id]["mention"])

    return s_surfaceForms

#########################

def get_freq_surface_forms(dd_data):
    d_freqSurfaceForms = dict()
    for id in dd_data.keys():
        surfaceForm = dd_data[id]["mention"]
        d_freqSurfaceForms[surfaceForm] = 0
    for id in dd_data.keys():
        surfaceForm = dd_data[id]["mention"]
        d_freqSurfaceForms[surfaceForm] += 1
    return d_freqSurfaceForms

def get_freq_surface_forms_in_whole(ddd_data):
    d_freqSurfaceForms = dict()
    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            surfaceForm = ddd_data[foldName][id]["mention"]
            d_freqSurfaceForms[surfaceForm] = 0
    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            surfaceForm = ddd_data[foldName][id]["mention"]
            d_freqSurfaceForms[surfaceForm] += 1
    return d_freqSurfaceForms

def get_unique_surface_forms(d_freqSurfaceForms):
    l_unique_surface_forms = list()
    for surfaceForm in d_freqSurfaceForms.keys():
        if d_freqSurfaceForms[surfaceForm] == 1:
            l_unique_surface_forms.append(surfaceForm)
    return l_unique_surface_forms

#########################

def get_average_number_mentions_per_concept(dd_data):
    nb = 0

    dl_mentionsPerConcept = dict()

    for id in dd_data.keys():
        l_cuis = dd_data[id]["cui"]
        for concept in l_cuis:
            dl_mentionsPerConcept[concept] = list()

    for id in dd_data.keys():
        mention = dd_data[id]["mention"]
        l_cuis = dd_data[id]["cui"]
        for concept in l_cuis:
            dl_mentionsPerConcept[concept].append(mention)

    for concept in dl_mentionsPerConcept.keys():
        nb += len(dl_mentionsPerConcept[concept])

    nb = (1.0*nb) / len(dl_mentionsPerConcept.keys())

    return nb


def get_average_number_mentions_per_concept_in_whole(ddd_data):
    nb = 0

    dl_mentionsPerConcept = dict()

    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            concept = ddd_data[foldName][id]["cui"]
            dl_mentionsPerConcept[concept] = list()

    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            mention = ddd_data[foldName][id]["mention"]
            concept = ddd_data[foldName][id]["cui"]
            dl_mentionsPerConcept[concept].append(mention)

    for concept in dl_mentionsPerConcept.keys():
        nb += len(dl_mentionsPerConcept[concept])

    nb = (1.0*nb) / len(dl_mentionsPerConcept.keys())

    return nb

#########################

def get_std_number_mentions_per_concept(dd_data):

    dl_mentionsPerConcept = dict()

    for id in dd_data.keys():
        l_cuis = dd_data[id]["cui"]
        for concept in l_cuis:
            dl_mentionsPerConcept[concept] = list()

    for id in dd_data.keys():
        mention = dd_data[id]["mention"]
        l_cuis = dd_data[id]["cui"]
        for concept in l_cuis:
            dl_mentionsPerConcept[concept].append(mention)

    l_values = list()
    for concept in dl_mentionsPerConcept.keys():
        l_values.append(len(dl_mentionsPerConcept[concept]))

    std_value = std(l_values)
    median_value = median(l_values)

    return std_value, median_value


def get_std_number_mentions_per_concept_in_whole(ddd_data):

    dl_mentionsPerConcept = dict()

    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            concept = ddd_data[foldName][id]["cui"]
            dl_mentionsPerConcept[concept] = list()

    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            mention = ddd_data[foldName][id]["mention"]
            concept = ddd_data[foldName][id]["cui"]
            dl_mentionsPerConcept[concept].append(mention)

    l_values = list()
    for concept in dl_mentionsPerConcept.keys():
        l_values.append(len(dl_mentionsPerConcept[concept]))

    std_value = std(l_values)
    median_value = median(l_values)

    return std_value, median_value

#########################

def get_number_of_surface_forms_with_different_labels(dd_data):
    nbSurfaceForms = 0
    nbMentions = 0

    ds_surfaceFormsWithLabels = dict()
    for id in dd_data.keys():
        surfaceForm = dd_data[id]["mention"]
        ds_surfaceFormsWithLabels[surfaceForm] = set()

    for id in dd_data.keys():
        surfaceForm = dd_data[id]["mention"]
        l_cuis = dd_data[id]["cui"]
        for cui in l_cuis:
            ds_surfaceFormsWithLabels[surfaceForm].add(cui)

    for surfaceForm in ds_surfaceFormsWithLabels.keys():
        if len(ds_surfaceFormsWithLabels[surfaceForm]) > 1:
            nbSurfaceForms+=1

    for id in dd_data.keys():
        surfaceForm = dd_data[id]["mention"]
        if len(ds_surfaceFormsWithLabels[surfaceForm]) > 1:
            nbMentions+=1


    return nbSurfaceForms, nbMentions



def get_number_of_surface_forms_with_different_labels_in_whole(ddd_data):
    nb=0
    nbMentions = 0

    ds_surfaceFormsWithLabels = dict()
    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            surfaceForm = ddd_data[foldName][id]["mention"]
            ds_surfaceFormsWithLabels[surfaceForm] = set()

    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            surfaceForm = ddd_data[foldName][id]["mention"]
            cui = ddd_data[foldName][id]["cui"]
            ds_surfaceFormsWithLabels[surfaceForm].add(cui)

    for surfaceForm in ds_surfaceFormsWithLabels.keys():
        if len(ds_surfaceFormsWithLabels[surfaceForm]) > 1:
            nb+=1

    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            surfaceForm = ddd_data[foldName][id]["mention"]
            if len(ds_surfaceFormsWithLabels[surfaceForm]) > 1:
                nbMentions+=1

    return nb, nbMentions


#########################

def get_nb_concepts_with_only_one_mention(dd_data, rate=1):
    nb=0

    dl_mentionsPerConcept = dict()

    for id in dd_data.keys():
        l_cuis = dd_data[id]["cui"]
        for cui in l_cuis:
            dl_mentionsPerConcept[cui] = list()

    for id in dd_data.keys():
        mention = dd_data[id]["mention"]
        l_cuis = dd_data[id]["cui"]
        for cui in l_cuis:
            dl_mentionsPerConcept[cui].append(mention)

    for cui in dl_mentionsPerConcept.keys():
        if len(dl_mentionsPerConcept[cui]) == rate:
            nb+=1

    return nb


def get_nb_concepts_with_only_one_mention_in_whole(ddd_data, rate=1):
    nb=0

    dl_mentionsPerConcept = dict()

    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            cui = ddd_data[foldName][id]["cui"]
            dl_mentionsPerConcept[cui] = list()

    for foldName in ddd_data.keys():
        for id in ddd_data[foldName].keys():
            cui = ddd_data[foldName][id]["cui"]
            mention = ddd_data[foldName][id]["mention"]
            dl_mentionsPerConcept[cui].append(mention)


    for cui in dl_mentionsPerConcept.keys():
        if len(dl_mentionsPerConcept[cui]) == rate:
            nb+=1

    return nb


#########################

def get_nb_mentions_with_one_concept(dd_data):
    nb=0

    for id in dd_data.keys():
        if "multi" in dd_data[id].keys():
            if dd_data[id]["multi"]== True:
                nb+=1

    return (len(dd_data.keys()) - nb)

###################################################
# Inter-analysers:
###################################################
# List of all id of examples in test which can be seen in train:
def get_same_examples_from_test_in_train(dd_dataTrain, dd_dataTest):
    l_sameExamples = list()
    d_examplesFromTestInTrain = dict()

    for idTest in dd_dataTest.keys():
        mention = dd_dataTest[idTest]["mention"]
        cui = dd_dataTest[idTest]["cui"]
        for idTrain in dd_dataTrain.keys():
            if dd_dataTrain[idTrain]["mention"] == mention and dd_dataTrain[idTrain]["cui"] == cui:
                d_examplesFromTestInTrain[idTest] = dict()
                d_examplesFromTestInTrain[idTest]["mention"] = mention
                d_examplesFromTestInTrain[idTest]["idExamplesTrain"] = list()


    for idTest in dd_dataTest.keys():
        mention = dd_dataTest[idTest]["mention"]
        cui = dd_dataTest[idTest]["cui"]
        for idTrain in dd_dataTrain.keys():
            if dd_dataTrain[idTrain]["mention"] == mention and dd_dataTrain[idTrain]["cui"] == cui:
                d_examplesFromTestInTrain[idTest]["mention"] = mention
                d_examplesFromTestInTrain[idTest]["idExamplesTrain"].append(idTrain)

    return d_examplesFromTestInTrain


def get_mentions_from_test_in_train(dd_dataTrain, dd_dataTest):
    d_mentionsFromTestInTrain = dict()

    for idTest in dd_dataTest.keys():
        mention = dd_dataTest[idTest]["mention"]
        for idTrain in dd_dataTrain.keys():
            if dd_dataTrain[idTrain]["mention"] == mention:
                d_mentionsFromTestInTrain[idTest] = dict()
                d_mentionsFromTestInTrain[idTest]["mention"] = mention
                d_mentionsFromTestInTrain[idTest]["idsTrain"] = list()

    for idTest in d_mentionsFromTestInTrain.keys():
        for idTrain in dd_dataTrain.keys():
            if dd_dataTrain[idTrain]["mention"] == d_mentionsFromTestInTrain[idTest]["mention"]:
                d_mentionsFromTestInTrain[idTest]["idsTrain"].append(idTrain)

    return d_mentionsFromTestInTrain



def get_surface_forms_from_test_in_train(dd_dataTrain, dd_dataTest):
    d_surfaceFormsFromTestInTrain = dict()

    for idTest in dd_dataTest.keys():
        mention = dd_dataTest[idTest]["mention"]
        for idTrain in dd_dataTrain.keys():
            if dd_dataTrain[idTrain]["mention"] == mention:
                d_surfaceFormsFromTestInTrain[mention] = dict()
                d_surfaceFormsFromTestInTrain[mention]["idTest"] = idTest
                d_surfaceFormsFromTestInTrain[mention]["idsTrain"] = list()

    for surfaceForm in d_surfaceFormsFromTestInTrain.keys():
        for idTrain in dd_dataTrain.keys():
            if dd_dataTrain[idTrain]["mention"] == surfaceForm:
                d_surfaceFormsFromTestInTrain[surfaceForm]["idsTrain"].append(idTrain)

    return d_surfaceFormsFromTestInTrain


###################################################
# Tools:
###################################################

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



###################################################
# Printers:
###################################################
def get_log(dd_train, dd_test, tag1="train", tag2="test"):

    print("\nBeginning of analyis...")
    print("\nIntra-analysis...")

    # Concepts used
    s_cui_train = get_all_used_cui_in_fold(dd_train)
    print("\nAll CUIs used in the "+tag1+" dataset: ", len(s_cui_train))
    try:
        s_cui_test = get_all_used_cui_in_fold(dd_test)
        print("All CUIs used in the "+tag2+" dataset: ", len(s_cui_test))
    except:
        print("Non Available data for "+tag2+" dataset.")

    # Number of examples: (WARNING: different from all the examples in the original corpus (because overlapping between folds)
    print("\nNumber of examples in "+tag1+":", len(dd_train))
    print("Number of examples in "+tag2+":", len(dd_test))

    print("\nUnique examples:")
    print("Number of unique examples in "+tag1+":", get_unique_example(get_freq_examples(dd_train)))
    try:
        print("Number of unique examples in "+tag2+":", get_unique_example(get_freq_examples(dd_test)))
    except:
        print("Non Available data for "+tag2+" dataset.")

    print("\nDistinct examples:")
    print("Number of distinct examples in "+tag1+":", len(get_distinct_examples(dd_train)))
    try:
        print("Number of distinct examples in "+tag2+":", len(get_distinct_examples(dd_test)))
    except:
        print("Non Available data for "+tag2+" dataset.")

    print("\nDistinct surface forms:")
    s_surfaceFormsTrain = get_all_surface_forms_in_fold(dd_train)
    print("All surface forms in the "+tag1+": ", len(s_surfaceFormsTrain))
    s_surfaceFormsTest = get_all_surface_forms_in_fold(dd_test)
    print("All surface forms in the "+tag2+": ", len(s_surfaceFormsTest))

    print("\nUnique surface forms:")
    print("Nb of unique surface forms in the "+tag1+": ", len(get_unique_surface_forms(get_freq_surface_forms(dd_train))))
    print("Nb of unique surface forms in the "+tag2+": ", len(get_unique_surface_forms(get_freq_surface_forms(dd_test))))

    print("\nAverage number of mentions per concepts:")
    print("Average number of mentions per concepts in the "+tag1+": ", get_average_number_mentions_per_concept(dd_train))
    try:
        print("Average number of mentions per concepts in the "+tag2+": ", get_average_number_mentions_per_concept(dd_test))
    except:
        print("Non Available data for "+tag2+" dataset.")

    print("\nStandard deviation and median of mentions per concepts:")
    print("Standard deviation and median of mentions per concepts in the "+tag1+": ", get_std_number_mentions_per_concept(dd_train))
    try:
        print("Standard deviation and median of mentions per concepts in the "+tag2+": ", get_std_number_mentions_per_concept(dd_test))
    except:
        print("Non Available data for "+tag2+" dataset.")

    print("\nSingle shot situation?")
    print("Number concepts with only one mention in the "+tag1+": ", get_nb_concepts_with_only_one_mention(dd_train))
    try:
        print("Number concepts with only one mention in the "+tag2+": ", get_nb_concepts_with_only_one_mention(dd_test))
    except:
        print("Non Available data for "+tag2+" dataset.")

    #################
    print("\nNumber of mentions normalized by a unique concept (no multi-norm):")
    print("Number of mentions normalized by a unique concept in the "+tag1+": ", get_nb_mentions_with_one_concept(dd_train))
    try:
        print("Number of mentions normalized by a unique concept in the "+tag2+": ", get_nb_mentions_with_one_concept(dd_test))
    except:
        print("Non Available data for "+tag2+" dataset.")

    print("\nHow many surface forms have more than one annotating concept (=ambiguity mention) ?")
    print("Number of surface forms with different possible labels in the "+tag1+": ", get_number_of_surface_forms_with_different_labels(dd_train))
    try:
        print("Number of surface forms with different possible labels in the "+tag2+": ", get_number_of_surface_forms_with_different_labels(dd_test))
    except:
        print("Non Available data for "+tag2+" dataset.")


    print("\nInter-analysis...")

    print("\nIntersection "+tag1+"/"+tag2+":")
    try:
        print(len(get_same_examples_from_test_in_train(dd_train, dd_test).keys()))
    except:
        print("Non Available because non-available data for "+tag2+" dataset.")

    print("\nNb mentions in "+tag2+" also seen in "+tag1+":", len(get_mentions_from_test_in_train(dd_train, dd_test).keys()))

    print("\nNumber of surface forms from "+tag2+" also present in "+tag1+":", len(get_surface_forms_from_test_in_train(dd_train, dd_test).keys()))

    print("\nNumber of concepts mentioned in "+tag2+" which are mentioned in "+tag1+":")
    try:
        nb=0
        for cui in s_cui_test:
            if cui in s_cui_train:
                nb+=1
        print(nb)
    except:
        print("Non Available because non-available data for "+tag2+" dataset.")

    print("\nEnd of analysis.")





#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':

    """
    ################################################
    print("\n\n\n\nCADEC\n")
    ################################################

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

    """
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
    print(len(dd_Train.keys()))

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
    get_log(dd_Full, dd_Full, tag1="Full", tag2="Full")








    """
    # To find possible mentions in corpus with a CUI not in the reference (0 in the train):
    i=0
    for id in dd_habTrain.keys():
        l_cuis = dd_habTrain[id]["cui"]

        for cui in l_cuis:
            cuiMesh = "MESH:"+str(cui)
            if cui not in s_medic:
                if cuiMesh not in s_medic:
                    print(dd_habTrain[id]["cui"], dd_habTrain[id])
                    i += 1

    print(i)
    """


    print("\n\n")







    sys.exit(0)

