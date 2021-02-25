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
# Intrinsic analysers:
###################################################
###
# WARNING: Add alternative CUIs here too, I think...
###
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
        if len(dd_data[id]["cui"]) > 1:
            nb+=1

    return (len(dd_data.keys()) - nb), nb

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



def extract_data_without_file(ddd_data):
    dd_data = dict()
    for file in ddd_data.keys():
        for id in ddd_data[file].keys():
            dd_data[id] = ddd_data[file][id]
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

    ################################################
    print("\n\n\n\nCustom CADEC\n")
    ################################################

    from loaders import loader_snomed_ct_au, get_cui_list, loader_all_random_cadec_folds

    dd_sct = loader_snomed_ct_au(
        "../CADEC/SNOMED_CT_AU_20140531/SnomedCT_Release_AU1000036_20140531/RF2 Release\Snapshot/Terminology/sct2_Description_Snapshot-en-AU_AU1000036_20140531.txt")
    l_sctFromCadec = get_cui_list("../CADEC/custom_CUI_list.txt")

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

    from loaders import loader_all_random_cadec_folds, extract_one_cadec_fold

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

    from loaders import loader_all_initial_cadec_folds, loader_amt

    ddd_data = loader_all_initial_cadec_folds("../CADEC/0_Original_CADEC/AMT-SCT/")
    dd_cadec = extract_data_without_file(ddd_data)

    print("test ddd_data['ARTHROTEC.32']:", ddd_data["ARTHROTEC.32"])

    s_cuis = set()
    l_altCuis = list()
    compt = 0
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
                compt += 1
            if ddd_data[file][id]["cui"] == ["CONCEPT_LESS"]:
                cuiLess += 1

    print("len(s_cuis):", len(s_cuis))
    print("len(l_altCuis):", len(l_altCuis), l_altCuis)
    print("multi", compt, "cui LSS:", cuiLess)

    # Load AMT:
    dd_amt = loader_amt("../CADEC/AMT_v2.56/Uuid_sct_concepts_au.gov.nehta.amt.standalone_2.56.txt")

    s_amt = set()
    for cui in dd_amt.keys():
        s_amt.add(cui)

    print(len(s_amt), len(dd_amt.keys()))

    print("\n\n")

    cmpInSCT = 0
    cmpInAMT = 0
    conceptLess = 0
    Nbmentions = 0
    wtf = 0
    s1 = set()
    s2 = set()
    s3 = set()
    for file in ddd_data.keys():
        for id in ddd_data[file].keys():
            Nbmentions += 1
            for cui in ddd_data[file][id]["cui"]:
                if cui in dd_sct.keys() and cui != "CONCEPT_LESS":
                    cmpInSCT += 1
                    s1.add(cui)
                elif cui in dd_amt.keys() and cui != "CONCEPT_LESS":
                    cmpInAMT += 1
                    s2.add(cui)
                elif cui == "CONCEPT_LESS":
                    conceptLess += 1
                else:
                    wtf += 1
                    s3.add(cui)
                    print(file, ddd_data[file][id])

    print(cmpInSCT, cmpInAMT, conceptLess, Nbmentions, wtf)
    print(len(s1), len(s2), len(s3))

    get_log(dd_cadec, dd_cadec, tag1="Full", tag2="Full")



