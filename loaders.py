# Author: Arnaud Ferré
# RALI, Montreal University
#
# Description :



#######################################################################################################
# Imports:
#######################################################################################################


from os import listdir
from os.path import isfile, join, splitext
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
                    ddd_data[foldFileNameWithoutExt][exampleId]["cui"] = cui.rstrip()

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
                                            ddd_data[fileNameWithoutExt][id]["cui"] = cui
                                        elif ddd_data[fileNameWithoutExt][id]["type"] == "Microorganism":
                                            cui = l_info[2].split(':')[1].rstrip()
                                            ddd_data[fileNameWithoutExt][id]["cui"] = cui


    return ddd_data



def extract_data(ddd_data, l_type=[]):
    dd_data = dict()

    for fileName in ddd_data.keys():
        for id in ddd_data[fileName].keys():
            if ddd_data[fileName][id]["type"] in l_type:
                dd_data[id] = ddd_data[fileName][id]

    return dd_data




###################################################
# Intrinsic analysers:
###################################################

def get_all_used_cui_in_fold(dd_data):
    s_cui = set()
    for id in dd_data.keys():
        s_cui.add(dd_data[id]["cui"])
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
        cui = dd_data[id]["cui"]
        dd_freq_example[surfaceForm][cui] = 0

    for id in dd_data.keys():
        dd_freq_example[ dd_data[id]["mention"] ][ dd_data[id]["cui"] ] += 1

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
        concept = dd_data[id]["cui"]
        dl_mentionsPerConcept[concept] = list()

    for id in dd_data.keys():
        mention = dd_data[id]["mention"]
        concept = dd_data[id]["cui"]
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
        concept = dd_data[id]["cui"]
        dl_mentionsPerConcept[concept] = list()

    for id in dd_data.keys():
        mention = dd_data[id]["mention"]
        concept = dd_data[id]["cui"]
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
        cui = dd_data[id]["cui"]
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
        cui = dd_data[id]["cui"]
        dl_mentionsPerConcept[cui] = list()

    for id in dd_data.keys():
        cui = dd_data[id]["cui"]
        mention = dd_data[id]["mention"]
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

    #######################################################################################################
    # BB4

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

    from pronto import Ontology
    onto = Ontology("../BB4/OntoBiotope_BioNLP-OST-2019.obo")





    get_log(dd_habTrain, dd_habTest, tag1="train_hab", tag2="test_hab")
    print("\n\n")
    get_log(dd_habTrain, dd_habDev, tag1="train_hab", tag2="dev_hab")
    print("\n\n")
    get_log(dd_habTrainDev, dd_habTest, tag1="train+dev_hab", tag2="test_hab")
    sys.exit(0)





    # Concepts used
    s_cui_train = get_all_used_cui_in_fold(dd_habTrain)
    print("All CUIs used in the whole train dataset: ", len(s_cui_train))
    s_cui_dev = get_all_used_cui_in_fold(dd_habDev)
    print("All CUIs used in the whole train dataset: ", len(s_cui_dev))
    s_cui_TrainDev = get_all_used_cui_in_fold(dd_habTrainDev)
    print("All CUIs used in the whole train dataset: ", len(s_cui_TrainDev))


    print("\n\n")


    # Number of examples: (WARNING: different from all the examples in the original corpus (because overlapping between folds)
    print("Number of examples in train:", len(dd_habTrain))
    print("Number of examples in dev:", len(dd_habDev))
    print("Number of examples in train+dev:", len(dd_habTrainDev))
    print("Number of examples in test:", len(dd_habTest))
    print("Number of examples in all:", len(dd_habAll))


    print("\n\n")


    print("Unique examples:\n")
    print("Number of unique examples in train:", get_unique_example(get_freq_examples(dd_habTrain)))
    print("Number of unique examples in dev:", get_unique_example(get_freq_examples(dd_habDev)))
    print("Number of unique examples in train+dev:", get_unique_example(get_freq_examples(dd_habTrainDev)))


    print("\n\n")


    print("Distinct examples:\n")
    print("Number of distinct examples in train:", len(get_distinct_examples(dd_habTrain)))
    print("Number of distinct examples in dev:", len(get_distinct_examples(dd_habDev)))
    print("Number of distinct examples in train+dev:", len(get_distinct_examples(dd_habTrainDev)))


    print("\n\n")


    print("Distinct surface forms:\n")
    s_surfaceFormsTrain = get_all_surface_forms_in_fold(dd_habTrain)
    print("All surface forms in the train : ", len(s_surfaceFormsTrain))
    s_surfaceFormsDev = get_all_surface_forms_in_fold(dd_habDev)
    print("All surface forms in the dev : ", len(s_surfaceFormsDev))
    s_surfaceFormsTest = get_all_surface_forms_in_fold(dd_habTest)
    print("All surface forms in the test : ", len(s_surfaceFormsTest))
    s_surfaceFormsTrainDev = get_all_surface_forms_in_fold(dd_habTrainDev)
    print("All surface forms in the train+dev : ", len(s_surfaceFormsTrainDev))
    s_surfaceFormsAll = get_all_surface_forms_in_fold(dd_habAll)
    print("All surface forms in all : ", len(s_surfaceFormsAll))


    print("\n\n")

    print("Unique surface forms:\n")
    print("Nb of unique surface forms in the train: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_habTrain))))
    print("Nb of unique surface forms in the dev: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_habDev))))
    print("Nb of unique surface forms in the test: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_habTest))))
    print("Nb of unique surface forms in the train+dev: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_habTrainDev))))
    print("Nb of unique surface forms in the all: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_habAll))))


    print("\n\n")

    print("Average number of mentions per concepts:\n")
    print("Average number of mentions per concepts in the train: ", get_average_number_mentions_per_concept(dd_habTrain))
    print("Average number of mentions per concepts in the dev: ", get_average_number_mentions_per_concept(dd_habDev))
    print("Average number of mentions per concepts in the train+dev: ", get_average_number_mentions_per_concept(dd_habTrainDev))


    print("Standard deviation and median of mentions per concepts:\n")
    print("Standard deviation and median of mentions per concepts in the train: ", get_std_number_mentions_per_concept(dd_habTrain))
    print("Standard deviation and median of mentions per concepts in the dev: ", get_std_number_mentions_per_concept(dd_habDev))
    print("Standard deviation and median of mentions per concepts in the train+dev: ", get_std_number_mentions_per_concept(dd_habTrainDev))


    print("\n\n")

    print("Single shot situation?\n")
    print("Number concepts with only one mention in the train: ", get_nb_concepts_with_only_one_mention(dd_habTrain))
    print("Number concepts with only one mention in the dev: ", get_nb_concepts_with_only_one_mention(dd_habDev))
    print("Number concepts with only one mention in the train+dev: ", get_nb_concepts_with_only_one_mention(dd_habTrainDev))


    print("\n\n")

    print("How many surface forms have more than one annotating concept (=ambiguity mention) ?\n")
    print("Number of surface forms with different possible labels in the train: ", get_number_of_surface_forms_with_different_labels(dd_habTrain))
    print("Number of surface forms with different possible labels in the dev: ", get_number_of_surface_forms_with_different_labels(dd_habDev))
    print("Number of surface forms with different possible labels in the train+dev: ", get_number_of_surface_forms_with_different_labels(dd_habTrainDev))


    print("\n\n")
    ########################
    print("Extrinsic analysis:")
    ########################

    print("Intersection train/dev:", len(get_same_examples_from_test_in_train(dd_habTrain, dd_habDev).keys()))

    print("Nb mentions in test also seen in train:", len(get_mentions_from_test_in_train(dd_habTrain, dd_habDev).keys()))

    print("Number of surface forms from dev also present in train:", len(get_surface_forms_from_test_in_train(dd_habTrain, dd_habDev).keys()))

    nb=0
    for cui in s_cui_dev:
        if cui in s_cui_train:
            nb+=1
    print("Number of concepts mentioned in dev which are mentioned in train", nb)

    sys.exit(0)







































    sys.exit(0)
    #######################################################################################################
    # CADEC


    ########################
    # Open all folds
    ########################

    #Option 1: folds data     (#Option 2: train/dev/test data# dd_data)
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


    print("\n\n")


    ########################
    # Intrinsic analysis:
    ########################

    # Used concepts: (WARNING: don't look in the reference, juste the different CUIs in the dataset)
    s_cui = get_all_used_cui(ddd_data)
    print("All CUIs used in the whole dataset: ", len(s_cui), s_cui)

    s_cui_train = get_all_used_cui_in_fold(dd_train_data)
    print("All CUIs used in the whole train dataset: ", len(s_cui_train))
    s_cui_test = get_all_used_cui_in_fold(dd_test_data)
    print("All CUIs used in the whole test dataset: ", len(s_cui_test))

    s_cui_train0 = get_all_used_cui_in_fold(dd_train0)
    print("All CUIs used in the train0 fold: ", len(s_cui_train0))
    s_cui_train1 = get_all_used_cui_in_fold(dd_train1)
    print("All CUIs used in the train1 fold: ", len(s_cui_train1))
    s_cui_train2 = get_all_used_cui_in_fold(dd_train2)
    print("All CUIs used in the train2 fold: ", len(s_cui_train2))
    s_cui_train3 = get_all_used_cui_in_fold(dd_train3)
    print("All CUIs used in the train3 fold: ", len(s_cui_train3))
    s_cui_train4 = get_all_used_cui_in_fold(dd_train4)
    print("All CUIs used in the train4 fold: ", len(s_cui_train4))

    s_cui_test0 = get_all_used_cui_in_fold(dd_test0)
    print("All CUIs used in the test0 fold: ", len(s_cui_test0))
    s_cui_test1 = get_all_used_cui_in_fold(dd_test1)
    print("All CUIs used in the test1 fold: ", len(s_cui_test1))
    s_cui_test2 = get_all_used_cui_in_fold(dd_test2)
    print("All CUIs used in the test2 fold: ", len(s_cui_test2))
    s_cui_test3 = get_all_used_cui_in_fold(dd_test3)
    print("All CUIs used in the test3 fold: ", len(s_cui_test3))
    s_cui_test4 = get_all_used_cui_in_fold(dd_test4)
    print("All CUIs used in the test4 fold: ", len(s_cui_test4))


    print("\n\n")


    # Number of examples: (WARNING: different from all the examples in the original corpus (because overlapping between folds)
    NbLabelsInWhole = 0
    for fold in ddd_data.keys():
        NbLabelsInWhole += len(ddd_data[fold].keys())
    print("Number of examples in all folds:", NbLabelsInWhole)

    print("Number of examples in all train folds:", len(dd_train_data))
    print("Number of examples in all train folds:", len(dd_test_data))
    print("\n")
    print("Number of examples in train+test0 fold:", len(dd_train_test_0))
    print("Number of examples in train0 fold:", len(dd_train0))
    print("Number of examples in test0 fold:", len(dd_test0))
    print("Number of examples in train+test1 fold:", len(dd_train_test_1))
    print("Number of examples in train1 fold:", len(dd_train1))
    print("Number of examples in test1 fold:", len(dd_test1))
    print("Number of examples in train+test2 fold:", len(dd_train_test_2))
    print("Number of examples in train2 fold:", len(dd_train2))
    print("Number of examples in test2 fold:", len(dd_test2))
    print("Number of examples in train+test3 fold:", len(dd_train_test_3))
    print("Number of examples in train3 fold:", len(dd_train3))
    print("Number of examples in test3 fold:", len(dd_test3))
    print("Number of examples in train+test4 fold:", len(dd_train_test_4))
    print("Number of examples in train4 fold:", len(dd_train4))
    print("Number of examples in test4 fold:", len(dd_test4))



    print("\n\n")


    print("Unique examples:\n")

    print("Number of unique examples in all folds: ", get_unique_example(get_freq_examples_in_whole(ddd_data)))

    print("\nNumber of unique examples in all train folds:", get_unique_example(get_freq_examples(dd_train_data)))
    print("Number of unique examples in all test folds:", get_unique_example(get_freq_examples(dd_test_data)))

    print("\nNumber of unique examples in train+test0: ", get_unique_example(get_freq_examples(dd_train_test_0)))
    print("Number of unique examples in train0: ", get_unique_example(get_freq_examples(dd_train0)))
    print("Number of unique examples in test0: ", get_unique_example(get_freq_examples(dd_test0)))
    print("Number of unique examples in train+test1: ", get_unique_example(get_freq_examples(dd_train_test_1)))
    print("Number of unique examples in train1: ", get_unique_example(get_freq_examples(dd_train1)))
    print("Number of unique examples in test1: ", get_unique_example(get_freq_examples(dd_test1)))
    print("Number of unique examples in train+test2: ", get_unique_example(get_freq_examples(dd_train_test_2)))
    print("Number of unique examples in train2: ", get_unique_example(get_freq_examples(dd_train2)))
    print("Number of unique examples in test2: ", get_unique_example(get_freq_examples(dd_test2)))
    print("Number of unique examples in train+test3: ", get_unique_example(get_freq_examples(dd_train_test_3)))
    print("Number of unique examples in train3: ", get_unique_example(get_freq_examples(dd_train3)))
    print("Number of unique examples in test3: ", get_unique_example(get_freq_examples(dd_test3)))
    print("Number of unique examples in train+test4: ", get_unique_example(get_freq_examples(dd_train_test_4)))
    print("Number of unique examples in train4: ", get_unique_example(get_freq_examples(dd_train4)))
    print("Number of unique examples in test4: ", get_unique_example(get_freq_examples(dd_test4)))


    print("\n\n")


    print("Distinct examples:\n")

    print("Number of distinct examples in all folds: ", len(get_distinct_examples_in_whole(ddd_data)))

    print("\nNumber of distinct examples in all train folds: ", len(get_distinct_examples(dd_train_data)))
    print("Number of distinct examples in all test folds: ", len(get_distinct_examples(dd_test_data)))

    print("\nNumber of distinct examples in train+test0: ", len(get_distinct_examples(dd_train_test_0)))
    print("Number of distinct examples in train0: ", len(get_distinct_examples(dd_train0)))
    print("Number of distinct examples in test0: ", len(get_distinct_examples(dd_test0)))
    print("Number of distinct examples in train+test1: ", len(get_distinct_examples(dd_train_test_1)))
    print("Number of distinct examples in train1: ", len(get_distinct_examples(dd_train1)))
    print("Number of distinct examples in test1: ", len(get_distinct_examples(dd_test1)))
    print("Number of distinct examples in train+test2: ", len(get_distinct_examples(dd_train_test_2)))
    print("Number of distinct examples in train2: ", len(get_distinct_examples(dd_train2)))
    print("Number of distinct examples in test2: ", len(get_distinct_examples(dd_test2)))
    print("Number of distinct examples in train+test3: ", len(get_distinct_examples(dd_train_test_3)))
    print("Number of distinct examples in train3: ", len(get_distinct_examples(dd_train3)))
    print("Number of distinct examples in test3: ", len(get_distinct_examples(dd_test3)))
    print("Number of distinct examples in train+test4: ", len(get_distinct_examples(dd_train_test_4)))
    print("Number of distinct examples in train4: ", len(get_distinct_examples(dd_train4)))
    print("Number of distinct examples in test4: ", len(get_distinct_examples(dd_test4)))


    print("\n\n")


    print("Distinct surface forms:\n")
    s_surfaceForms = get_all_surface_forms(ddd_data)
    print("All surface forms in the whole dataset: ", len(s_surfaceForms))

    s_surfaceFormsTrain = get_all_surface_forms_in_fold(dd_train_data)
    print("\nNumber of distinct surface forms all train folds: ", len(s_surfaceFormsTrain))
    s_surfaceFormsTest = get_all_surface_forms_in_fold(dd_test_data)
    print("Number of distinct surface forms all test folds: ", len(s_surfaceFormsTest))

    s_surfaceFormsTrainTest0 = get_all_surface_forms_in_fold(dd_train_test_0)
    s_surfaceFormsTrain0 = get_all_surface_forms_in_fold(dd_train0)
    s_surfaceFormsTest0 = get_all_surface_forms_in_fold(dd_test0)
    s_surfaceFormsTrainTest1 = get_all_surface_forms_in_fold(dd_train_test_1)
    s_surfaceFormsTrain1 = get_all_surface_forms_in_fold(dd_train1)
    s_surfaceFormsTest1 = get_all_surface_forms_in_fold(dd_test1)
    s_surfaceFormsTrainTest2 = get_all_surface_forms_in_fold(dd_train_test_2)
    s_surfaceFormsTrain2 = get_all_surface_forms_in_fold(dd_train2)
    s_surfaceFormsTest2 = get_all_surface_forms_in_fold(dd_test2)
    s_surfaceFormsTrainTest3 = get_all_surface_forms_in_fold(dd_train_test_3)
    s_surfaceFormsTrain3 = get_all_surface_forms_in_fold(dd_train3)
    s_surfaceFormsTest3 = get_all_surface_forms_in_fold(dd_test3)
    s_surfaceFormsTrainTest4 = get_all_surface_forms_in_fold(dd_train_test_4)
    s_surfaceFormsTrain4 = get_all_surface_forms_in_fold(dd_train4)
    s_surfaceFormsTest4 = get_all_surface_forms_in_fold(dd_test4)

    print("\nAll surface forms in the train+test folds 0 : ", len(s_surfaceFormsTrainTest0))
    print("All surface forms in the train fold 0 : ", len(s_surfaceFormsTrain0))
    print("All surface forms in the test fold 0 : ", len(s_surfaceFormsTest0))
    print("All surface forms in the train+test folds 1 : ", len(s_surfaceFormsTrainTest1))
    print("All surface forms in the train fold 1 : ", len(s_surfaceFormsTrain1))
    print("All surface forms in the test fold 1 : ", len(s_surfaceFormsTest1))
    print("All surface forms in the train+test folds 2 : ", len(s_surfaceFormsTrainTest2))
    print("All surface forms in the train fold 2 : ", len(s_surfaceFormsTrain2))
    print("All surface forms in the test fold 2 : ", len(s_surfaceFormsTest2))
    print("All surface forms in the train+test folds 3 : ", len(s_surfaceFormsTrainTest3))
    print("All surface forms in the train fold 3 : ", len(s_surfaceFormsTrain3))
    print("All surface forms in the test fold 3 : ", len(s_surfaceFormsTest3))
    print("All surface forms in the train+test folds 4 : ", len(s_surfaceFormsTrainTest4))
    print("All surface forms in the train fold 4 : ", len(s_surfaceFormsTrain4))
    print("All surface forms in the test fold 4 : ", len(s_surfaceFormsTest4))


    print("\n\n")


    print("Unique surface forms:\n")
    print("Number of unique surface forms in the whole dataset: ", len(get_unique_surface_forms( get_freq_surface_forms_in_whole(ddd_data) )))

    print("\nNb of unique surface forms in all train folds:: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_train_data))))
    print("Nb of unique surface forms in all test folds:: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_test_data))))

    print("\nNb of unique surface forms in the train+test 0 folds: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_train_test_0))))
    print("Nb of unique surface forms in the train0 fold: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_train0))))
    print("Nb of unique surface forms in the test0 fold: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_test0))))
    print("Nb of unique surface forms in the train+test 1 folds: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_train_test_1))))
    print("Nb of unique surface forms in the train1 fold: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_train1))))
    print("Nb of unique surface forms in the test1 fold: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_test1))))
    print("Nb of unique surface forms in the train+test 2 folds: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_train_test_2))))
    print("Nb of unique surface forms in the train2 fold: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_train2))))
    print("Nb of unique surface forms in the test2 fold: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_test2))))
    print("Nb of unique surface forms in the train+test 3 folds: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_train_test_3))))
    print("Nb of unique surface forms in the train3 fold: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_train3))))
    print("Nb of unique surface forms in the test3 fold: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_test3))))
    print("Nb of unique surface forms in the train+test 4 folds: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_train_test_4))))
    print("Nb of unique surface forms in the train4 fold: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_train4))))
    print("Nb of unique surface forms in the test4 fold: ", len(get_unique_surface_forms(get_freq_surface_forms(dd_test4))))


    print("\n\n")


    print("Average number of mentions per concepts:\n")
    print("Number of of mentions per concepts in the whole dataset: ", get_average_number_mentions_per_concept_in_whole(ddd_data))

    print("\nAverage number of mentions per concepts in all train folds:: ", get_average_number_mentions_per_concept(dd_train_data))
    print("Average number of mentions per concepts in all test folds:: ", get_average_number_mentions_per_concept(dd_test_data))

    print("\nAverage number of mentions per concepts in train+test 0 folds: ", get_average_number_mentions_per_concept(dd_train_test_0))
    print("Average number of mentions per concepts in the train0 fold: ",get_average_number_mentions_per_concept(dd_train0))
    print("Average number of mentions per concepts in the test0 fold: ", get_average_number_mentions_per_concept(dd_test0))
    print("Average number of mentions per concepts in train+test 1 folds: ", get_average_number_mentions_per_concept(dd_train_test_1))
    print("Average number of mentions per concepts in the train1 fold: ",get_average_number_mentions_per_concept(dd_train1))
    print("Average number of mentions per concepts  in the test1 fold: ", get_average_number_mentions_per_concept(dd_test1))
    print("Average number of mentions per concepts in train+test 2 folds: ", get_average_number_mentions_per_concept(dd_train_test_2))
    print("Average number of mentions per concepts in the train2 fold: ",get_average_number_mentions_per_concept(dd_train2))
    print("Average number of mentions per concepts in the test2 fold: ", get_average_number_mentions_per_concept(dd_test2))
    print("Average number of mentions per concepts in train+test 3 folds: ", get_average_number_mentions_per_concept(dd_train_test_3))
    print("Average number of mentions per concepts in the train3 fold: ",get_average_number_mentions_per_concept(dd_train3))
    print("Average number of mentions per concepts in the test3 fold: ", get_average_number_mentions_per_concept(dd_test3))
    print("Average number of mentions per concepts in train+test 4 folds: ", get_average_number_mentions_per_concept(dd_train_test_4))
    print("Average number of mentions per concepts in the train4 fold: ",get_average_number_mentions_per_concept(dd_train4))
    print("Average number of mentions per concepts in the test4 fold: ", get_average_number_mentions_per_concept(dd_test4))


    print("Standard deviation and median of mentions per concepts:\n")
    print("Standard deviation and median of of mentions per concepts in the whole dataset: ", get_std_number_mentions_per_concept_in_whole(ddd_data))

    print("\nStandard deviation and median of mentions per concepts in all train folds:: ", get_std_number_mentions_per_concept(dd_train_data))
    print("Standard deviation and median of mentions per concepts in all test folds:: ", get_std_number_mentions_per_concept(dd_test_data))

    print("\nStandard deviation and median of mentions per concepts in train+test 0 folds: ", get_std_number_mentions_per_concept(dd_train_test_0))
    print("Standard deviation and median of mentions per concepts in the train0 fold: ",get_std_number_mentions_per_concept(dd_train0))
    print("Standard deviation and median of mentions per concepts in the test0 fold: ", get_std_number_mentions_per_concept(dd_test0))
    print("Standard deviation and median of mentions per concepts in train+test 1 folds: ", get_std_number_mentions_per_concept(dd_train_test_1))
    print("Standard deviation and median of mentions per concepts in the train1 fold: ",get_std_number_mentions_per_concept(dd_train1))
    print("Standard deviation and median of mentions per concepts  in the test1 fold: ", get_std_number_mentions_per_concept(dd_test1))
    print("Standard deviation and median of mentions per concepts in train+test 2 folds: ", get_std_number_mentions_per_concept(dd_train_test_2))
    print("Standard deviation and median of mentions per concepts in the train2 fold: ",get_std_number_mentions_per_concept(dd_train2))
    print("Standard deviation and median of mentions per concepts in the test2 fold: ", get_std_number_mentions_per_concept(dd_test2))
    print("Standard deviation and median of mentions per concepts in train+test 3 folds: ", get_std_number_mentions_per_concept(dd_train_test_3))
    print("Standard deviation and median of mentions per concepts in the train3 fold: ",get_std_number_mentions_per_concept(dd_train3))
    print("Standard deviation and median of mentions per concepts in the test3 fold: ", get_std_number_mentions_per_concept(dd_test3))
    print("Standard deviation and median of mentions per concepts in train+test 4 folds: ", get_std_number_mentions_per_concept(dd_train_test_4))
    print("Standard deviation and median of mentions per concepts in the train4 fold: ",get_std_number_mentions_per_concept(dd_train4))
    print("Standard deviation and median of mentions per concepts in the test4 fold: ", get_std_number_mentions_per_concept(dd_test4))


    print("\n\n")


    print("Single shot situation?\n")
    print("Number concepts with only one mention in the whole dataset: ", get_nb_concepts_with_only_one_mention_in_whole(ddd_data))

    print("\nNumber concepts with only one mention in all train folds:: ", get_nb_concepts_with_only_one_mention(dd_train_data))
    print("Number concepts with only one mention in all test folds:: ", get_nb_concepts_with_only_one_mention(dd_test_data))

    print("\nNumber concepts with only one mention in train+test 0 folds: ", get_nb_concepts_with_only_one_mention(dd_train_test_0))
    print("Number concepts with only one mention in the train0 fold: ",get_nb_concepts_with_only_one_mention(dd_train0))
    print("Number concepts with only one mention in the test0 fold: ", get_nb_concepts_with_only_one_mention(dd_test0))
    print("Number concepts with only one mention in train+test 1 folds: ", get_nb_concepts_with_only_one_mention(dd_train_test_1))
    print("Number concepts with only one mention in the train1 fold: ",get_nb_concepts_with_only_one_mention(dd_train1))
    print("Number concepts with only one mention  in the test1 fold: ", get_nb_concepts_with_only_one_mention(dd_test1))
    print("Number concepts with only one mention in train+test 2 folds: ", get_nb_concepts_with_only_one_mention(dd_train_test_2))
    print("Number concepts with only one mention in the train2 fold: ",get_nb_concepts_with_only_one_mention(dd_train2))
    print("Number concepts with only one mention in the test2 fold: ", get_nb_concepts_with_only_one_mention(dd_test2))
    print("Number concepts with only one mention in train+test 3 folds: ", get_nb_concepts_with_only_one_mention(dd_train_test_3))
    print("Number concepts with only one mention in the train3 fold: ",get_nb_concepts_with_only_one_mention(dd_train3))
    print("Number concepts with only one mention in the test3 fold: ", get_nb_concepts_with_only_one_mention(dd_test3))
    print("Number concepts with only one mention in train+test 4 folds: ", get_nb_concepts_with_only_one_mention(dd_train_test_4))
    print("Number concepts with only one mention in the train4 fold: ",get_nb_concepts_with_only_one_mention(dd_train4))
    print("Number concepts with only one mention in the test4 fold: ", get_nb_concepts_with_only_one_mention(dd_test4))


    print("\n\n")


    print("How many surface forms have more than one annotating concept (=ambiguity mention) ?\n")
    print("Number of surface forms with different possible labels in the whole dataset: ", get_number_of_surface_forms_with_different_labels_in_whole(ddd_data))

    print("\nNumber of surface forms with different possible labels in all train folds:: ", get_number_of_surface_forms_with_different_labels(dd_train_data))
    print("Number of surface forms with different possible labels in all test folds:: ", get_number_of_surface_forms_with_different_labels(dd_test_data))

    print("\nNumber of surface forms with different possible labels in train+test 0 folds: ", get_number_of_surface_forms_with_different_labels(dd_train_test_0))
    print("Number of surface forms with different possible labels in the train0 fold: ",get_number_of_surface_forms_with_different_labels(dd_train0))
    print("Number of surface forms with different possible labels in the test0 fold: ", get_number_of_surface_forms_with_different_labels(dd_test0))
    print("Number of surface forms with different possible labels in train+test 1 folds: ", get_number_of_surface_forms_with_different_labels(dd_train_test_1))
    print("Number of surface forms with different possible labels in the train1 fold: ",get_number_of_surface_forms_with_different_labels(dd_train1))
    print("Number of surface forms with different possible labels  in the test1 fold: ", get_number_of_surface_forms_with_different_labels(dd_test1))
    print("Number of surface forms with different possible labels in train+test 2 folds: ", get_number_of_surface_forms_with_different_labels(dd_train_test_2))
    print("Number of surface forms with different possible labels in the train2 fold: ",get_number_of_surface_forms_with_different_labels(dd_train2))
    print("Number of surface forms with different possible labels in the test2 fold: ", get_number_of_surface_forms_with_different_labels(dd_test2))
    print("Number of surface forms with different possible labels in train+test 3 folds: ", get_number_of_surface_forms_with_different_labels(dd_train_test_3))
    print("Number of surface forms with different possible labels in the train3 fold: ",get_number_of_surface_forms_with_different_labels(dd_train3))
    print("Number of surface forms with different possible labels in the test3 fold: ", get_number_of_surface_forms_with_different_labels(dd_test3))
    print("Number of surface forms with different possible labels in train+test 4 folds: ", get_number_of_surface_forms_with_different_labels(dd_train_test_4))
    print("Number of surface forms with different possible labels in the train4 fold: ",get_number_of_surface_forms_with_different_labels(dd_train4))
    print("Number of surface forms with different possible labels in the test4 fold: ", get_number_of_surface_forms_with_different_labels(dd_test4))


    print("\n\n")
    ########################
    # Extrinsic analysis:
    ########################








    sys.exit(0)


