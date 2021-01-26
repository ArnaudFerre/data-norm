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


###################################################
# Inter-analysers:
###################################################





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



#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':


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


    print("\n\n")











    sys.exit(0)





    ########################

    m=str(1)

    foldPath = "../CADEC/2_Custom_folds/test_"+m+".csv"
    print("load: ", foldPath)
    dd_data_test = loader_cadec_one_fold_mention(foldPath)
    print(len(dd_data_test.keys()), dd_data_test)


    foldPath = "../CADEC/2_Custom_folds/train_"+m+".csv"
    print("load: ", foldPath)
    dd_data_train = loader_cadec_one_fold_mention(foldPath)
    print(len(dd_data_train.keys()), dd_data_train)









    ########################
    #dd_data_train = dd_data_test
    ########################




    print("\n\n")


    s_labels = set()
    for id in dd_data_train.keys():
        s_labels.add(dd_data_train[id]["label"])
    print("Nb labels: ", len(s_labels))


    print("\n\n")


    s_surfaceForms = set()
    for id in dd_data_train.keys():
        if dd_data_train[id]["mention"] not in s_surfaceForms:
            s_surfaceForms.add(dd_data_train[id]["mention"])

    print("Nb formes de surface: ", len(s_surfaceForms))


    print("\n\n")


    k=0
    for id in dd_data_train.keys():
        mention = dd_data_train[id]["mention"]
        label = dd_data_train[id]["label"]

        for idTest in dd_data_test.keys():
            if dd_data_test[idTest]["mention"] == mention:# and dd_data_test[idTest]["label"] == label:
                k+=1
                break


    print("recouvrement train/test: ", k)


    print("\n\n")


    d_surfaceCount = dict()
    for id in dd_data_train.keys():
        surface = dd_data_train[id]["mention"]
        d_surfaceCount[surface] = 0
    for id in dd_data_train.keys():
        surface = dd_data_train[id]["mention"]
        d_surfaceCount[surface] += 1

    cmpt = 0
    for surface in d_surfaceCount.keys():
        if d_surfaceCount[surface] == 1:
            cmpt += 1

    print("d_surfaceCount: ", d_surfaceCount)
    print("unique:", cmpt)


    print("\n\n")


    d_mentionsPerConcepts = dict()
    for id in dd_data_train.keys():
        concept = dd_data_train[id]["label"]
        d_mentionsPerConcepts[concept] = 0
    for id in dd_data_train.keys():
        concept = dd_data_train[id]["label"]
        d_mentionsPerConcepts[concept] += 1

    print("d_mentionsPerConcepts: ", d_mentionsPerConcepts)

    mean = 0
    for concept in d_mentionsPerConcepts.keys():
        mean += d_mentionsPerConcepts[concept]
    mean = (1.0*mean) / len(d_mentionsPerConcepts)

    print("avg: ", mean)


    print("\n\n")


    d_surfacesToConcepts = dict()
    for id in dd_data_train.keys():
        mention = dd_data_train[id]["mention"]
        d_surfacesToConcepts[mention] = dict()
        d_surfacesToConcepts[mention]["set"] = set()
        d_surfacesToConcepts[mention]["labelsCount"] = 0
    for id in dd_data_train.keys():
        mention = dd_data_train[id]["mention"]
        label = dd_data_train[id]["label"]
        d_surfacesToConcepts[mention]["set"].add(label)

    for surface in d_surfacesToConcepts.keys():
        d_surfacesToConcepts[surface]["labelsCount"] = len(d_surfacesToConcepts[mention]["set"])

    i = 0
    for surface in d_surfacesToConcepts.keys():
        if d_surfacesToConcepts[surface]["labelsCount"] > 1:
            i+=1
    print("count i: ", i)
    print(d_surfacesToConcepts)


    print("\n\n")


    cmptNonUnique = 0
    i = 0
    for id1 in dd_data_train.keys():
        mention = dd_data_train[id1]["mention"]
        label = dd_data_train[id1]["label"]

        for id2 in dd_data_train.keys():
            if id1 != id2:
                if  dd_data_train[id2]["mention"] == mention :
                    print(mention, label, dd_data_train[id2]["label"])
                    i+=1
                    if dd_data_train[id2]["label"] == label:
                        cmptNonUnique += 1

    print("cmptNonUnique: ", cmptNonUnique)
    print("i:", i)




    print("\n\n")

    d_data = dict()
    i = 0
    with open("../CADEC/2_Custom_folds/vocab.txt") as file:
        for line in file:
            exampleId = "SCT_" + "{number:04}".format(number=i)
            d_data[exampleId] = dict()
            d_data[exampleId] = line.rstrip()
            i += 1

    print(d_data)


    s_labels = set()
    for id in d_data.keys():
        s_labels.add(d_data[id])
    print("len(s_labels): ", len(s_labels))

    i=0
    for id in dd_data_train.keys():
        if dd_data_train[id]["label"] not in s_labels:
            print(dd_data_train[id]["label"])
        else:
            i+=1
    print(i)












    sys.exit(0)

    repPath = "../CADEC/2_Custom_folds/"
    dd_data_full = loader_all_cadec(repPath)
    print("Nb of mentions-labels:", len(dd_data_full.keys()))


    print("\n\n")


    dd_data_train = loader_all_cadec(repPath, "train")
    print("Nb of mentions-labels:", len(dd_data_train.keys()))


    print("\n\n")


    dd_data_test = loader_all_cadec(repPath, "test")
    print("Nb of mentions-labels:", len(dd_data_test.keys()))


    print("\n\n")


    ############
    dd_data_tmp = dd_data_train # dd_data_test #  dd_data_full #  dd_data_test #
    print("dd_data_tmp:",dd_data_tmp)
    ############


    print("\n\n")


    # Unique mentions:

    d_uniqueData = dict()
    for id in dd_data_tmp.keys():
        for mention in dd_data_tmp[id].keys():
            d_uniqueData[mention] = list()

    for id in dd_data_tmp.keys():
        for mention in dd_data_tmp[id].keys():
            d_uniqueData[mention].append(dd_data_tmp[id][mention])

    print("d_uniqueData: ", d_uniqueData)

    j = 0
    for mention in d_uniqueData.keys():
        if len(d_uniqueData[mention]) > 1:
            j+=1
    print(j)

    print("Nb of surface forms:", len(d_uniqueData.keys()))


    print("\n\n")


    d_surfaceFormsCount = dict()
    for id in dd_data_tmp.keys():
        for mention in dd_data_tmp[id].keys():
            d_surfaceFormsCount[mention] = 0
    for id in dd_data_tmp.keys():
        for mention in dd_data_tmp[id].keys():
            d_surfaceFormsCount[mention] += 1

    print("d_surfaceFormsCount:",d_surfaceFormsCount)

    uniqueSF = 0
    for mention in d_surfaceFormsCount.keys():
        if d_surfaceFormsCount[mention] !=5:
            print(mention)
            uniqueSF += 1
    print("Nb of unique surface forms:", uniqueSF)


    print("\n\n")


    # Create the reference:
    d_concepts = dict()
    s_concepts = set()
    for id in dd_data_tmp.keys():
        for mention in dd_data_tmp[id].keys():
            s_concepts.add(dd_data_tmp[id][mention])
            d_concepts[dd_data_tmp[id][mention]] = 0

    for id in dd_data_tmp.keys():
        for mention in dd_data_tmp[id].keys():
            d_concepts[dd_data_tmp[id][mention]] += 1

    print(d_concepts)
    print(len(s_concepts), s_concepts)
    print("Nb de concepts utilises:",len(d_concepts))


    max = 0
    for label in d_concepts.keys():
        if d_concepts[label] > max:
            max = d_concepts[label]
    print("max=", max)


    mean = 0
    for label in d_concepts.keys():
        mean += d_concepts[label]
    mean = (1.0*mean) / len(d_concepts)
    print("mean=",mean)


    print("\n\n")


    # Nb of different labels per mention:
    print("Surface forms with different labels:")
    d_dictTmp = dict()
    for mention in d_uniqueData.keys():
        d_dictTmp[mention] = len(set(d_uniqueData[mention]))
        if d_dictTmp[mention] > 1:
            print(mention, set(d_uniqueData[mention]))


    print("\n\n")


    print("NIL?")
    for id in dd_data_full.keys():
        for mention in dd_data_full[id].keys():
            try:
                label1, label2 = dd_data_full[id][mention].split('\t')
                print(mention)
            except:
                pass


    print("----------------------------\n\n--------------------------")


    dd_data_full = detailedLoader_all_cadec(repPath)
    #print(dd_data_full)

    d_surfaceFormsCount = dict()
    for id in dd_data_full.keys():
        d_surfaceFormsCount[dd_data_full[id]["mention"]] = 0
    for id in dd_data_full.keys():
        d_surfaceFormsCount[dd_data_full[id]["mention"]] += 1
    print(d_surfaceFormsCount)

    j=0
    for mention in d_surfaceFormsCount.keys():
        if d_surfaceFormsCount[mention] == 1:
            j+=1
    print(j)


