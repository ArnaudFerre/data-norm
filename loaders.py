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


def loader_all_cadec_folds(repPath, subDatasetType="full"):
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

                    mention, label = line.split('\t')
                    ddd_data[foldFileNameWithoutExt][exampleId]["mention"] = mention
                    ddd_data[foldFileNameWithoutExt][exampleId]["label"] = label.rstrip()

                    i += 1

    return ddd_data










#######################################################################################################






def loader_cadec_one_fold_surface(foldPath):
    d_data = dict()

    with open(foldPath) as file:
        for line in file:
            mention, label = line.split('\t')
            d_data[mention] = label.rstrip()

    return d_data



def loader_cadec_one_fold_mention(foldPath):
    d_data = dict()

    i = 0
    with open(foldPath) as file:
        for line in file:
            exampleId = "cadec_" + "{number:06}".format(number=i)
            d_data[exampleId] = dict()

            mention, label = line.split('\t')
            d_data[exampleId]["mention"] = mention
            d_data[exampleId]["label"] = label.rstrip()
            i+=1

    return d_data




def loader_all_cadec(repPath, subDatasetType="full"):
    dd_data = dict()

    i = 0
    for foldFileName in listdir(repPath):
        foldFilePath = join(repPath, foldFileName)

        if isfile(foldFilePath):

            # Select all folds (train and test):
            if subDatasetType == "full":
                print("Load", foldFileName)
                with open(foldFilePath) as foldFile:

                    for line in foldFile:

                        s_exampleId = "cadec_"+"{number:06}".format(number=i)

                        dd_data[s_exampleId] = dict()
                        mention, label = line.split('\t')
                        dd_data[s_exampleId][mention] = label.rstrip()

                        i += 1

            # Select only folds with "train" or "test" in its name:
            elif subDatasetType == "train" or subDatasetType == "test":

                if re.match(subDatasetType, foldFileName):
                    print("Load", foldFileName)
                    with open(foldFilePath) as foldFile:

                        for line in foldFile:
                            s_exampleId = "cadec_" + "{number:06}".format(number=i)

                            dd_data[s_exampleId] = dict()
                            mention, label = line.split('\t')
                            dd_data[s_exampleId][mention] = label.rstrip()

                            i += 1

    return dd_data





def detailedLoader_all_cadec(repPath, subDatasetType="full"):
    dd_data = dict()

    i = 0
    for foldFileName in listdir(repPath):
        foldFilePath = join(repPath, foldFileName)

        if isfile(foldFilePath):

            # Select all folds (train and test):
            if subDatasetType == "full":
                print("Load", foldFileName)
                with open(foldFilePath) as foldFile:

                    for line in foldFile:

                        s_exampleId = "cadec_"+"{number:06}".format(number=i)
                        mention, label = line.split('\t')

                        dd_data[s_exampleId] = dict()
                        dd_data[s_exampleId]["mention"] = mention
                        dd_data[s_exampleId]["label"] = label.rstrip()
                        dd_data[s_exampleId]["file"] = foldFileName

                        i += 1

            # Select only folds with "train" or "test" in its name:
            elif subDatasetType == "train" or subDatasetType == "test":

                if re.match(subDatasetType, foldFileName):
                    print("Load", foldFileName)
                    with open(foldFilePath) as foldFile:

                        for line in foldFile:

                            s_exampleId = "cadec_" + "{number:06}".format(number=i)
                            mention, label = line.split('\t')

                            dd_data[s_exampleId] = dict()
                            dd_data[s_exampleId]["mention"] = mention
                            dd_data[s_exampleId]["label"] = label.rstrip()
                            dd_data[s_exampleId]["file"] = foldFileName

                            i += 1

    return dd_data




#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':


    ########################
    # Open all folds

    #Option 1: folds data
    ddd_data = loader_all_cadec_folds("../CADEC/2_Custom_folds/")
    print(ddd_data)









    sys.exit(0)



    #Option 2: train/dev/test data
    # dd_data

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


