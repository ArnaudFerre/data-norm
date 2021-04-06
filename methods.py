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

def by_heart_matcher(dd_mentions, dd_lesson):
    """
   Description: If a mention has been seen in dd_lesson, then predict same
   """
    dd_predictions = dict()
    nbMentions = len(dd_mentions.keys())
    progresssion = -1

    dd_count = dict()

    for i, id in enumerate(dd_mentions.keys()):
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []
        mention = dd_mentions[id]["mention"]
        dd_count[mention] = dict()

    # Initialization of the counting of different possible CUI for a same surface form mention:
    for i, id in enumerate(dd_mentions.keys()):
        mention = dd_mentions[id]["mention"]
        for idByHeart in dd_lesson.keys():
            if dd_lesson[idByHeart]["mention"] == mention:
                for cui in dd_lesson[idByHeart]["cui"]:
                    dd_count[mention][cui] = 0

    # Counting of different possible CUI for a same surface form mention:
    for i, id in enumerate(dd_mentions.keys()):
        mention = dd_mentions[id]["mention"]
        for idByHeart in dd_lesson.keys():
            if dd_lesson[idByHeart]["mention"] == mention:
                for cui in dd_lesson[idByHeart]["cui"]:
                    dd_count[mention][cui] += 1

    # Choosing (one of) the most frequent predicted CUI for each seen mention:
    for i, id in enumerate(dd_mentions.keys()):
        mention = dd_mentions[id]["mention"]
        compt = 0
        for possibleCui in dd_count[mention].keys():
            if dd_count[mention][possibleCui] > compt:
                compt = dd_count[mention][possibleCui]

        for possibleCui in dd_count[mention].keys():
            if dd_count[mention][possibleCui] == compt:
                dd_predictions[id]["mention"] = mention
                dd_predictions[id]["pred_cui"] = [possibleCui]
                break

        # Print progression:
        currentProgression = round(100 * (i / nbMentions))
        if currentProgression > progresssion:
            print(str(currentProgression)+"%", end=" ")
            progresssion = currentProgression

    return dd_predictions



def exact_matcher(dd_mentions, dd_ref):
    """
    Description: Find the first exact match between each mention and label
    :param dd_mentions:
    :param dd_ref:
    :return:
    """
    dd_predictions = dict()
    nbMentions = len(dd_mentions.keys())
    progresssion = -1


    for id in dd_mentions.keys():
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []


    for i, id in enumerate(dd_mentions.keys()):
        mention = dd_mentions[id]["mention"]

        for cui in dd_ref.keys():
            l_tags = list()
            l_tags.append(dd_ref[cui]["label"])
            if "tags" in dd_ref[cui].keys():
                for tag in dd_ref[cui]["tags"]:
                    l_tags.append(tag)

            if mention in l_tags:
                dd_predictions[id]["mention"] = dd_mentions[id]["mention"]
                dd_predictions[id]["pred_cui"].append(cui)
                dd_predictions[id]["label"] = dd_ref[cui]["label"]


        #Print progression:
        currentProgression = round(100*(i/nbMentions))
        if currentProgression > progresssion:
            print(currentProgression, "%")
            progresssion = currentProgression

    return dd_predictions


###
# Maybe improve all rule-based methods by adding training mentions in tags of concepts.
###



def lemmes_exact_matcher():
    return None


def roots_exact_matcher():
    return None


def sieve():
    return None

def pyDNorm():
    return None



###################################################
# Tools:
###################################################

def lowercaser_mentions(dd_mentions):
    for id in dd_mentions.keys():
        dd_mentions[id]["mention"] = dd_mentions[id]["mention"].lower()
    return dd_mentions


def lowercaser_ref(dd_ref):
    for cui in dd_ref.keys():
        dd_ref[cui]["label"] = dd_ref[cui]["label"].lower()
        if "tags" in dd_ref[cui].keys():
            l_lowercasedTags = list()
            for tag in dd_ref[cui]["tags"]:
                l_lowercasedTags.append(tag.lower())
            dd_ref[cui]["tags"] = l_lowercasedTags
    return dd_ref


#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':

    from loaders import loader_clinical_finding_file, loader_amt, select_subpart_with_patterns_in_label, get_tags_in_ref, fusion_ref, get_cui_list
    from loaders import extract_data_without_file, loader_all_initial_cadec_folds, loader_all_random_cadec_folds, loader_all_custom_cadec_folds
    from loaders import loader_ontobiotope, select_subpart_hierarchy, loader_one_bb4_fold
    from loaders import loader_medic, loader_one_ncbi_fold, extract_data

    ################################################
    print("\n\n\nCADEC (3 datasets):\n")
    ################################################


    print("Loading of Clinical finding from SCT-AU...")
    dd_subSct = loader_clinical_finding_file("../CADEC/clinicalFindingSubPart.csv")
    print('Loaded.')

    print("loading AMTv2.56...")
    dd_amt = loader_amt("../CADEC/AMT_v2.56/Uuid_sct_concepts_au.gov.nehta.amt.standalone_2.56.txt")
    dd_subAmt = select_subpart_with_patterns_in_label(dd_amt)
    print("Done. (Nb of concepts in this subpart AMT =", len(dd_subAmt.keys()), ", Nb of tags =",
          len(get_tags_in_ref(dd_subAmt)), ")")

    print("\nFusion subSCT & subAMT in one reference...")
    dd_subsubRef = fusion_ref(dd_subSct, dd_subAmt)
    print("done. (Nb of concepts in subSCT+subAMT =", len(dd_subsubRef.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_subsubRef)), ")")

    print("\nLoading CUIs list used by [Miftahutdinov et al. 2019]...")
    l_sctFromCadec = get_cui_list("../CADEC/custom_CUI_list.txt")
    print("loaded.(Nb of concepts in the list (with CONCEPT_LESS =", len(l_sctFromCadec), ")")

    print("\n\nLoading initial CADEC corpus...")
    ddd_data = loader_all_initial_cadec_folds("../CADEC/3_MyCADEC/", "../CADEC/0_Initial_CADEC/CADEC_2_2016/original/")
    dd_initCadecDrugs = extract_data(ddd_data, l_type=["Drug"])
    dd_initCadecClinicalFindings = extract_data(ddd_data, l_type=["ADR", "Finding", "Disease", "Symptom"])
    dd_initCadec = extract_data_without_file(ddd_data)
    print("loaded.(Nb of mentions in initial CADEC =", len(dd_initCadec.keys()), ")")
    print("(Nb of drug mentions:", len(dd_initCadecDrugs.keys()), " ; Nb of clinical finding mentions:", len(dd_initCadecClinicalFindings.keys()), ")")

    print("\nLoading random CADEC corpus...")
    ddd_randData = loader_all_random_cadec_folds("../CADEC/1_Random_folds_AskAPatient/")
    dd_randCadec = extract_data_without_file(ddd_randData)
    print("loaded.(Nb of mentions in ALL folds for random CADEC =", len(dd_randCadec.keys()), ")")

    print("\nLoading custom CADEC corpus...")
    ddd_customData = loader_all_custom_cadec_folds("../CADEC/2_Custom_folds/")
    dd_customCadec = extract_data_without_file(ddd_customData)
    print("loaded.(Nb of mentions in ALL folds for custom CADEC =", len(dd_customCadec.keys()), ")")

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
    dd_habTrain = extract_data(ddd_dataTrain, l_type=["Habitat"])  # ["Habitat", "Phenotype", "Microorganism"]
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

    ################################################
    print("\n\n\n\nNCBI:\n")
    ################################################

    print("\nLoading MEDIC...")
    dd_medic = loader_medic("../NCBI/CTD_diseases_DNorm_v2012_07_6.tsv")
    print("loaded. (Nb of concepts in MEDIC =", len(dd_medic.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_medic)), ")")
    print("Note: Inconsistence errors in MEDIC -> MESH:D006938 = OMIM:144010 and MESH:C537710 = OMIM:153400.")


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



    ################################################
    print("\n\n\n\nPREPROCESSINGS:\n")
    ################################################

    print("\nLowercase mentions datasets...")
    dd_randCADEC_train0_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-0.train"])
    dd_randCADEC_train1_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-1.train"])
    dd_randCADEC_train2_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-2.train"])
    dd_randCADEC_train3_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-3.train"])
    dd_randCADEC_train4_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-4.train"])
    dd_randCADEC_train5_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-5.train"])
    dd_randCADEC_train6_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-6.train"])
    dd_randCADEC_train7_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-7.train"])
    dd_randCADEC_train8_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-8.train"])
    dd_randCADEC_train9_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-9.train"])

    dd_randCADEC_validation0_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-0.validation"])
    dd_randCADEC_validation1_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-1.validation"])
    dd_randCADEC_validation2_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-2.validation"])
    dd_randCADEC_validation3_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-3.validation"])
    dd_randCADEC_validation4_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-4.validation"])
    dd_randCADEC_validation5_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-5.validation"])
    dd_randCADEC_validation6_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-6.validation"])
    dd_randCADEC_validation7_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-7.validation"])
    dd_randCADEC_validation8_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-8.validation"])
    dd_randCADEC_validation9_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-9.validation"])

    dd_customCADEC_train0_lowercased = lowercaser_mentions(ddd_customData["train_0"])
    dd_customCADEC_train1_lowercased = lowercaser_mentions(ddd_customData["train_1"])
    dd_customCADEC_train2_lowercased = lowercaser_mentions(ddd_customData["train_2"])
    dd_customCADEC_train3_lowercased = lowercaser_mentions(ddd_customData["train_3"])
    dd_customCADEC_train4_lowercased = lowercaser_mentions(ddd_customData["train_4"])

    dd_customCADEC_validation0_lowercased = lowercaser_mentions(ddd_customData["test_0"])
    dd_customCADEC_validation1_lowercased = lowercaser_mentions(ddd_customData["test_1"])
    dd_customCADEC_validation2_lowercased = lowercaser_mentions(ddd_customData["test_2"])
    dd_customCADEC_validation3_lowercased = lowercaser_mentions(ddd_customData["test_3"])
    dd_customCADEC_validation4_lowercased = lowercaser_mentions(ddd_customData["test_4"])

    dd_BB4habTrain_lowercased = lowercaser_mentions(dd_habTrain)
    dd_BB4habDev_lowercased = lowercaser_mentions(dd_habDev)

    dd_NCBITrainFixed_lowercased = lowercaser_mentions(dd_TrainFixed)
    dd_NCBIDevFixed_lowercased = lowercaser_mentions(dd_DevFixed)

    print("Mentions lowercasing done.\n")


    print("Lowercase references...")
    dd_subsubRef_lowercased = lowercaser_ref(dd_subsubRef)
    dd_habObt_lowercased = lowercaser_ref(dd_medic)
    dd_medic_lowercased = lowercaser_ref(dd_medic)
    print("Done.")



    ################################################
    print("\n\n\n\nPREDICTING:\n")
    ################################################
    from evaluators import accuracy
    """
    print("By heart learning method:")

    dd_predictions_customCADEC0_onTrain = by_heart_matcher(dd_customCADEC_train0_lowercased, dd_customCADEC_train0_lowercased)
    BHscorecustomCADEC0_onTrain = accuracy(dd_predictions_customCADEC0_onTrain, dd_customCADEC_train0_lowercased)
    print("\nBHscorecustomCADEC0_onTrain:", BHscorecustomCADEC0_onTrain)

    dd_predictions_customCADEC0_onVal = by_heart_matcher(dd_customCADEC_validation0_lowercased, dd_customCADEC_train0_lowercased)
    BHscorecustomCADEC0_onVal = accuracy(dd_predictions_customCADEC0_onVal, dd_customCADEC_validation0_lowercased)
    print("\nBHscorecustomCADEC0_onVal:", BHscorecustomCADEC0_onVal)


    dd_predictions_randCADEC0_onTrain = by_heart_matcher(dd_randCADEC_train0_lowercased, dd_randCADEC_train0_lowercased)
    BHscoreRandCADEC0_onTrain = accuracy(dd_predictions_randCADEC0_onTrain, dd_randCADEC_train0_lowercased)
    print("\nBHscoreRandCADEC0_onTrain:", BHscoreRandCADEC0_onTrain)

    dd_predictions_randCADEC0_onVal = by_heart_matcher(dd_randCADEC_validation0_lowercased, dd_randCADEC_train0_lowercased)
    BHscoreRandCADEC0_onVal = accuracy(dd_predictions_randCADEC0_onVal, dd_randCADEC_validation0_lowercased)
    print("\nBHscoreRandCADEC0_onVal:", BHscoreRandCADEC0_onVal)


    dd_predictions_BB4_onTrain = by_heart_matcher(dd_BB4habTrain_lowercased, dd_BB4habTrain_lowercased)
    BHscore_BB4_onTrain = accuracy(dd_predictions_BB4_onTrain, dd_habTrain)
    print("\nBHscore_BB4_onTrain:", BHscore_BB4_onTrain)

    dd_predictions_BB4_onVal = by_heart_matcher(dd_BB4habDev_lowercased, dd_BB4habTrain_lowercased)
    BHscore_BB4_onVal = accuracy(dd_predictions_BB4_onVal, dd_BB4habDev_lowercased)
    print("\nBHscore_BB4_onVal:", BHscore_BB4_onVal)


    dd_predictions_NCBI_onTrain = by_heart_matcher(dd_NCBITrainFixed_lowercased, dd_NCBITrainFixed_lowercased)
    BHscore_NCBI_onTrain = accuracy(dd_predictions_NCBI_onTrain, dd_NCBITrainFixed_lowercased)
    print("\nBHscore_NCBI_onTrain:", BHscore_NCBI_onTrain)

    dd_predictions_NCBI_onVal = by_heart_matcher(dd_NCBIDevFixed_lowercased, dd_NCBITrainFixed_lowercased)
    BHscore_NCBI_onVal = accuracy(dd_predictions_NCBI_onVal, dd_NCBIDevFixed_lowercased)
    print("\nBHscore_NCBI_onVal:", BHscore_NCBI_onVal)
    """


    print("\n\nExact Matching method:")
    """
    dd_EMpredictions_customCADEC0_onTrain = exact_matcher(dd_customCADEC_train0_lowercased, dd_subsubRef_lowercased)
    EMscorecustomCADEC0_onTrain = accuracy(dd_EMpredictions_customCADEC0_onTrain, dd_customCADEC_train0_lowercased)
    print("\nEMscorecustomCADEC0_onTrain:", EMscorecustomCADEC0_onTrain)

    dd_EMpredictions_customCADEC0_onVal = exact_matcher(dd_customCADEC_validation0_lowercased, dd_subsubRef_lowercased)
    EMscorecustomCADEC0_onVal = accuracy(dd_EMpredictions_customCADEC0_onVal, dd_customCADEC_validation0_lowercased)
    print("\nEMscorecustomCADEC0_onVal:", EMscorecustomCADEC0_onVal)


    dd_EMpredictions_randCADEC0_onTrain = exact_matcher(dd_randCADEC_train0_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC0_onTrain = accuracy(dd_EMpredictions_randCADEC0_onTrain, dd_randCADEC_train0_lowercased)
    print("\nEMscoreRandCADEC0_onTrain:", EMscoreRandCADEC0_onTrain)

    dd_EMpredictions_randCADEC0_onVal = exact_matcher(dd_randCADEC_validation0_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC0_onVal = accuracy(dd_EMpredictions_randCADEC0_onVal, dd_randCADEC_validation0_lowercased)
    print("\nEMscoreRandCADEC0_onVal:", EMscoreRandCADEC0_onVal)
    """

    dd_EMpredictions_BB4_onTrain = exact_matcher(dd_BB4habTrain_lowercased, dd_habObt_lowercased)
    EMscore_BB4_onTrain = accuracy(dd_EMpredictions_BB4_onTrain, dd_habTrain)
    print("\nEMscore_BB4_onTrain:", EMscore_BB4_onTrain)

    


    sys.exit(0)

    dd_EMpredictions_BB4_onVal = exact_matcher(dd_BB4habDev_lowercased, dd_habObt_lowercased)
    EMscore_BB4_onVal = accuracy(dd_EMpredictions_BB4_onVal, dd_habDev)
    print("\nEMscore_BB4_onVal:", EMscore_BB4_onVal)


    dd_EMpredictions_NCBI_onTrain = exact_matcher(dd_NCBITrainFixed_lowercased, dd_medic_lowercased)
    EMscore_NCBI_onTrain = accuracy(dd_EMpredictions_NCBI_onTrain, dd_NCBITrainFixed_lowercased)
    print("\nEMscore_NCBI_onTrain:", EMscore_NCBI_onTrain)

    dd_EMpredictions_NCBI_onVal = exact_matcher(dd_NCBIDevFixed_lowercased, dd_medic_lowercased)
    EMscore_NCBI_onVal = accuracy(dd_EMpredictions_NCBI_onVal, dd_NCBIDevFixed_lowercased)
    print("\nEMscore_NCBI_onVal:", EMscore_NCBI_onVal)










    sys.exit(0)

    print("By heart learning method:")

    print("Random CADEC (10 train-> 10 validation):\n")
    dd_predictions_CADEC0 = by_heart_matcher(dd_randCADEC_train0_lowercased, dd_randCADEC_train0_lowercased)
    dd_predictions_CADEC1 = by_heart_matcher(dd_randCADEC_train1_lowercased, dd_randCADEC_train1_lowercased)
    dd_predictions_CADEC2 = by_heart_matcher(dd_randCADEC_train2_lowercased, dd_randCADEC_train2_lowercased)
    dd_predictions_CADEC3 = by_heart_matcher(dd_randCADEC_train3_lowercased, dd_randCADEC_train3_lowercased)
    dd_predictions_CADEC4 = by_heart_matcher(dd_randCADEC_train4_lowercased, dd_randCADEC_train4_lowercased)
    dd_predictions_CADEC5 = by_heart_matcher(dd_randCADEC_train5_lowercased, dd_randCADEC_train5_lowercased)
    dd_predictions_CADEC6 = by_heart_matcher(dd_randCADEC_train6_lowercased, dd_randCADEC_train6_lowercased)
    dd_predictions_CADEC7 = by_heart_matcher(dd_randCADEC_train7_lowercased, dd_randCADEC_train7_lowercased)
    dd_predictions_CADEC8 = by_heart_matcher(dd_randCADEC_train8_lowercased, dd_randCADEC_train8_lowercased)
    dd_predictions_CADEC9 = by_heart_matcher(dd_randCADEC_train9_lowercased, dd_randCADEC_train9_lowercased)
    print("\nCustom CADEC (5 train-> 5 validation):\n")
    dd_predictions_customCADEC0 = by_heart_matcher(dd_customCADEC_train0_lowercased, dd_customCADEC_train0_lowercased)
    dd_predictions_customCADEC1 = by_heart_matcher(dd_customCADEC_train1_lowercased, dd_customCADEC_train1_lowercased)
    dd_predictions_customCADEC2 = by_heart_matcher(dd_customCADEC_train2_lowercased, dd_customCADEC_train2_lowercased)
    dd_predictions_customCADEC3 = by_heart_matcher(dd_customCADEC_train3_lowercased, dd_customCADEC_train3_lowercased)
    dd_predictions_customCADEC4 = by_heart_matcher(dd_customCADEC_train4_lowercased, dd_customCADEC_train4_lowercased)
    print("\nBB4 (train->dev):\n")
    dd_predictions_BB4 = by_heart_matcher(dd_BB4habTrain_lowercased, dd_BB4habTrain_lowercased)
    print("\nNCBI (train->dev):\n")
    dd_predictions_NCBI = by_heart_matcher(dd_NCBITrainFixed_lowercased, dd_NCBITrainFixed_lowercased)
    print("\nPredicted.")


    print("\n\nExact Matching method:")

    print("\nRandom CADEC (10 validation) EM:\n")
    """
    dd_EMpredictions_CADEC0 = exact_matcher(dd_randCADEC_train0_lowercased, dd_subsubRef_lowercased)
    dd_EMpredictions_CADEC1 = exact_matcher(dd_randCADEC_train1_lowercased, dd_subsubRef_lowercased)
    dd_EMpredictions_CADEC2 = exact_matcher(dd_randCADEC_train2_lowercased, dd_subsubRef_lowercased)
    dd_EMpredictions_CADEC3 = exact_matcher(dd_randCADEC_train3_lowercased, dd_subsubRef_lowercased)
    dd_EMpredictions_CADEC4 = exact_matcher(dd_randCADEC_train4_lowercased, dd_subsubRef_lowercased)
    dd_EMpredictions_CADEC5 = exact_matcher(dd_randCADEC_train5_lowercased, dd_subsubRef_lowercased)
    dd_EMpredictions_CADEC6 = exact_matcher(dd_randCADEC_train6_lowercased, dd_subsubRef_lowercased)
    dd_EMpredictions_CADEC7 = exact_matcher(dd_randCADEC_train7_lowercased, dd_subsubRef_lowercased)
    dd_EMpredictions_CADEC8 = exact_matcher(dd_randCADEC_train8_lowercased, dd_subsubRef_lowercased)
    dd_EMpredictions_CADEC9 = exact_matcher(dd_randCADEC_train9_lowercased, dd_subsubRef_lowercased)
    """
    print("Custom CADEC (5 test) EM:\n")
    dd_EMpredictions_customCADEC0 = exact_matcher(dd_customCADEC_train0_lowercased, dd_subsubRef_lowercased)
    dd_EMpredictions_customCADEC1 = exact_matcher(dd_customCADEC_train1_lowercased, dd_subsubRef_lowercased)
    dd_EMpredictions_customCADEC2 = exact_matcher(dd_customCADEC_train2_lowercased, dd_subsubRef_lowercased)
    dd_EMpredictions_customCADEC3 = exact_matcher(dd_customCADEC_train3_lowercased, dd_subsubRef_lowercased)
    dd_EMpredictions_customCADEC4 = exact_matcher(dd_customCADEC_train4_lowercased, dd_subsubRef_lowercased)
    print("BB4 EM (dev):\n")
    dd_EMpredictions_BB4 = exact_matcher(dd_BB4habTrain_lowercased, dd_habObt_lowercased)
    print("NCBI (dev):\n")
    dd_EMpredictions_NCBI = exact_matcher(dd_NCBITrainFixed_lowercased, dd_medic_lowercased)
    print("\nPredicted.")


    ################################################
    print("\n\n\n\nSCORING:\n")
    ################################################
    from evaluators import accuracy

    scoreCADEC0 = accuracy(dd_predictions_CADEC0, dd_randCADEC_train0_lowercased)
    scoreCADEC1 = accuracy(dd_predictions_CADEC1, dd_randCADEC_train1_lowercased)
    scoreCADEC2 = accuracy(dd_predictions_CADEC2, dd_randCADEC_train2_lowercased)
    scoreCADEC3 = accuracy(dd_predictions_CADEC3, dd_randCADEC_train3_lowercased)
    scoreCADEC4 = accuracy(dd_predictions_CADEC4, dd_randCADEC_train4_lowercased)
    scoreCADEC5 = accuracy(dd_predictions_CADEC5, dd_randCADEC_train5_lowercased)
    scoreCADEC6 = accuracy(dd_predictions_CADEC6, dd_randCADEC_train6_lowercased)
    scoreCADEC7 = accuracy(dd_predictions_CADEC7, dd_randCADEC_train7_lowercased)
    scoreCADEC8 = accuracy(dd_predictions_CADEC8, dd_randCADEC_train8_lowercased)
    scoreCADEC9 = accuracy(dd_predictions_CADEC9, dd_randCADEC_train9_lowercased)
    scorecustomCADEC0 = accuracy(dd_predictions_customCADEC0, dd_customCADEC_train0_lowercased)
    scorecustomCADEC1 = accuracy(dd_predictions_customCADEC1, dd_customCADEC_train1_lowercased)
    scorecustomCADEC2 = accuracy(dd_predictions_customCADEC2, dd_customCADEC_train2_lowercased)
    scorecustomCADEC3 = accuracy(dd_predictions_customCADEC3, dd_customCADEC_train3_lowercased)
    scorecustomCADEC4 = accuracy(dd_predictions_customCADEC4, dd_customCADEC_train4_lowercased)
    scoreBB4 = accuracy(dd_predictions_BB4, dd_BB4habTrain_lowercased)
    scoreNCBI = accuracy(dd_predictions_NCBI, dd_NCBITrainFixed_lowercased)


    """
    EMscoreCADEC0 = accuracy(dd_EMpredictions_CADEC0, dd_randCADEC_validation0_lowercased)
    EMscoreCADEC1 = accuracy(dd_EMpredictions_CADEC1, dd_randCADEC_validation1_lowercased)
    EMscoreCADEC2 = accuracy(dd_EMpredictions_CADEC2, dd_randCADEC_validation2_lowercased)
    EMscoreCADEC3 = accuracy(dd_EMpredictions_CADEC3, dd_randCADEC_validation3_lowercased)
    EMscoreCADEC4 = accuracy(dd_EMpredictions_CADEC4, dd_randCADEC_validation4_lowercased)
    EMscoreCADEC5 = accuracy(dd_EMpredictions_CADEC5, dd_randCADEC_validation5_lowercased)
    EMscoreCADEC6 = accuracy(dd_EMpredictions_CADEC6, dd_randCADEC_validation6_lowercased)
    EMscoreCADEC7 = accuracy(dd_EMpredictions_CADEC7, dd_randCADEC_validation7_lowercased)
    EMscoreCADEC8 = accuracy(dd_EMpredictions_CADEC8, dd_randCADEC_validation8_lowercased)
    EMscoreCADEC9 = accuracy(dd_EMpredictions_CADEC9, dd_randCADEC_validation9_lowercased)
    """
    EMscorecustomCADEC0 = accuracy(dd_EMpredictions_customCADEC0, dd_customCADEC_train0_lowercased)
    EMscorecustomCADEC1 = accuracy(dd_EMpredictions_customCADEC1, dd_customCADEC_train1_lowercased)
    EMscorecustomCADEC2 = accuracy(dd_EMpredictions_customCADEC2, dd_customCADEC_train2_lowercased)
    EMscorecustomCADEC3 = accuracy(dd_EMpredictions_customCADEC3, dd_customCADEC_train3_lowercased)
    EMscorecustomCADEC4 = accuracy(dd_EMpredictions_customCADEC4, dd_customCADEC_train4_lowercased)
    EMscoreBB4 = accuracy(dd_EMpredictions_BB4, dd_BB4habTrain_lowercased)
    EMscoreNCBI = accuracy(dd_EMpredictions_NCBI, dd_NCBITrainFixed_lowercased)


    print("By heart learning method:")

    print("AccuracyBH(CADEC0)=", scoreCADEC0)
    print("AccuracyBH(CADEC1)=", scoreCADEC1)
    print("AccuracyBH(CADEC2)=", scoreCADEC2)
    print("AccuracyBH(CADEC3)=", scoreCADEC3)
    print("AccuracyBH(CADEC4)=", scoreCADEC4)
    print("AccuracyBH(CADEC5)=", scoreCADEC5)
    print("AccuracyBH(CADEC6)=", scoreCADEC6)
    print("AccuracyBH(CADEC7)=", scoreCADEC7)
    print("AccuracyBH(CADEC8)=", scoreCADEC8)
    print("AccuracyBH(CADEC9)=", scoreCADEC9)
    print("\nAccuracyBH(CADEC0)=", scorecustomCADEC0)
    print("AccuracyBH(CADEC1)=", scorecustomCADEC1)
    print("AccuracyBH(CADEC2)=", scorecustomCADEC2)
    print("AccuracyBH(CADEC3)=", scorecustomCADEC3)
    print("AccuracyBH(CADEC4)=", scorecustomCADEC4)
    print("\nAccuracyBH(BB4)=", scoreBB4)
    print("\nAccuracyBH(NCBI)=", scoreNCBI)

    print("\n\n")

    print("Exact Matching method:")
    """
    print("AccuracyEM(CADEC0)=", EMscoreCADEC0)
    print("AccuracyEM(CADEC1)=", EMscoreCADEC1)
    print("AccuracyEM(CADEC2)=", EMscoreCADEC2)
    print("AccuracyEM(CADEC3)=", EMscoreCADEC3)
    print("AccuracyEM(CADEC4)=", EMscoreCADEC4)
    print("AccuracyEM(CADEC5)=", EMscoreCADEC5)
    print("AccuracyEM(CADEC6)=", EMscoreCADEC6)
    print("AccuracyEM(CADEC7)=", EMscoreCADEC7)
    print("AccuracyEM(CADEC8)=", EMscoreCADEC8)
    print("AccuracyEM(CADEC9)=", EMscoreCADEC9)
    """
    print("\nAccuracyEM(CADEC0)=", EMscorecustomCADEC0)
    print("AccuracyEM(CADEC1)=", EMscorecustomCADEC1)
    print("AccuracyEM(CADEC2)=", EMscorecustomCADEC2)
    print("AccuracyEM(CADEC3)=", EMscorecustomCADEC3)
    print("AccuracyEM(CADEC4)=", EMscorecustomCADEC4)
    print("\nAccuracyEM(BB4)=", EMscoreBB4)
    print("\nAccuracyEM(NCBI)=", EMscoreNCBI)



