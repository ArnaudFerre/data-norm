# Author: Arnaud Ferré
# RALI, Montreal University
#
# Description :




#######################################################################################################
# Imports:
#######################################################################################################

from os import listdir, makedirs
from os.path import isfile, join, splitext, basename, exists
import re
import sys
import copy

from numpy import std, median, any, zeros, linalg
import json

from pronto import Ontology
from tensorflow.keras import layers, models, Model, Input, regularizers, optimizers, metrics, losses, initializers, backend, callbacks, activations
from scipy.spatial.distance import cosine, euclidean, cdist


from writers import write_ref
from loaders import loader_ontobiotope, get_tags_in_ref, select_subpart_hierarchy, loader_one_bb4_fold, extract_data, loader_one_ncbi_fold, loader_medic, \
    get_cuis_set_from_corpus, loader_clinical_finding_file, loader_amt, select_subpart_with_patterns_in_label, fusion_ref, get_cui_list, \
    loader_all_custom_cadec_folds, extract_data_without_file
from methods import dense_layer_method, stem_lowercase_mentions, stem_lowercase_ref, by_heart_and_exact_matching, lowercaser_mentions
from evaluators import accuracy




#######################################################################################################
# Functions:
#######################################################################################################

##################################################
# BioSyn adapters:
##################################################

def BioSyn_train_dictionary_adaptater(newFile, dd_ref):

    with open(newFile, 'w', encoding="utf8") as newFoldFile:

        for cui in dd_ref.keys():

            line = cui+'||'+dd_ref[cui]["label"].lower()+"\n"
            newFoldFile.write(line)

            if "tags" in dd_ref[cui].keys():
                for tag in dd_ref[cui]["tags"]:
                    if tag != "":
                        line = cui + '||' + tag.lower()+"\n"
                        newFoldFile.write(line)

    print("\nSaved in", newFile, ".")



def BioSyn_training_files(outputDir, ddd_data, l_selectedTypes=[]):

    for file in ddd_data.keys():
        with open(outputDir+'/'+file+".concept", 'w', encoding="utf8") as newFoldFile:

            for id in ddd_data[file].keys():

                if "type" in ddd_data[file][id].keys():
                    typeInfo = ddd_data[file][id]["type"]
                    if len(l_selectedTypes) > 0:
                        if typeInfo not in l_selectedTypes:
                            continue
                else:
                    typeInfo = "UnknownType"


                line = file + "||" + "na" + '|' + "na" + "||" + typeInfo + "||" + ddd_data[file][id]["mention"].lower() + "||"
                for i, cui in enumerate(ddd_data[file][id]["cui"]):
                    if i < (len(ddd_data[file][id]["cui"])-1):
                        line += cui + "|" # WARNING: NOT SURE THAT BioSy TAKE INTO ACCOUNT MULTI-NORM...
                    else:
                        line += cui + "\n"
                newFoldFile.write(line)


        print("\nSaved in", outputDir+'/'+file+".concept", ".")



def BioSyn_json_predictions_adapter(jsonFilePath, dd_mentionsToNormalized):
    dd_pred = dict()
    for id in dd_mentionsToNormalized.keys():
        dd_pred[id] = dict()
        dd_pred[id]["mention"] = dd_mentionsToNormalized[id]["mention"]
        dd_pred[id]["pred_cui"] = []

    with open(jsonFilePath) as f:
        ddd_BioSyn_pred = json.load(f)

    for dld_mention in ddd_BioSyn_pred["queries"]:

        if len(dld_mention["mentions"]) == 1:

            surfaceForm = dld_mention["mentions"][0]["mention"]
            pred_cui = dld_mention["mentions"][0]["candidates"][0]["cui"] #top-ranked candidate

            for id in dd_mentionsToNormalized.keys():
                if dd_mentionsToNormalized[id]["mention"].lower() == surfaceForm.strip():
                    dd_pred[id]["pred_cui"] = [pred_cui]

        else:
            print("composite mentions? meaning?")
            for truc in dld_mention["mentions"]:
                print(truc)

    return dd_pred



def find_composite_mentions(jsonFilePath, dd_dataset):
    with open(jsonFilePath) as f:
        ddd_BioSyn_pred = json.load(f)

    for dld_mention in ddd_BioSyn_pred["queries"]:

        if len(dld_mention["mentions"]) > 1:
            for partOfCompositeMention in dld_mention["mentions"]:
                print(partOfCompositeMention["mention"], "--- pred:", partOfCompositeMention["candidates"][0])
            print("\n")


###################################################
# Erase 'OMIM:" in CUIs in MEDIC:
def adapt_ref_cuis(dd_medic):

    dd_newRef = dict()

    request = re.compile('OMIM:(.*)')
    for cui in dd_medic.keys():
        m = request.match(cui)
        if m:
            newCui = m.group(1)
            dd_newRef[newCui] = copy.deepcopy(dd_medic[cui])
        else:
            dd_newRef[cui] = copy.deepcopy(dd_medic[cui])

    for cui in dd_newRef.keys():
        if "alt_cui" in dd_newRef[cui].keys():
            for i, altCui in enumerate(dd_newRef[cui]["alt_cui"]):
                m = request.match(altCui)
                if m:
                    newAltCui = m.group(1)
                    dd_newRef[cui]["alt_cui"][i] = newAltCui

    return dd_newRef


def get_all_cui_in_ref(dd_ref):
    s_cuis = set()
    for cui in dd_ref.keys():
        s_cuis.add(cui)
        if "alt_cui" in dd_ref[cui].keys():
            for altCui in dd_ref[cui]["alt_cui"]:
                s_cuis.add(altCui)
    return s_cuis


###################################################

def biosyn_data_adapter(processedDataDirPath):

    dd_dataset = dict()
    i = 0
    request = re.compile('.*\+.*')

    for fileName in listdir(processedDataDirPath):
        filePath = join(processedDataDirPath, fileName)
        with open(filePath, 'r', encoding="utf8") as file:

            for line in file:

                l_line = line.split("||")

                id = "biosyn_ncbi_"+str(i)
                dd_dataset[id] = dict()

                dd_dataset[id]["mention"] = l_line[3] # can be composite

                dd_dataset[id]["type"] = l_line[2]

                dd_dataset[id]["biosyntype"] = None # Just to memorize composite mention and multi-norm cases (and alt)

                m = request.match(l_line[4])
                if m:
                    dd_dataset[id]["mention"] = dd_dataset[id]["mention"]
                    dd_dataset[id]["cui"] = list()
                    for cui in (l_line[4].rstrip()).split('+'):
                        dd_dataset[id]["cui"].append(cui.strip())
                    dd_dataset[id]["biosyntype"] = "multinorm"

                else:
                    l_mentions = dd_dataset[id]["mention"].split('|')

                    if len(l_mentions) > 1:
                        dd_dataset[id]["mention"] = l_mentions
                        dd_dataset[id]["cui"] = list()
                        for cui in (l_line[4].rstrip()).split('|'):
                            dd_dataset[id]["cui"].append(cui.strip())
                        dd_dataset[id]["biosyntype"] = "CompositeMention"

                    else:
                        dd_dataset[id]["mention"] = l_mentions[0]
                        dd_dataset[id]["cui"] = list()
                        for cui in (l_line[4].rstrip()).split('|'):
                            dd_dataset[id]["cui"].append(cui.strip())

                        # composite not separeted by preprocessing:
                        if len(dd_dataset[id]["cui"]) > 1:
                            dd_dataset[id]["biosyntype"] = "CompositeMention"


                i+=1

    return dd_dataset



def biosyn_composite_adapter(dd_biosynData):

    dd_newData = dict()

    for id in dd_biosynData.keys():

        if len(dd_biosynData[id]["cui"]) > 1 and dd_biosynData[id]["biosyntype"] == "CompositeMention":

            if type(dd_biosynData[id]["mention"]) is list:
                if len(dd_biosynData[id]["mention"]) == len(dd_biosynData[id]["cui"]):
                    for i, mention in enumerate(dd_biosynData[id]["mention"]):
                        newId = id + '_' + str(i)
                        dd_newData[newId] = copy.deepcopy(dd_biosynData[id])
                        dd_newData[newId]["mention"] = dd_biosynData[id]["mention"][i]
                        dd_newData[newId]["cui"] = [dd_biosynData[id]["cui"][i]]

            else: # fail to segment composite mentions with the biosyn preprocessing:
                for i, cui in enumerate(dd_biosynData[id]["cui"]):
                    newId = id + '_' + str(i)
                    dd_newData[newId] = copy.deepcopy(dd_biosynData[id])
                    dd_newData[newId]["cui"] = [dd_biosynData[id]["cui"][i]]

        # 1-1 mention-cui and multi-norm cases:
        else:
            dd_newData[id] = copy.deepcopy(dd_biosynData[id])

    return dd_newData

"""
biosyn_ncbi_14_0 {'mention': 'colorectal cancers', 'type': 'CompositeMention', 'biosyntype': 'CompositeMention', 'cui': ['D010051']}
biosyn_ncbi_14_1 {'mention': 'endometrial cancers', 'type': 'CompositeMention', 'biosyntype': 'CompositeMention', 'cui': ['D016889']}
biosyn_ncbi_14_2 {'mention': 'ovarian cancers', 'type': 'CompositeMention', 'biosyntype': 'CompositeMention', 'cui': ['D015179']}
"""
def merge_composite_pred(dd_pred):
    dd_mergedPred = dict()
    request = re.compile('biosyn_ncbi_([0-9]+)_[0-9]+')

    for id in dd_pred.keys():
        m = request.match(id)
        if m:
            mergedId = 'biosyn_ncbi_'+str(m.group(1))
            dd_mergedPred[mergedId] = dict()
            dd_mergedPred[mergedId]["pred_cui"] = list()

    for id in dd_pred.keys():
        m = request.match(id)
        if m:
            mergedId = 'biosyn_ncbi_' + str(m.group(1))
            dd_mergedPred[mergedId]["pred_cui"].append(dd_pred[id]["pred_cui"][0])
        else:
            dd_mergedPred[id] = copy.deepcopy(dd_pred[id])

    return dd_mergedPred




def biosyn_ncbi_accuracy(dd_pred, dd_resp):

    totalScore = 0.0

    for id in dd_resp.keys():
        score = 0
        l_cuiPred = dd_pred[id]["pred_cui"]
        l_cuiResp = dd_resp[id]["cui"]

        if dd_resp[id]["biosyntype"] == "CompositeMention":
            compositeMentionHit = 0
            for cuiPred in l_cuiPred:
                if cuiPred in l_cuiResp:
                    compositeMentionHit += 1
            if compositeMentionHit == len(l_cuiResp):
                score = 1

        elif dd_resp[id]["biosyntype"] == "multinorm":  # biosyn evaluation (1 pt if just 1 hit)
            for cuiPred in l_cuiPred:
                if cuiPred in l_cuiResp:
                    score = 1
                    break

        else:
            soluce = l_cuiResp[0]
            pred = l_cuiPred[0]
            if pred == soluce:
                score = 1

        totalScore += score

    totalScore = totalScore / len(dd_resp.keys())

    return totalScore


###################################################

def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])


def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i + 1]  # to get acc@(i+1)
                mention_hit += any([candidate['label'] for candidate in candidates])

            # When all mentions in a query are predicted correctly,
            # we consider it as a hit
            if mention_hit == len(mentions):
                hit += 1

        data['acc{}'.format(i + 1)] = hit / len(queries)

    return data



def evaluate_top1_acc(data, dd_dataset, dd_ref):
    """
    Description: An evaluation of the real score of a BioSyn pred even with multi-norm.
    It approximative, thus, don't use for real evaluation.
    """
    queries = data['queries']
    print(len(queries), len(dd_dataset.keys()))

    hit = 0
    for i, query in enumerate(queries):
        mentions = query['mentions'] # list containing in general 1 mention, and their candidates CUIs (top20).

        mention_hit = 0
        for mention in mentions:

            candidate = mention['candidates'][0]  # to get acc@(i+1)
            if candidate['label'] == 1:

                mq = False
                surfaceForm = mentions[0]["mention"]
                pred_cui = candidate["cui"]  # top-ranked candidate
                l_maxi=list()

                for id in dd_dataset.keys():
                    if dd_dataset[id]["mention"].lower() == surfaceForm.strip():
                        l_maxi.append(len(dd_dataset[id]["cui"]))

                hit += 1.0 / ( max (l_maxi) )

            break

    data['acc{}'.format(1)] = hit / len(queries)

    return data


##################################################
# C-Norm adapters:
##################################################

def cnorm_terms_adapter(dd_dataset, outputFilePath):
    #{"mention_unique_id1": ["token11", "token12", …, "token1m"], "mention_unique_id2": ["token21", … "token2n"], … }

    dl_jsonDataset = dict()
    for id in dd_dataset.keys():
        l_tokens = dd_dataset[id]["mention"].split()
        dl_jsonDataset[id] = l_tokens

    with open(outputFilePath, 'w') as outfile:
        json.dump(dl_jsonDataset, outfile)

    print(outputFilePath, "saved.")



def cnorm_terms_and_attributions_adapter(dd_trainingDataset, outputTermsPath, outputAttributionsPath):
    #{"mention_unique_idA":["concept_identifierA1", "concept_identifierA2", …], "mention_unique_idB":["concept_identifierB1"], …}

    dl_jsonTerms = dict()
    dl_attributions = dict()
    for id in dd_trainingDataset.keys():
        l_tokens = dd_trainingDataset[id]["mention"].split()
        dl_jsonTerms[id] = l_tokens
        dl_attributions[id] = dd_trainingDataset[id]["cui"]

    with open(outputTermsPath, 'w') as outfile:
        json.dump(dl_jsonTerms, outfile)
    print(outputTermsPath, "saved.")

    with open(outputAttributionsPath, 'w') as outfile:
        json.dump(dl_attributions, outfile)
    print(outputAttributionsPath, "saved.")



##################################################
# Tools:
##################################################

def extract_subref_from_cui_set(dd_ref, s_cuiSet):
    dd_subRef = dict()

    for cui in dd_ref.keys():
        if cui in s_cuiSet:
            dd_subRef[cui] = copy.deepcopy(dd_ref[cui])
        else:
            if "alt_cui" in dd_ref[cui].keys():
                for altCui in dd_ref[cui]["alt_cui"]:
                    if altCui in s_cuiSet:
                        dd_subRef[cui] = copy.deepcopy(dd_ref[cui])
                        break

    return dd_subRef



#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':


    ####################
    # Sieve on NCBI:
    ####################

    print("\nLoading MEDIC...")
    dd_medic = loader_medic("../NCBI/CTD_diseases_DNorm_v2012_07_6.tsv")
    print("loaded. (Nb of concepts in MEDIC =", len(dd_medic.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_medic)),")")
    print("Note: Inconsistence errors in MEDIC -> MESH:D006938 = OMIM:144010 and MESH:C537710 = OMIM:153400.")


    dd_trainDevExamples = biosyn_data_adapter("../NCBI/BioSynPreprocessedVersion/processed_traindev/")
    dd_testExamples = biosyn_data_adapter("../NCBI/BioSynPreprocessedVersion/processed_test/")
    # To take into account composite mentions as separate data:
    dd_trainDevExamples2 = biosyn_composite_adapter(dd_trainDevExamples)
    dd_testExamples2 = biosyn_composite_adapter(dd_testExamples)

    dd_biosyn_medic = adapt_ref_cuis(dd_medic)
    s_altAndMainCuis = get_all_cui_in_ref(dd_biosyn_medic)

    s_cuisNCBITrainDev = get_cuis_set_from_corpus(dd_trainDevExamples2)
    s_cuisNCBITest = get_cuis_set_from_corpus(dd_testExamples2)
    dd_trainDevMedic = extract_subref_from_cui_set(dd_biosyn_medic, s_cuisNCBITrainDev)
    dd_testMedic = extract_subref_from_cui_set(dd_biosyn_medic, s_cuisNCBITest)


    from methods import sieve
    from gensim.models import KeyedVectors
    embeddings = KeyedVectors.load_word2vec_format('../PubMed-w2v.bin', binary=True)


    dd_pred = sieve(dd_badTrainDev, dd_testExamples2, dd_testMedic, embeddings)
    #dd_pred = sieve(dd_trainDevExamples2, dd_testExamples2, dd_testMedic, embeddings)
    print("dd_pred:", dd_pred)
    dd_remergedPred = merge_composite_pred(dd_pred)
    print("dd_remergedPred:", len(dd_remergedPred.keys()), dd_remergedPred)


    BioSyn_score_NCBI_trainOnTest = biosyn_ncbi_accuracy(dd_remergedPred, dd_testExamples)
    print("\n\nBioSyn_score_NCBI_trainOnTest:", BioSyn_score_NCBI_trainOnTest)




    sys.exit(0)
    """

    ####################


    # Sieve on true NCBI corpus without preprocessing:
    ddd_dataTrainDev = loader_one_ncbi_fold(["../NCBI/Voff/NCBItrainset_corpus.txt", "../NCBI/Voff/NCBIdevelopset_corpus.txt"])
    dd_TrainDev = extract_data(ddd_dataTrainDev, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in train+dev corpus =", len(dd_TrainDev.keys()), ")")
    ddd_dataTest = loader_one_ncbi_fold(["../NCBI/Voff/NCBItestset_corpus.txt"])
    dd_Test = extract_data(ddd_dataTest, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in test corpus =", len(dd_Test.keys()), ")")

    print("\nLoading MEDIC...")
    dd_medic = loader_medic("../NCBI/CTD_diseases_DNorm_v2012_07_6.tsv")
    print("loaded. (Nb of concepts in MEDIC =", len(dd_medic.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_medic)), ")")
    print("Note: Inconsistence errors in MEDIC -> MESH:D006938 = OMIM:144010 and MESH:C537710 = OMIM:153400.")


    s_cuisNCBITrainDev = get_cuis_set_from_corpus(dd_TrainDev)
    s_cuisNCBITest = get_cuis_set_from_corpus(dd_Test)
    dd_trainDevMedic = extract_subref_from_cui_set(dd_medic, s_cuisNCBITrainDev)
    dd_testMedic = extract_subref_from_cui_set(dd_medic, s_cuisNCBITest)


    from methods import sieve
    from gensim.models import KeyedVectors
    embeddings = KeyedVectors.load_word2vec_format('../PubMed-w2v.bin', binary=True)

    dd_biosynTest = dict()
    for id in dd_Test.keys():
        dd_biosynTest[id] = dict()
        dd_biosynTest[id] = copy.deepcopy(dd_Test[id])
        dd_biosynTest[id]["biosyntype"] = None
        if len(dd_biosynTest[id]["cui"]) > 1:
            dd_biosynTest[id]["biosyntype"] == "CompositeMention"


    dd_pred = sieve(dd_TrainDev, dd_Test, dd_testMedic, embeddings)
    print("dd_pred:", dd_pred)
    BioSyn_score_NCBI_trainOnTest = biosyn_ncbi_accuracy(dd_pred, dd_biosynTest)
    print("\n\nBioSyn_score_NCBI_trainOnTest:", BioSyn_score_NCBI_trainOnTest)

    """

    ####################
    # CNorm:
    ####################
    """
    print("\n\nBB4:")

    print("loading OntoBiotope...")
    dd_obt = loader_ontobiotope("../BB4/OntoBiotope_BioNLP-OST-2019.obo")
    print("loaded. (Nb of concepts in SCT =", len(dd_obt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_obt)), ")")
    print("\nExtracting Bacterial Habitat hierarchy:")
    dd_habObt = select_subpart_hierarchy(dd_obt, 'OBT:000001')
    print("Done. (Nb of concepts in this subpart of OBT =", len(dd_habObt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_habObt)), ")")

    print("\nLoading BB4 dev corpora...")
    ddd_dataTrain = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_train"])
    dd_habTrain = extract_data(ddd_dataTrain, l_type=["Habitat"])  # ["Habitat", "Phenotype", "Microorganism"]
    print("loaded.(Nb of mentions in train =", len(dd_habTrain.keys()), ")")
    ddd_dataDev = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_dev"])
    dd_habDev = extract_data(ddd_dataDev, l_type=["Habitat"])
    print("loaded.(Nb of mentions in dev =", len(dd_habDev.keys()), ")")
    ddd_dataTrainDev = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_train", "../BB4/BioNLP-OST-2019_BB-norm_dev"])
    dd_habTrainDev = extract_data(ddd_dataTrainDev, l_type=["Habitat"])
    print("loaded.(Nb of mentions in train+dev =", len(dd_habTrainDev.keys()), ")")

    cnorm_terms_and_attributions_adapter(dd_habTrain, "trainingMentions.json", "trainingAttributions.json")
    cnorm_terms_and_attributions_adapter(dd_habTrain, "devMentions.json", "devAttributions.json")

    sys.exit(0)

    """


    ####################
    # BioSyn:
    ####################


    print("\n\nCADEC:")
    """

    print("Loading of Clinical finding from SCT-AU...")
    dd_subSct = loader_clinical_finding_file("../CADEC/clinicalFindingSubPart.csv")
    print('Loaded.')

    print("loading AMTv2.56...")
    dd_amt = loader_amt("../CADEC/AMT_v2.56/Uuid_sct_concepts_au.gov.nehta.amt.standalone_2.56.txt")
    dd_subAmt = select_subpart_with_patterns_in_label(dd_amt)
    print("Done. (Nb of concepts in this subpart AMT =", len(dd_subAmt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_subAmt)), ")")

    print("\nFusion subSCT & subAMT in one reference...")
    dd_subsubRef = fusion_ref(dd_subSct, dd_subAmt)
    print("done. (Nb of concepts in subSCT+subAMT =", len(dd_subsubRef.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_subsubRef)), ")")

    print("\nLoading CUIs list used by [Miftahutdinov et al. 2019]...")
    l_sctFromCadec = get_cui_list("../CADEC/custom_CUI_list.txt")
    print("loaded.(Nb of concepts in the list (with CONCEPT_LESS =", len(l_sctFromCadec),")")


    print("\nLoading custom CADEC corpus...")
    ddd_customData = loader_all_custom_cadec_folds("../CADEC/2_Custom_folds/")
    dd_customCadec = extract_data_without_file(ddd_customData)
    print("loaded.(Nb of mentions in ALL folds for custom CADEC =", len(dd_customCadec.keys()),")")


    print("\n\nLoading cuis set in corpus...")
    s_cuisInCustomCadec = get_cuis_set_from_corpus(dd_customCadec)
    print("Loaded.(Nb of distinct used concepts in custom", len(s_cuisInCustomCadec),")")

    """
    """
    # --train_dictionary_path

    dd_customSctAmt = extract_subref_from_cui_set(dd_subsubRef, s_cuisInCustomCadec) # 181 CUIs in all folds

    BioSyn_train_dictionary_adaptater("sctAmt_custom_dictionary.txt", dd_customSctAmt)


    # --train_dir
    BioSyn_training_files("cadec_processed_train", ddd_customData, l_selectedTypes=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    """

    """
    # Rescoring:
    dd_pred0 = BioSyn_json_predictions_adapter("predictions_eval_trainOnTest_cadec0.json", ddd_customData["test_0"])
    BioSyn_score_cadec_trainOnTest = accuracy(dd_pred0, ddd_customData["test_0"])
    print("\n\nBioSyn_score_cadec_trainOnTest:", BioSyn_score_cadec_trainOnTest)
    """

    """
    for i in range(4):
        repertoireTrain = "cadec_processed_train_"+str(i)
        repertoireTest = "cadec_processed_test_"+str(i)
        if not exists(repertoireTrain):
            makedirs(repertoireTrain)
        if not exists(repertoireTest):
            makedirs(repertoireTest)



        BioSyn_training_files(repertoireTrain, ddd_customData["train_"+str(i)], l_selectedTypes=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
        BioSyn_training_files(repertoireTest, ddd_customData["test_"+str(i)], l_selectedTypes=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    """



    print("\n\nInitial NCBI Disease Corpus:\n")


    print("\nLoading MEDIC...")
    dd_medic = loader_medic("../NCBI/CTD_diseases_DNorm_v2012_07_6.tsv")
    print("loaded. (Nb of concepts in MEDIC =", len(dd_medic.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_medic)),")")
    print("Note: Inconsistence errors in MEDIC -> MESH:D006938 = OMIM:144010 and MESH:C537710 = OMIM:153400.")


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
    print("Loaded.(Nb of distinct used concepts in Full/train/dev/train+dev/test NCBI folds =", len(s_cuisNCBIFull), len(s_cuisNCBITrain), len(s_cuisNCBIDev), len(s_cuisNCBITrainDev), len(s_cuisNCBITest), ")")


    """
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

    ddd_dataTrainDevFixed = loader_one_ncbi_fold( ["../NCBI/FixedVersion/NCBItrainset_corpus_fixed.txt", "../NCBI/FixedVersion/NCBIdevelopset_corpus_fixed.txt"])
    dd_TrainDevFixed = extract_data(ddd_dataTrainDevFixed, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in train+dev corpus =", len(dd_TrainDevFixed.keys()), ")")

    ddd_dataTestFixed = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItestset_corpus_fixed.txt"])
    dd_TestFixed = extract_data(ddd_dataTestFixed, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in test corpus =", len(dd_TestFixed.keys()), ")")

    print("\nLoading cuis set in corpus...")
    s_cuisNCBIFull = get_cuis_set_from_corpus(dd_FullFixed)
    s_cuisNCBITrain = get_cuis_set_from_corpus(dd_TrainFixed)
    s_cuisNCBIDev = get_cuis_set_from_corpus(dd_DevFixed)
    s_cuisNCBITrainDev = get_cuis_set_from_corpus(dd_TrainDevFixed)
    s_cuisNCBITest = get_cuis_set_from_corpus(dd_TestFixed)
    print("Loaded.(Nb of distinct used concepts in Full/train/dev/train+dev/test NCBI folds =", len(s_cuisNCBIFull),len(s_cuisNCBITrain),len(s_cuisNCBIDev),len(s_cuisNCBITrainDev),len(s_cuisNCBITest),")")
    """

    nbMulti=0
    nbComposite=0
    for id in dd_Full.keys():
        if len(dd_Full[id]["cui"]) > 1:
            if dd_Full[id]["type"] != 'CompositeMention':
                nbMulti+=1
            else:
                nbComposite+=1
    print("(Nb of multi-normalized in NCBI corpus:", nbMulti, nbComposite, ")\n")


    # --train_dictionary_path
    dd_trainDevMedic = extract_subref_from_cui_set(dd_medic, s_cuisNCBITrainDev)
    dd_testMedic = extract_subref_from_cui_set(dd_medic, s_cuisNCBITest)

    BioSyn_train_dictionary_adaptater("medic_trainDev_dictionary.txt", dd_trainDevMedic)
    BioSyn_train_dictionary_adaptater("medic_test_dictionary.txt", dd_testMedic)
    BioSyn_train_dictionary_adaptater("medic_full_dictionary.txt", dd_medic)


    # --train_dir
    BioSyn_training_files("my_processed_trainDev", ddd_dataTrainDev, l_selectedTypes=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    BioSyn_training_files("my_processed_test", ddd_dataTest, l_selectedTypes=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])


    sys.exit(0)


    dd_pred = BioSyn_json_predictions_adapter("predictions_eval_trainDevOnTest_fullMedic.json", dd_TestFixed)

    BioSyn_score_BB4_trainOnTest = accuracy(dd_pred, dd_TestFixed)
    print("\n\nBioSyn_score_BB4_trainOnTest:", BioSyn_score_BB4_trainOnTest)



    sys.exit(0)

    print("\n\nBB4:")


    print("loading OntoBiotope...")
    dd_obt = loader_ontobiotope("../BB4/OntoBiotope_BioNLP-OST-2019.obo")
    print("loaded. (Nb of concepts in SCT =", len(dd_obt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_obt)), ")")
    print("\nExtracting Bacterial Habitat hierarchy:")
    dd_habObt = select_subpart_hierarchy(dd_obt, 'OBT:000001')
    print("Done. (Nb of concepts in this subpart of OBT =", len(dd_habObt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_habObt)), ")")


    print("\nLoading BB4 dev corpora...")
    ddd_dataDev = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_dev"])
    dd_habDev = extract_data(ddd_dataDev, l_type=["Habitat"])
    print("loaded.(Nb of mentions in dev =", len(dd_habDev.keys()), ")")
    """
    BioSyn_training_files("processed_dev", ddd_dataDev, l_selectedTypes=["Habitat"])

    print("\nLoading BB4 train corpora...")
    ddd_dataAll = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_train"])
    dd_habAll = extract_data(ddd_dataAll, l_type=["Habitat"])
    print("loaded.(Nb of mentions in whole corpus =", len(dd_habAll.keys()), ")")
    BioSyn_training_files("processed_train", ddd_dataAll, l_selectedTypes=["Habitat"])
    """

    print("\nLoading cuis set in corpus...")
    #s_cuisHabTrain = get_cuis_set_from_corpus(dd_habTrain)
    s_cuisHabDev = get_cuis_set_from_corpus(dd_habDev)
    #s_cuisHabTrainDev = get_cuis_set_from_corpus(dd_habTrainDev)
    #print("Loaded.(Nb of distinct used concepts in train/dev/train+dev hab corpora =", len(s_cuisHabTrain),len(s_cuisHabDev),len(s_cuisHabTrainDev),")")

    dd_habDevObt = extract_subref_from_cui_set(dd_habObt, s_cuisHabDev)
    BioSyn_train_dictionary_adaptater("obt_dev_dictionary.txt", dd_habDevObt)


    dd_pred = BioSyn_json_predictions_adapter("predictions_eval_trainOnDev_devObt.json", dd_habDev)

    BioSyn_score_BB4_trainOnDev = accuracy(dd_pred, dd_habDev)
    print("\n\nBioSyn_score_BB4_trainOnDev:", BioSyn_score_BB4_trainOnDev)




    sys.exit(0)


    """



    print("\n\nNCBI:")

    print("\nLoading Fixed NCBI corpora...")
    ddd_dataTestFixed = loader_one_ncbi_fold(["../NCBI/FixedVersion/NCBItestset_corpus_fixed.txt"])
    dd_TestFixed = extract_data(ddd_dataTestFixed, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])
    print("loaded.(Nb of mentions in test corpus =", len(dd_TestFixed.keys()), ")")

    dd_pred = BioSyn_json_predictions_adapter("predictions_eval.json", dd_TestFixed)
    BioSyn_score_NCBI_trainDevOnTest = accuracy(dd_pred, dd_TestFixed)
    print("\n\nBioSyn_score_NCBI_trainDevOnTest:", BioSyn_score_NCBI_trainDevOnTest)

    ### score problem

    with open("predictions_eval.json", 'r') as file:
        data = json.load(file)
    result = evaluate_topk_acc(data)
    for key in result.keys():
        if key!="queries":
            print(key, result[key])


    ### check dif
    queries = data['queries']
    for k, id in enumerate(dd_pred.keys()):
        surfaceForm = dd_pred[id]["mention"].lower()
        print(k, surfaceForm)

        for i, query in enumerate(queries):
            mentions = query['mentions']  # list containing in general 1 mention, and their candidates CUIs (top20).
            for j, mention in enumerate(mentions):
                if mention["mention"] == surfaceForm:

                    print("\tpred:", mention['candidates'][0])
                    print("\tMyPred:", dd_pred[id]["pred_cui"])



    """