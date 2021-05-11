# Author: Arnaud Ferré
# RALI, Montreal University
#
# Description :



#######################################################################################################
# Imports:
#######################################################################################################


from nltk.stem import WordNetLemmatizer, PorterStemmer
from tensorflow.keras import layers, models, Model, Input, regularizers, optimizers, metrics, losses, initializers, backend, callbacks, activations
from gensim.models import KeyedVectors, Word2Vec
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean, cdist

import sys
import numpy
import copy
import json

from loaders import loader_clinical_finding_file, loader_amt, select_subpart_with_patterns_in_label, get_tags_in_ref, fusion_ref, get_cui_list
from loaders import extract_data_without_file, loader_all_initial_cadec_folds, loader_all_random_cadec_folds, loader_all_custom_cadec_folds
from loaders import loader_ontobiotope, select_subpart_hierarchy, loader_one_bb4_fold
from loaders import loader_medic, loader_one_ncbi_fold, extract_data
from evaluators import accuracy




#######################################################################################################
# Functions:
#######################################################################################################

def optimized_by_heart_matcher(dd_mentions, dd_lesson):
    dd_predictions = dict()

    dd_rogueLesson = dict()
    for id in dd_lesson.keys():
        dd_rogueLesson[dd_lesson[id]["mention"]] = dict()
    for id in dd_lesson.keys():
        for cui in dd_lesson[id]["cui"]:
            if cui in dd_rogueLesson[dd_lesson[id]["mention"]].keys():
                dd_rogueLesson[dd_lesson[id]["mention"]][cui] += 1
            else:
                dd_rogueLesson[dd_lesson[id]["mention"]][cui] = 0

    for id in dd_mentions.keys():
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []
    for i, id in enumerate(dd_mentions.keys()):
        mention = dd_mentions[id]["mention"]
        if mention in dd_rogueLesson.keys():
            dd_predictions[id]["mention"] = mention
            compt = 0
            for possibleCui in dd_rogueLesson[mention].keys():
                if dd_rogueLesson[mention][possibleCui] > compt:
                    compt = dd_rogueLesson[mention][possibleCui]
            for possibleCui in dd_rogueLesson[mention].keys():
                if dd_rogueLesson[mention][possibleCui] == compt:
                    dd_predictions[id]["pred_cui"] = [possibleCui]
                    break

    return dd_predictions


##################################################

def optimized_exact_matcher(dd_mentions, dd_ref):
    dd_predictions = dict()

    dd_rogueRef = dict()
    for cui in dd_ref.keys():
        dd_rogueRef[dd_ref[cui]["label"]] = cui
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                dd_rogueRef[tag] = cui

    for id in dd_mentions.keys():
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []
    for i, id in enumerate(dd_mentions.keys()):
        mention = dd_mentions[id]["mention"]
        if mention in dd_rogueRef.keys():
            dd_predictions[id]["mention"] = dd_mentions[id]["mention"]
            dd_predictions[id]["pred_cui"].append(dd_rogueRef[mention])
            dd_predictions[id]["label"] = dd_ref[cui]["label"]

    return dd_predictions


##################################################

def by_heart_and_exact_matching(dd_mentions, dd_lesson, dd_ref):
    dd_predictions = dict()

    dd_allSurfaceForms = dict()

    for cui in dd_ref.keys():
        dd_allSurfaceForms[dd_ref[cui]["label"]] = dict()
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                dd_allSurfaceForms[tag] = dict()

    for cui in dd_ref.keys():
        if cui in dd_allSurfaceForms[dd_ref[cui]["label"]].keys():
            dd_allSurfaceForms[dd_ref[cui]["label"]][cui] += 1
        else:
            dd_allSurfaceForms[dd_ref[cui]["label"]][cui] = 0
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                if cui in dd_allSurfaceForms[tag].keys():
                    dd_allSurfaceForms[tag][cui] += 1
                else:
                    dd_allSurfaceForms[tag][cui] = 0


    for id in dd_lesson.keys():
        dd_allSurfaceForms[dd_lesson[id]["mention"]] = dict()

    for id in dd_lesson.keys():
        for cui in dd_lesson[id]["cui"]:
            if cui in dd_allSurfaceForms[dd_lesson[id]["mention"]].keys():
                dd_allSurfaceForms[dd_lesson[id]["mention"]][cui] += 1
            else:
                dd_allSurfaceForms[dd_lesson[id]["mention"]][cui] = 0


    for id in dd_mentions.keys():
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []
    for i, id in enumerate(dd_mentions.keys()):
        mention = dd_mentions[id]["mention"]
        if mention in dd_allSurfaceForms.keys():
            dd_predictions[id]["mention"] = mention
            compt = 0
            for possibleCui in dd_allSurfaceForms[mention].keys():
                if dd_allSurfaceForms[mention][possibleCui] > compt:
                    compt = dd_allSurfaceForms[mention][possibleCui]
            for possibleCui in dd_allSurfaceForms[mention].keys():
                if dd_allSurfaceForms[mention][possibleCui] == compt:
                    dd_predictions[id]["pred_cui"] = [possibleCui]
                    break


    return dd_predictions


##################################################


# A static method with embeddings
def embeddings_similarity_method_with_tags(dd_mentions, dd_ref, embeddings):

    dd_predictions = dict()
    for id in dd_mentions.keys():
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []

    vocabSize = len(embeddings.wv.vocab)
    sizeVST = embeddings.wv.vector_size
    X_train = numpy.zeros((len(dd_mentions.keys()), sizeVST))
    print("vocabSize:", vocabSize, "sizeVST:", sizeVST)


    s_knownTokensInMentions = set()
    d_mentionVectors = dict()
    dd_score = dict()
    for i, id in enumerate(dd_mentions.keys()):
        d_mentionVectors[id] = numpy.zeros(sizeVST)
        l_tokens = dd_mentions[id]["mention"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                d_mentionVectors[id] += ( embeddings[token] / numpy.linalg.norm(embeddings[token]) )
                s_knownTokensInMentions.add(token)
                dd_score[id] = dict()
        d_mentionVectors[id] = d_mentionVectors[id] / len(l_tokens)
        X_train[i] = d_mentionVectors[id]


    dd_conceptVectors = dict()
    nbLabtags = 0
    for cui in dd_ref.keys():
        dd_conceptVectors[cui] = dict()
        dd_conceptVectors[cui][dd_ref[cui]["label"]] = numpy.zeros(sizeVST)
        nbLabtags+=1
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                nbLabtags += 1
                dd_conceptVectors[cui][tag] = numpy.zeros(sizeVST)

    for cui in dd_ref.keys():
        l_tokens = dd_ref[cui]["label"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                dd_conceptVectors[cui][dd_ref[cui]["label"]] += ( embeddings[token] / numpy.linalg.norm(embeddings[token]) )

        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                l_currentTagTokens = tag.split()
                for currentToken in l_currentTagTokens:
                    if currentToken in embeddings.wv.vocab:
                        dd_conceptVectors[cui][tag] += ( embeddings[currentToken] / numpy.linalg.norm(embeddings[currentToken]) )
    del embeddings


    # Nearest neighbours calculation:
    labtagsVectorMatrix = numpy.zeros((nbLabtags, sizeVST))
    i = 0
    for cui in dd_conceptVectors.keys():
        for labtag in dd_conceptVectors[cui].keys():
            labtagsVectorMatrix[i] = dd_conceptVectors[cui][labtag]
            i += 1
    print('\nMatrix of distance calculation...')
    scoreMatrix = cdist(X_train, labtagsVectorMatrix, 'cosine')
    for i, id in enumerate(dd_mentions.keys()):
        j = -1
        for cui in dd_conceptVectors.keys():
            for labtag in dd_conceptVectors[cui].keys():
                j += 1
                scoreMatrix[i][j] = 1 - scoreMatrix[i][j]
    print("Done.\n")

    print("Generate predictions...")
    for i, id in enumerate(dd_mentions.keys()):
        maximumScore = max(scoreMatrix[i])
        j = -1
        stopSearch = False
        for cui in dd_conceptVectors.keys():
            if stopSearch == True:
                break
            for labtag in dd_conceptVectors[cui].keys():
                j += 1
                if scoreMatrix[i][j] == maximumScore:
                    dd_predictions[id]["pred_cui"] = [cui]
                    stopSearch = True
                    break
    del dd_conceptVectors
    print("Done.")

    # Supprimer les mentions avec vecteur nul pour l'entrainement...?
    # ToDo: Rendre la methode capable de ne faire que le train, que la pred, ou les 2.

    return dd_predictions


##################################################


# Embeddings+ML but take less RAM with big ref than WordCNN models:
def dense_layer_method(dd_train, dd_mentions, dd_ref, embeddings, dd_subRef=None):

    TFmodel = dense_layer_method_training(dd_train, dd_ref, embeddings)
    if dd_subRef is not None:
        dd_predictions = dense_layer_method_predicting(TFmodel, dd_mentions, dd_subRef, embeddings)
    else:
        dd_predictions = dense_layer_method_predicting(TFmodel, dd_mentions, dd_ref, embeddings)

    return dd_predictions



# Embeddings+ML but take less RAM with big ref than WordCNN models:
def dense_layer_method_training(dd_train, dd_ref, embeddings, savePath=None):

    nbMentions = len(dd_train.keys())
    vocabSize = len(embeddings.wv.vocab)
    sizeVST = embeddings.wv.vector_size
    sizeVSO = len(dd_ref.keys())
    print("vocabSize:", vocabSize, "sizeVST:", sizeVST, "sizeVSO:", sizeVSO)

    # Built mention embeddings from trainset:
    d_mentionVectors = dict()
    for id in dd_train.keys():
        d_mentionVectors[id] = numpy.zeros(sizeVST)
        l_tokens = dd_train[id]["mention"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                d_mentionVectors[id] += (embeddings[token] / numpy.linalg.norm(embeddings[token]))
        d_mentionVectors[id] = d_mentionVectors[id] / len(l_tokens)

    # Build labels/tags embeddings from ref:
    nbLabtags = 0
    dd_conceptVectors = dict()
    for cui in dd_ref.keys():
        dd_conceptVectors[cui] = dict()
        dd_conceptVectors[cui][dd_ref[cui]["label"]] = numpy.zeros(sizeVST)
        nbLabtags+=1
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                nbLabtags += 1
                dd_conceptVectors[cui][tag] = numpy.zeros(sizeVST)
    for cui in dd_ref.keys():
        l_tokens = dd_ref[cui]["label"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                dd_conceptVectors[cui][dd_ref[cui]["label"]] += (embeddings[token] / numpy.linalg.norm(embeddings[token]))
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                l_currentTagTokens = tag.split()
                for currentToken in l_currentTagTokens:
                    if currentToken in embeddings.wv.vocab:
                        dd_conceptVectors[cui][tag] += (embeddings[currentToken] / numpy.linalg.norm(embeddings[currentToken]))


    # Build training matrix:
    X_train = numpy.zeros((nbMentions, sizeVST))
    Y_train = numpy.zeros((nbMentions, sizeVST))
    for i, id in enumerate(dd_train.keys()):
        toPredCui = dd_train[id]["cui"][0]
        X_train[i] = d_mentionVectors[id]
        if toPredCui in dd_conceptVectors.keys():
            for j, tag in enumerate(dd_conceptVectors[toPredCui].keys()):
                if dd_conceptVectors[toPredCui][tag].any() and j<(len(dd_conceptVectors[toPredCui].keys())-1):
                    Y_train[i] = dd_conceptVectors[toPredCui][tag]
                    break  # Just taking the first label
                elif j==(len(dd_conceptVectors[toPredCui].keys())-1): # elif, taking the last (null or not)
                    Y_train[i] = dd_conceptVectors[toPredCui][tag]
    del d_mentionVectors


    # Build neural architecture:
    inputLayer = Input(shape=(sizeVST))
    denseLayer = (layers.Dense(sizeVST, activation=None, kernel_initializer=initializers.Identity()))(inputLayer)
    CNNmodel = Model(inputs=inputLayer, outputs=denseLayer)
    CNNmodel.summary()
    CNNmodel.compile(optimizer=optimizers.Nadam(), loss=losses.CosineSimilarity(), metrics=['cosine_similarity', 'logcosh'])

    # Training
    callback = callbacks.EarlyStopping(monitor='logcosh', patience=5, min_delta=0.0001)
    history = CNNmodel.fit(X_train, Y_train, epochs=200, batch_size=64, callbacks=[callback], verbose=0)
    # plt.plot(history.history['logcosh'], label='logcosh')
    # plt.show()
    print("\nTraining done.\n")

    # Saving model:
    if savePath is not None:
        CNNmodel.save(savePath)
        print("TF model saved as .h5 at:", savePath)

    return CNNmodel



# Embeddings+ML but take less RAM with big ref than WordCNN models:
def dense_layer_method_training_with_tags(dd_train, dd_ref, embeddings, savePath=None):
    # ToDo: Delete examples with null vector.

    nbMentions = len(dd_train.keys())
    print("Nb of mentions in the train:", nbMentions)
    vocabSize = len(embeddings.wv.vocab)
    sizeVST = embeddings.wv.vector_size
    sizeVSO = len(dd_ref.keys())
    print("vocabSize:", vocabSize, "sizeVST:", sizeVST, "sizeVSO:", sizeVSO)

    # Built mention embeddings from trainset:
    d_mentionVectors = dict()
    for id in dd_train.keys():
        d_mentionVectors[id] = numpy.zeros(sizeVST)
        l_tokens = dd_train[id]["mention"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                d_mentionVectors[id] += (embeddings[token] / numpy.linalg.norm(embeddings[token]))
        d_mentionVectors[id] = d_mentionVectors[id] / len(l_tokens)


    # Build labels/tags embeddings from ref:
    nbLabtags = 0
    dd_conceptVectors = dict()
    for cui in dd_ref.keys():
        dd_conceptVectors[cui] = dict()
        dd_conceptVectors[cui][dd_ref[cui]["label"]] = numpy.zeros(sizeVST)
        nbLabtags+=1
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                nbLabtags += 1
                dd_conceptVectors[cui][tag] = numpy.zeros(sizeVST)
    for cui in dd_ref.keys():
        l_tokens = dd_ref[cui]["label"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                dd_conceptVectors[cui][dd_ref[cui]["label"]] += (embeddings[token] / numpy.linalg.norm(embeddings[token]))
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                l_currentTagTokens = tag.split()
                for currentToken in l_currentTagTokens:
                    if currentToken in embeddings.wv.vocab:
                        dd_conceptVectors[cui][tag] += (embeddings[currentToken] / numpy.linalg.norm(embeddings[currentToken]))


    # Build training matrix:
    X_train = numpy.zeros((nbMentions, sizeVST))
    Y_train = numpy.zeros((nbMentions, sizeVST))
    for i, id in enumerate(dd_train.keys()):
        toPredCui = dd_train[id]["cui"][0]
        X_train[i] = d_mentionVectors[id]
        if toPredCui in dd_conceptVectors.keys():
            for j, tag in enumerate(dd_conceptVectors[toPredCui].keys()):
                if dd_conceptVectors[toPredCui][tag].any() and j<(len(dd_conceptVectors[toPredCui].keys())-1):
                    Y_train[i] = dd_conceptVectors[toPredCui][tag]
                    break  # Just taking the first label
                elif j==(len(dd_conceptVectors[toPredCui].keys())-1): # elif, taking the last (null or not)
                    Y_train[i] = dd_conceptVectors[toPredCui][tag]
    del d_mentionVectors


    # Build neural architecture:
    inputLayer = Input(shape=(sizeVST))
    denseLayer = (layers.Dense(sizeVST, activation=None, kernel_initializer=initializers.Identity()))(inputLayer)
    CNNmodel = Model(inputs=inputLayer, outputs=denseLayer)
    CNNmodel.summary()
    CNNmodel.compile(optimizer=optimizers.Nadam(), loss=losses.CosineSimilarity(), metrics=['cosine_similarity', 'logcosh'])

    # Training
    callback = callbacks.EarlyStopping(monitor='logcosh', patience=5, min_delta=0.0001)
    history = CNNmodel.fit(X_train, Y_train, epochs=200, batch_size=64, callbacks=[callback], verbose=0)
    # plt.plot(history.history['logcosh'], label='logcosh')
    # plt.show()
    print("\nTraining done.\n")

    # Saving model:
    if savePath is not None:
        CNNmodel.save(savePath)
        print("TF model saved as .h5 at:", savePath)

    return CNNmodel



# Embeddings+ML but take less RAM with big ref than WordCNN models:
def dense_layer_method_training_with_nearest_tags(dd_train, dd_ref, embeddings, savePath=None):
    # ToDo: Delete examples with null vector.

    nbMentions = len(dd_train.keys())
    print("Nb of mentions in the train:", nbMentions)
    vocabSize = len(embeddings.wv.vocab)
    sizeVST = embeddings.wv.vector_size
    sizeVSO = len(dd_ref.keys())
    print("vocabSize:", vocabSize, "sizeVST:", sizeVST, "sizeVSO:", sizeVSO)

    # Built mention embeddings from trainset:
    d_mentionVectors = dict()
    for id in dd_train.keys():
        d_mentionVectors[id] = numpy.zeros(sizeVST)
        l_tokens = dd_train[id]["mention"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                d_mentionVectors[id] += (embeddings[token] / numpy.linalg.norm(embeddings[token]))
        d_mentionVectors[id] = d_mentionVectors[id] / len(l_tokens)


    # Build labels/tags embeddings from ref:
    nbLabtags = 0
    dd_conceptVectors = dict()
    for cui in dd_ref.keys():
        dd_conceptVectors[cui] = dict()
        dd_conceptVectors[cui][dd_ref[cui]["label"]] = numpy.zeros(sizeVST)
        nbLabtags+=1
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                nbLabtags += 1
                dd_conceptVectors[cui][tag] = numpy.zeros(sizeVST)
    for cui in dd_ref.keys():
        l_tokens = dd_ref[cui]["label"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                dd_conceptVectors[cui][dd_ref[cui]["label"]] += (embeddings[token] / numpy.linalg.norm(embeddings[token]))
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                l_currentTagTokens = tag.split()
                for currentToken in l_currentTagTokens:
                    if currentToken in embeddings.wv.vocab:
                        dd_conceptVectors[cui][tag] += (embeddings[currentToken] / numpy.linalg.norm(embeddings[currentToken]))


    # Build training matrix:
    X_train = numpy.zeros((nbMentions, sizeVST))
    Y_train = numpy.zeros((nbMentions, sizeVST))
    for i, id in enumerate(dd_train.keys()):
        toPredCui = dd_train[id]["cui"][0]
        X_train[i] = d_mentionVectors[id]
        if toPredCui in dd_conceptVectors.keys():
            l_dist = list()
            for labtag in dd_conceptVectors[toPredCui].keys():
                if X_train[i].any() and dd_conceptVectors[toPredCui][labtag].any():
                    dist = 1 - cosine(X_train[i], dd_conceptVectors[toPredCui][labtag])
                else:
                    dist = 0.0
                l_dist.append(dist)
            nearestIndice = numpy.argmax(l_dist)
            for j, labtag in enumerate(dd_conceptVectors[toPredCui].keys()):
                if j == nearestIndice:
                    Y_train[i] = dd_conceptVectors[toPredCui][labtag]
                    break
    del d_mentionVectors


    # Build neural architecture:
    inputLayer = Input(shape=(sizeVST))
    denseLayer = (layers.Dense(sizeVST, activation=None, kernel_initializer=initializers.Identity()))(inputLayer)
    CNNmodel = Model(inputs=inputLayer, outputs=denseLayer)
    CNNmodel.summary()
    CNNmodel.compile(optimizer=optimizers.Nadam(), loss=losses.CosineSimilarity(), metrics=['cosine_similarity', 'logcosh'])

    # Training
    callback = callbacks.EarlyStopping(monitor='logcosh', patience=5, min_delta=0.0001)
    history = CNNmodel.fit(X_train, Y_train, epochs=200, batch_size=64, callbacks=[callback], verbose=0)
    # plt.plot(history.history['logcosh'], label='logcosh')
    # plt.show()
    print("\nTraining done.\n")

    # Saving model:
    if savePath is not None:
        CNNmodel.save(savePath)
        print("TF model saved as .h5 at:", savePath)

    return CNNmodel





def dense_layer_method_predicting(model, dd_mentions, dd_ref, embeddings, savePredPath=None):

    dd_predictions = dict()
    for id in dd_mentions.keys():
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []

    vocabSize = len(embeddings.wv.vocab)
    sizeVST = embeddings.wv.vector_size
    sizeVSO = len(dd_ref.keys())
    print("vocabSize:", vocabSize, "sizeVST:", sizeVST, "sizeVSO:", sizeVSO)

    # Mentions embeddings:
    print("Mentions embeddings...")
    d_mentionVectors = dict()
    for id in dd_mentions.keys():
        d_mentionVectors[id] = numpy.zeros(sizeVST)
        l_tokens = dd_mentions[id]["mention"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                d_mentionVectors[id] += (embeddings[token] / numpy.linalg.norm(embeddings[token]))
        d_mentionVectors[id] = d_mentionVectors[id] / len(l_tokens)
    print("Done.\n")

    # Prediction:
    print("Prediction...")
    Y_pred = numpy.zeros((len(dd_mentions.keys()), sizeVST))
    for i, id in enumerate(dd_mentions.keys()):
        x_test = numpy.zeros((1, sizeVST))
        x_test[0] = d_mentionVectors[id]
        Y_pred[i] = model.predict(x_test)[0]
    del d_mentionVectors
    print("Done.\n")


    # Calculating vector for (possibly new) reference:
    # Build labels/tags embeddings from ref:
    nbLabtags = 0
    dd_conceptVectors = dict()
    for cui in dd_ref.keys():
        dd_conceptVectors[cui] = dict()
        dd_conceptVectors[cui][dd_ref[cui]["label"]] = numpy.zeros(sizeVST)
        nbLabtags += 1
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                nbLabtags += 1
                dd_conceptVectors[cui][tag] = numpy.zeros(sizeVST)
    for cui in dd_ref.keys():
        l_tokens = dd_ref[cui]["label"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                dd_conceptVectors[cui][dd_ref[cui]["label"]] += (
                embeddings[token] / numpy.linalg.norm(embeddings[token]))
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                l_currentTagTokens = tag.split()
                for currentToken in l_currentTagTokens:
                    if currentToken in embeddings.wv.vocab:
                        dd_conceptVectors[cui][tag] += (
                        embeddings[currentToken] / numpy.linalg.norm(embeddings[currentToken]))


    # Nearest neighbours calculation:
    labtagsVectorMatrix = numpy.zeros((nbLabtags, sizeVST))
    i = 0
    for cui in dd_conceptVectors.keys():
        for labtag in dd_conceptVectors[cui].keys():
            labtagsVectorMatrix[i] = dd_conceptVectors[cui][labtag]
            i += 1
    print('\nMatrix of distance calculation...')
    scoreMatrix = cdist(Y_pred, labtagsVectorMatrix, 'cosine')
    for i, id in enumerate(dd_mentions.keys()):
        j = -1
        for cui in dd_conceptVectors.keys():
            for labtag in dd_conceptVectors[cui].keys():
                j += 1
                scoreMatrix[i][j] = 1 - scoreMatrix[i][j]
    print("Done.\n")

    for i, id in enumerate(dd_mentions.keys()):
        maximumScore = max(scoreMatrix[i])
        j = -1
        stopSearch = False
        for cui in dd_conceptVectors.keys():
            if stopSearch == True:
                break
            for labtag in dd_conceptVectors[cui].keys():
                j += 1
                if scoreMatrix[i][j] == maximumScore:
                    dd_predictions[id]["pred_cui"] = [cui]
                    stopSearch = True
                    break
    del dd_conceptVectors


    if savePredPath is not None:
        with open(savePredPath, 'w') as fp:
            json.dump(dd_conceptVectors, fp)
            print("Predictions saved as .json at:", savePredPath)

    # Supprimer les mentions avec vecteur nul pour l'entrainement...?

    return dd_predictions


##################################################

def sieve(dd_train, dd_mentions, dd_trainingRef, embeddings, dd_subRef=None):
    """
    Description: Begin by by_heart_and_exact_matching(), then dense_layer_method() on mentions without predictions.
    :return:
    """

    dd_lowercasedTrain = lowercaser_mentions(dd_train)
    dd_lowercasedMentions = lowercaser_mentions(dd_mentions)
    dd_lowercasedTrainingRef = lowercaser_ref(dd_trainingRef)

    TFmodel = dense_layer_method_training_with_nearest_tags(dd_train, dd_trainingRef, embeddings) #dense_layer_method_training(dd_lowercasedTrain, dd_lowercasedTrainingRef, embeddings) #
    if dd_subRef is not None:
        dd_lowercasedSubRef = lowercaser_ref(dd_subRef)
        dd_predMLE = dense_layer_method_predicting(TFmodel, dd_lowercasedMentions, dd_lowercasedSubRef, embeddings)
    else:
        dd_predMLE = dense_layer_method_predicting(TFmodel, dd_lowercasedMentions, dd_trainingRef, embeddings)

    dd_stemLowercasedMmentions = stem_lowercase_mentions(dd_mentions)
    dd_stemLowercasedTrain = stem_lowercase_mentions(dd_train)
    dd_stemAndLowercasedTrainingRef = stem_lowercase_ref(dd_trainingRef)
    dd_predBHEM = by_heart_and_exact_matching(dd_stemLowercasedMmentions, dd_stemLowercasedTrain, dd_stemAndLowercasedTrainingRef) # Is it relevant to use a different ref here?

    dd_sievePred = dict()
    for id in dd_mentions.keys():
        dd_sievePred[id] = dict()
        dd_sievePred[id]["pred_cui"] = []

    for id in dd_mentions.keys():
        if len(dd_predBHEM[id]["pred_cui"]) > 0:
            dd_sievePred[id]["pred_cui"] = dd_predBHEM[id]["pred_cui"]
        else:
            dd_sievePred[id]["pred_cui"] = dd_predMLE[id]["pred_cui"]

    return dd_sievePred



###################################################
# Preprocessing tools:
###################################################

def lowercaser_mentions(dd_mentions):
    dd_lowercasedMentions = copy.deepcopy(dd_mentions)
    for id in dd_lowercasedMentions.keys():
        dd_lowercasedMentions[id]["mention"] = dd_mentions[id]["mention"].lower()
    return dd_lowercasedMentions

def lowercaser_ref(dd_ref):
    dd_lowercasedRef = copy.deepcopy(dd_ref)
    for cui in dd_ref.keys():
        dd_lowercasedRef[cui]["label"] = dd_ref[cui]["label"].lower()
        if "tags" in dd_ref[cui].keys():
            l_lowercasedTags = list()
            for tag in dd_ref[cui]["tags"]:
                l_lowercasedTags.append(tag.lower())
            dd_lowercasedRef[cui]["tags"] = l_lowercasedTags
    return dd_lowercasedRef


def lemmatize_lowercase_mentions(dd_mentions):
    lemmatizer = WordNetLemmatizer()
    for id in dd_mentions.keys():
        dd_mentions[id]["mention"] = lemmatizer.lemmatize(dd_mentions[id]["mention"].lower())
    return dd_mentions

def lemmatize_lowercase_ref(dd_ref):
    lemmatizer = WordNetLemmatizer()
    for cui in dd_ref.keys():
        dd_ref[cui]["label"] = lemmatizer.lemmatize(dd_ref[cui]["label"].lower())
        if "tags" in dd_ref[cui].keys():
            l_lowercasedTags = list()
            for tag in dd_ref[cui]["tags"]:
                l_lowercasedTags.append(lemmatizer.lemmatize(tag.lower()))
            dd_ref[cui]["tags"] = l_lowercasedTags
    return dd_ref


def stem_lowercase_mentions(dd_mentions):
    dd_stemLowercasedMentions = copy.deepcopy(dd_mentions)
    ps = PorterStemmer()
    for id in dd_mentions.keys():
        dd_stemLowercasedMentions[id]["mention"] = ps.stem( (dd_mentions[id]["mention"]).lower() )
    return dd_stemLowercasedMentions

def stem_lowercase_ref(dd_ref):
    dd_stemLowercasedRef = copy.deepcopy(dd_ref)
    ps = PorterStemmer()
    for cui in dd_ref.keys():
        dd_stemLowercasedRef[cui]["label"] = ps.stem(dd_ref[cui]["label"].lower())
        if "tags" in dd_ref[cui].keys():
            l_lowercasedTags = list()
            for tag in dd_ref[cui]["tags"]:
                l_lowercasedTags.append(ps.stem(tag.lower()))
                dd_stemLowercasedRef[cui]["tags"] = l_lowercasedTags
    return dd_stemLowercasedRef


def stopword_filtering_mentions(dd_mentions):
    ps = PorterStemmer()
    for id in dd_mentions.keys():
        dd_mentions[id]["mention"] = ps.stem(dd_mentions[id]["mention"].lower())


     #ToDo

    return dd_mentions


def get_vocab(l_folds=None, dd_reference=None):

    if l_folds is None and dd_reference is None:
        print("ERROR: give at least one between dd_mentions or dd_ref...")
        sys.exit(0)

    s_tokens = set()

    if l_folds is not None:
        for dd_data in l_folds:
            for id in dd_data.keys():
                l_mentionTokens = dd_data[id]["mention"].split()
                for token in l_mentionTokens:
                    s_tokens.add(token)

    if dd_reference is not None:
        for cui in dd_reference.keys():
            l_labelTokens = dd_reference[cui]["label"].split()
            for token in l_labelTokens:
                s_tokens.add(token)
            if "tags" in dd_reference[cui].keys():
                for tag in dd_reference[cui]["tags"]:
                    l_tagTokens = tag.split()
                    for token in l_tagTokens:
                        s_tokens.add(token)

    return s_tokens


def TF(token, l_tokens):
    value=0
    for elt in l_tokens:
        if token == elt:
            value+=1
    return value


def IDF(token, dd_ref):
    value=0
    nbOfTags=0
    for cui in dd_ref.keys():
        nbOfTags+=1
        l_tokens = dd_ref[cui]["label"].split()
        if token in l_tokens:
            value+=1

        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                nbOfTags+=1
                l_tokens = tag.split()
                if token in l_tokens:
                    value+=1

    value = numpy.log( nbOfTags / (value+1) )

    """
        # Alternative calculation which consider IDF by concept and not names:
        for cui in dd_ref.keys():
        l_tokens.append(dd_ref[cui]["label"])
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                l_tokens.append(tag)
        if token in l_tokens:
            value+=1
    """
    return value


#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':


    ################################################
    print("\n\n\nCADEC (3 datasets):\n")
    ################################################


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

    dd_randCADEC_test0_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-0.test"])
    dd_randCADEC_test1_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-1.test"])
    dd_randCADEC_test2_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-2.test"])
    dd_randCADEC_test3_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-3.test"])
    dd_randCADEC_test4_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-4.test"])
    dd_randCADEC_test5_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-5.test"])
    dd_randCADEC_test6_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-6.test"])
    dd_randCADEC_test7_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-7.test"])
    dd_randCADEC_test8_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-8.test"])
    dd_randCADEC_test9_lowercased = lowercaser_mentions(ddd_randData["AskAPatient.fold-9.test"])

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
    dd_NCBITrainDevFixed_lowercased = lowercaser_mentions(dd_TrainDevFixed)
    dd_NCBITestFixed_lowercased = lowercaser_mentions(dd_TestFixed)


    print("Mentions lowercasing done.\n")


    print("Lowercase references...")
    dd_subsubRef_lowercased = lowercaser_ref(dd_subsubRef)
    dd_habObt_lowercased = lowercaser_ref(dd_habObt)
    dd_medic_lowercased = lowercaser_ref(dd_medic)
    print("Done.")



    ################################################
    print("\n\n\n\nPREDICTING:\n")
    ################################################

    #######################
    print("By heart learning method:")
    #######################
    """
    dd_predictions_customCADEC0_onTrain = optimized_by_heart_matcher(dd_customCADEC_train0_lowercased, dd_customCADEC_train0_lowercased)
    BHscorecustomCADEC0_onTrain = accuracy(dd_predictions_customCADEC0_onTrain, ddd_customData["train_0"])
    print("\n\nBHscorecustomCADEC0_onTrain:", BHscorecustomCADEC0_onTrain)
    dd_predictions_customCADEC1_onTrain = optimized_by_heart_matcher(dd_customCADEC_train1_lowercased, dd_customCADEC_train1_lowercased)
    BHscorecustomCADEC1_onTrain = accuracy(dd_predictions_customCADEC1_onTrain, ddd_customData["train_1"])
    print("\nBHscorecustomCADEC1_onTrain:", BHscorecustomCADEC1_onTrain)
    dd_predictions_customCADEC2_onTrain = optimized_by_heart_matcher(dd_customCADEC_train2_lowercased, dd_customCADEC_train2_lowercased)
    BHscorecustomCADEC2_onTrain = accuracy(dd_predictions_customCADEC2_onTrain, ddd_customData["train_2"])
    print("\nBHscorecustomCADEC2_onTrain:", BHscorecustomCADEC2_onTrain)
    dd_predictions_customCADEC3_onTrain = optimized_by_heart_matcher(dd_customCADEC_train3_lowercased, dd_customCADEC_train3_lowercased)
    BHscorecustomCADEC3_onTrain = accuracy(dd_predictions_customCADEC3_onTrain, ddd_customData["train_3"])
    print("\nBHscorecustomCADEC3_onTrain:", BHscorecustomCADEC3_onTrain)
    dd_predictions_customCADEC4_onTrain = optimized_by_heart_matcher(dd_customCADEC_train4_lowercased, dd_customCADEC_train4_lowercased)
    BHscorecustomCADEC4_onTrain = accuracy(dd_predictions_customCADEC4_onTrain, ddd_customData["train_4"])
    print("\nBHscorecustomCADEC4_onTrain:", BHscorecustomCADEC4_onTrain)

    dd_predictions_customCADEC0_onVal = optimized_by_heart_matcher(dd_customCADEC_validation0_lowercased, dd_customCADEC_train0_lowercased)
    BHscorecustomCADEC0_onVal = accuracy(dd_predictions_customCADEC0_onVal, ddd_customData["test_0"])
    print("\n\nBHscorecustomCADEC0_onVal:", BHscorecustomCADEC0_onVal)
    dd_predictions_customCADEC1_onVal = optimized_by_heart_matcher(dd_customCADEC_validation1_lowercased, dd_customCADEC_train1_lowercased)
    BHscorecustomCADEC1_onVal = accuracy(dd_predictions_customCADEC1_onVal, ddd_customData["test_1"])
    print("\nBHscorecustomCADEC1_onVal:", BHscorecustomCADEC1_onVal)
    dd_predictions_customCADEC2_onVal = optimized_by_heart_matcher(dd_customCADEC_validation2_lowercased, dd_customCADEC_train2_lowercased)
    BHscorecustomCADEC2_onVal = accuracy(dd_predictions_customCADEC2_onVal, ddd_customData["test_2"])
    print("\nBHscorecustomCADEC2_onVal:", BHscorecustomCADEC2_onVal)
    dd_predictions_customCADEC3_onVal = optimized_by_heart_matcher(dd_customCADEC_validation3_lowercased, dd_customCADEC_train3_lowercased)
    BHscorecustomCADEC3_onVal = accuracy(dd_predictions_customCADEC3_onVal, ddd_customData["test_3"])
    print("\nBHscorecustomCADEC3_onVal:", BHscorecustomCADEC3_onVal)
    dd_predictions_customCADEC4_onVal = optimized_by_heart_matcher(dd_customCADEC_validation4_lowercased, dd_customCADEC_train4_lowercased)
    BHscorecustomCADEC4_onVal = accuracy(dd_predictions_customCADEC4_onVal, ddd_customData["test_4"])
    print("\nBHscorecustomCADEC4_onVal:", BHscorecustomCADEC4_onVal)


    dd_predictions_randCADEC0_onTrain = optimized_by_heart_matcher(dd_randCADEC_train0_lowercased, dd_randCADEC_train0_lowercased)
    BHscoreRandCADEC0_onTrain = accuracy(dd_predictions_randCADEC0_onTrain, ddd_randData["AskAPatient.fold-0.train"])
    print("\n\nBHscoreRandCADEC0_onTrain:", BHscoreRandCADEC0_onTrain)
    dd_predictions_randCADEC1_onTrain = optimized_by_heart_matcher(dd_randCADEC_train1_lowercased, dd_randCADEC_train1_lowercased)
    BHscoreRandCADEC1_onTrain = accuracy(dd_predictions_randCADEC1_onTrain, ddd_randData["AskAPatient.fold-1.train"])
    print("\nBHscoreRandCADEC1_onTrain:", BHscoreRandCADEC1_onTrain)
    dd_predictions_randCADEC2_onTrain = optimized_by_heart_matcher(dd_randCADEC_train2_lowercased, dd_randCADEC_train2_lowercased)
    BHscoreRandCADEC2_onTrain = accuracy(dd_predictions_randCADEC2_onTrain, ddd_randData["AskAPatient.fold-2.train"])
    print("\nBHscoreRandCADEC2_onTrain:", BHscoreRandCADEC2_onTrain)
    dd_predictions_randCADEC3_onTrain = optimized_by_heart_matcher(dd_randCADEC_train3_lowercased, dd_randCADEC_train3_lowercased)
    BHscoreRandCADEC3_onTrain = accuracy(dd_predictions_randCADEC3_onTrain, ddd_randData["AskAPatient.fold-3.train"])
    print("\nBHscoreRandCADEC3_onTrain:", BHscoreRandCADEC3_onTrain)
    dd_predictions_randCADEC4_onTrain = optimized_by_heart_matcher(dd_randCADEC_train4_lowercased, dd_randCADEC_train4_lowercased)
    BHscoreRandCADEC4_onTrain = accuracy(dd_predictions_randCADEC4_onTrain, ddd_randData["AskAPatient.fold-4.train"])
    print("\nBHscoreRandCADEC4_onTrain:", BHscoreRandCADEC4_onTrain)
    dd_predictions_randCADEC5_onTrain = optimized_by_heart_matcher(dd_randCADEC_train5_lowercased, dd_randCADEC_train5_lowercased)
    BHscoreRandCADEC5_onTrain = accuracy(dd_predictions_randCADEC5_onTrain, ddd_randData["AskAPatient.fold-5.train"])
    print("\nBHscoreRandCADEC5_onTrain:", BHscoreRandCADEC5_onTrain)
    dd_predictions_randCADEC6_onTrain = optimized_by_heart_matcher(dd_randCADEC_train6_lowercased, dd_randCADEC_train6_lowercased)
    BHscoreRandCADEC6_onTrain = accuracy(dd_predictions_randCADEC6_onTrain, ddd_randData["AskAPatient.fold-6.train"])
    print("\nBHscoreRandCADEC6_onTrain:", BHscoreRandCADEC6_onTrain)
    dd_predictions_randCADEC7_onTrain = optimized_by_heart_matcher(dd_randCADEC_train7_lowercased, dd_randCADEC_train7_lowercased)
    BHscoreRandCADEC7_onTrain = accuracy(dd_predictions_randCADEC7_onTrain, ddd_randData["AskAPatient.fold-7.train"])
    print("\nBHscoreRandCADEC7_onTrain:", BHscoreRandCADEC7_onTrain)
    dd_predictions_randCADEC8_onTrain = optimized_by_heart_matcher(dd_randCADEC_train8_lowercased, dd_randCADEC_train8_lowercased)
    BHscoreRandCADEC8_onTrain = accuracy(dd_predictions_randCADEC8_onTrain, ddd_randData["AskAPatient.fold-8.train"])
    print("\nBHscoreRandCADEC8_onTrain:", BHscoreRandCADEC8_onTrain)
    dd_predictions_randCADEC9_onTrain = optimized_by_heart_matcher(dd_randCADEC_train9_lowercased, dd_randCADEC_train9_lowercased)
    BHscoreRandCADEC9_onTrain = accuracy(dd_predictions_randCADEC9_onTrain, ddd_randData["AskAPatient.fold-9.train"])
    print("\nBHscoreRandCADEC9_onTrain:", BHscoreRandCADEC9_onTrain)

    dd_predictions_randCADEC0_onVal = optimized_by_heart_matcher(dd_randCADEC_validation0_lowercased, dd_randCADEC_train0_lowercased)
    BHscoreRandCADEC0_onVal = accuracy(dd_predictions_randCADEC0_onVal, ddd_randData["AskAPatient.fold-0.validation"])
    print("\n\nBHscoreRandCADEC0_onVal:", BHscoreRandCADEC0_onVal)
    dd_predictions_randCADEC1_onVal = optimized_by_heart_matcher(dd_randCADEC_validation1_lowercased, dd_randCADEC_train1_lowercased)
    BHscoreRandCADEC1_onVal = accuracy(dd_predictions_randCADEC1_onVal, ddd_randData["AskAPatient.fold-1.validation"])
    print("\nBHscoreRandCADEC1_onVal:", BHscoreRandCADEC1_onVal)
    dd_predictions_randCADEC2_onVal = optimized_by_heart_matcher(dd_randCADEC_validation2_lowercased, dd_randCADEC_train2_lowercased)
    BHscoreRandCADEC2_onVal = accuracy(dd_predictions_randCADEC2_onVal, ddd_randData["AskAPatient.fold-2.validation"])
    print("\nBHscoreRandCADEC2_onVal:", BHscoreRandCADEC2_onVal)
    dd_predictions_randCADEC3_onVal = optimized_by_heart_matcher(dd_randCADEC_validation3_lowercased, dd_randCADEC_train3_lowercased)
    BHscoreRandCADEC3_onVal = accuracy(dd_predictions_randCADEC3_onVal, ddd_randData["AskAPatient.fold-3.validation"])
    print("\nBHscoreRandCADEC3_onVal:", BHscoreRandCADEC3_onVal)
    dd_predictions_randCADEC4_onVal = optimized_by_heart_matcher(dd_randCADEC_validation4_lowercased, dd_randCADEC_train4_lowercased)
    BHscoreRandCADEC4_onVal = accuracy(dd_predictions_randCADEC4_onVal, ddd_randData["AskAPatient.fold-4.validation"])
    print("\nBHscoreRandCADEC4_onVal:", BHscoreRandCADEC4_onVal)
    dd_predictions_randCADEC5_onVal = optimized_by_heart_matcher(dd_randCADEC_validation5_lowercased, dd_randCADEC_train5_lowercased)
    BHscoreRandCADEC5_onVal = accuracy(dd_predictions_randCADEC5_onVal, ddd_randData["AskAPatient.fold-5.validation"])
    print("\nBHscoreRandCADEC5_onVal:", BHscoreRandCADEC5_onVal)
    dd_predictions_randCADEC6_onVal = optimized_by_heart_matcher(dd_randCADEC_validation6_lowercased, dd_randCADEC_train6_lowercased)
    BHscoreRandCADEC6_onVal = accuracy(dd_predictions_randCADEC6_onVal, ddd_randData["AskAPatient.fold-6.validation"])
    print("\nBHscoreRandCADEC6_onVal:", BHscoreRandCADEC6_onVal)
    dd_predictions_randCADEC7_onVal = optimized_by_heart_matcher(dd_randCADEC_validation7_lowercased, dd_randCADEC_train7_lowercased)
    BHscoreRandCADEC7_onVal = accuracy(dd_predictions_randCADEC7_onVal, ddd_randData["AskAPatient.fold-7.validation"])
    print("\nBHscoreRandCADEC7_onVal:", BHscoreRandCADEC7_onVal)
    dd_predictions_randCADEC8_onVal = optimized_by_heart_matcher(dd_randCADEC_validation8_lowercased, dd_randCADEC_train8_lowercased)
    BHscoreRandCADEC8_onVal = accuracy(dd_predictions_randCADEC8_onVal, ddd_randData["AskAPatient.fold-8.validation"])
    print("\nBHscoreRandCADEC8_onVal:", BHscoreRandCADEC8_onVal)
    dd_predictions_randCADEC9_onVal = optimized_by_heart_matcher(dd_randCADEC_validation9_lowercased, dd_randCADEC_train9_lowercased)
    BHscoreRandCADEC9_onVal = accuracy(dd_predictions_randCADEC9_onVal, ddd_randData["AskAPatient.fold-9.validation"])
    print("\nBHscoreRandCADEC9_onVal:", BHscoreRandCADEC9_onVal)

    dd_predictions_randCADEC0_onTest = optimized_by_heart_matcher(dd_randCADEC_test0_lowercased, dd_randCADEC_train0_lowercased)
    BHscoreRandCADEC0_onTest = accuracy(dd_predictions_randCADEC0_onTest, ddd_randData["AskAPatient.fold-0.test"])
    print("\n\nBHscoreRandCADEC0_onTest:", BHscoreRandCADEC0_onTest)
    dd_predictions_randCADEC1_onTest = optimized_by_heart_matcher(dd_randCADEC_test1_lowercased, dd_randCADEC_train1_lowercased)
    BHscoreRandCADEC1_onTest = accuracy(dd_predictions_randCADEC1_onTest, ddd_randData["AskAPatient.fold-1.test"])
    print("\nBHscoreRandCADEC1_onTest:", BHscoreRandCADEC1_onTest)
    dd_predictions_randCADEC2_onTest = optimized_by_heart_matcher(dd_randCADEC_test2_lowercased, dd_randCADEC_train2_lowercased)
    BHscoreRandCADEC2_onTest = accuracy(dd_predictions_randCADEC2_onTest, ddd_randData["AskAPatient.fold-2.test"])
    print("\nBHscoreRandCADEC2_onTest:", BHscoreRandCADEC2_onTest)
    dd_predictions_randCADEC3_onTest = optimized_by_heart_matcher(dd_randCADEC_test3_lowercased, dd_randCADEC_train3_lowercased)
    BHscoreRandCADEC3_onTest = accuracy(dd_predictions_randCADEC3_onTest, ddd_randData["AskAPatient.fold-3.test"])
    print("\nBHscoreRandCADEC3_onTest:", BHscoreRandCADEC3_onTest)
    dd_predictions_randCADEC4_onTest = optimized_by_heart_matcher(dd_randCADEC_test4_lowercased, dd_randCADEC_train4_lowercased)
    BHscoreRandCADEC4_onTest = accuracy(dd_predictions_randCADEC4_onTest, ddd_randData["AskAPatient.fold-4.test"])
    print("\nBHscoreRandCADEC4_onTest:", BHscoreRandCADEC4_onTest)
    dd_predictions_randCADEC5_onTest = optimized_by_heart_matcher(dd_randCADEC_test5_lowercased, dd_randCADEC_train5_lowercased)
    BHscoreRandCADEC5_onTest = accuracy(dd_predictions_randCADEC5_onTest, ddd_randData["AskAPatient.fold-5.test"])
    print("\nBHscoreRandCADEC5_onTest:", BHscoreRandCADEC5_onTest)
    dd_predictions_randCADEC6_onTest = optimized_by_heart_matcher(dd_randCADEC_test6_lowercased, dd_randCADEC_train6_lowercased)
    BHscoreRandCADEC6_onTest = accuracy(dd_predictions_randCADEC6_onTest, ddd_randData["AskAPatient.fold-6.test"])
    print("\nBHscoreRandCADEC6_onTest:", BHscoreRandCADEC6_onTest)
    dd_predictions_randCADEC7_onTest = optimized_by_heart_matcher(dd_randCADEC_test7_lowercased, dd_randCADEC_train7_lowercased)
    BHscoreRandCADEC7_onTest = accuracy(dd_predictions_randCADEC7_onTest, ddd_randData["AskAPatient.fold-7.test"])
    print("\nBHscoreRandCADEC7_onTest:", BHscoreRandCADEC7_onTest)
    dd_predictions_randCADEC8_onTest = optimized_by_heart_matcher(dd_randCADEC_test8_lowercased, dd_randCADEC_train8_lowercased)
    BHscoreRandCADEC8_onTest = accuracy(dd_predictions_randCADEC8_onTest, ddd_randData["AskAPatient.fold-8.test"])
    print("\nBHscoreRandCADEC8_onTest:", BHscoreRandCADEC8_onTest)
    dd_predictions_randCADEC9_onTest = optimized_by_heart_matcher(dd_randCADEC_test9_lowercased, dd_randCADEC_train9_lowercased)
    BHscoreRandCADEC9_onTest = accuracy(dd_predictions_randCADEC9_onTest, ddd_randData["AskAPatient.fold-9.test"])
    print("\nBHscoreRandCADEC9_onTest:", BHscoreRandCADEC9_onTest)


    dd_predictions_BB4_onTrain = optimized_by_heart_matcher(dd_BB4habTrain_lowercased, dd_BB4habTrain_lowercased)
    BHscore_BB4_onTrain = accuracy(dd_predictions_BB4_onTrain, dd_habTrain)
    print("\n\nBHscore_BB4_onTrain:", BHscore_BB4_onTrain)
    dd_predictions_BB4_onVal = optimized_by_heart_matcher(dd_BB4habDev_lowercased, dd_BB4habTrain_lowercased)
    BHscore_BB4_onVal = accuracy(dd_predictions_BB4_onVal, dd_habDev)
    print("\nBHscore_BB4_onVal:", BHscore_BB4_onVal)

    #Applied to BB4 test, but need a formater to a1/a2 files to evaluate.


    dd_predictions_NCBI_onTrain = optimized_by_heart_matcher(dd_NCBITrainFixed_lowercased, dd_NCBITrainFixed_lowercased)
    BHscore_NCBI_onTrain = accuracy(dd_predictions_NCBI_onTrain, dd_TrainFixed)
    print("\n\nBHscore_NCBI_onTrain:", BHscore_NCBI_onTrain)
    dd_predictions_NCBI_onVal = optimized_by_heart_matcher(dd_NCBIDevFixed_lowercased, dd_NCBITrainFixed_lowercased)
    BHscore_NCBI_onVal = accuracy(dd_predictions_NCBI_onVal, dd_DevFixed)
    print("\nBHscore_NCBI_onVal:", BHscore_NCBI_onVal)
    dd_predictions_NCBI_onTestWithTrainDev = optimized_by_heart_matcher(dd_NCBITestFixed_lowercased, dd_NCBITrainDevFixed_lowercased)
    BHscore_NCBI_onTestWithTrainDev = accuracy(dd_predictions_NCBI_onTestWithTrainDev, dd_TestFixed)
    print("\nBHscore_NCBI_onTestWithTrainDev:", BHscore_NCBI_onTestWithTrainDev)
    dd_predictions_NCBI_onTest = optimized_by_heart_matcher(dd_NCBITestFixed_lowercased, dd_NCBITrainFixed_lowercased)
    BHscore_NCBI_onTest = accuracy(dd_predictions_NCBI_onTest, dd_TestFixed)
    print("\nBHscore_NCBI_onTest:", BHscore_NCBI_onTest)

    """

    #######################
    print("\n\n\nExact Matching method:\n")
    #######################
    """
    dd_EMpredictions_customCADEC0_onTrain = optimized_exact_matcher(dd_customCADEC_train0_lowercased, dd_subsubRef_lowercased)
    EMscorecustomCADEC0_onTrain = accuracy(dd_EMpredictions_customCADEC0_onTrain, ddd_customData["train_0"])
    print("\nEMscorecustomCADEC0_onTrain:", EMscorecustomCADEC0_onTrain)
    dd_EMpredictions_customCADEC1_onTrain = optimized_exact_matcher(dd_customCADEC_train1_lowercased, dd_subsubRef_lowercased)
    EMscorecustomCADEC1_onTrain = accuracy(dd_EMpredictions_customCADEC1_onTrain, ddd_customData["train_1"])
    print("\nEMscorecustomCADEC1_onTrain:", EMscorecustomCADEC1_onTrain)
    dd_EMpredictions_customCADEC2_onTrain = optimized_exact_matcher(dd_customCADEC_train2_lowercased, dd_subsubRef_lowercased)
    EMscorecustomCADEC2_onTrain = accuracy(dd_EMpredictions_customCADEC2_onTrain, ddd_customData["train_2"])
    print("\nEMscorecustomCADEC2_onTrain:", EMscorecustomCADEC2_onTrain)
    dd_EMpredictions_customCADEC3_onTrain = optimized_exact_matcher(dd_customCADEC_train3_lowercased, dd_subsubRef_lowercased)
    EMscorecustomCADEC3_onTrain = accuracy(dd_EMpredictions_customCADEC3_onTrain, ddd_customData["train_3"])
    print("\nEMscorecustomCADEC3_onTrain:", EMscorecustomCADEC3_onTrain)
    dd_EMpredictions_customCADEC4_onTrain = optimized_exact_matcher(dd_customCADEC_train4_lowercased, dd_subsubRef_lowercased)
    EMscorecustomCADEC4_onTrain = accuracy(dd_EMpredictions_customCADEC4_onTrain, ddd_customData["train_4"])
    print("\nEMscorecustomCADEC4_onTrain:", EMscorecustomCADEC4_onTrain)

    dd_EMpredictions_customCADEC0_onVal = optimized_exact_matcher(dd_customCADEC_validation0_lowercased, dd_subsubRef_lowercased)
    EMscorecustomCADEC0_onVal = accuracy(dd_EMpredictions_customCADEC0_onVal, ddd_customData["test_0"])
    print("\n\nEMscorecustomCADEC0_onVal:", EMscorecustomCADEC0_onVal)
    dd_EMpredictions_customCADEC1_onVal = optimized_exact_matcher(dd_customCADEC_validation1_lowercased, dd_subsubRef_lowercased)
    EMscorecustomCADEC1_onVal = accuracy(dd_EMpredictions_customCADEC1_onVal, ddd_customData["test_1"])
    print("\nEMscorecustomCADEC1_onVal:", EMscorecustomCADEC1_onVal)
    dd_EMpredictions_customCADEC2_onVal = optimized_exact_matcher(dd_customCADEC_validation2_lowercased, dd_subsubRef_lowercased)
    EMscorecustomCADEC2_onVal = accuracy(dd_EMpredictions_customCADEC2_onVal, ddd_customData["test_2"])
    print("\nEMscorecustomCADEC2_onVal:", EMscorecustomCADEC2_onVal)
    dd_EMpredictions_customCADEC3_onVal = optimized_exact_matcher(dd_customCADEC_validation3_lowercased, dd_subsubRef_lowercased)
    EMscorecustomCADEC3_onVal = accuracy(dd_EMpredictions_customCADEC3_onVal, ddd_customData["test_3"])
    print("\nEMscorecustomCADEC3_onVal:", EMscorecustomCADEC3_onVal)
    dd_EMpredictions_customCADEC4_onVal = optimized_exact_matcher(dd_customCADEC_validation4_lowercased, dd_subsubRef_lowercased)
    EMscorecustomCADEC4_onVal = accuracy(dd_EMpredictions_customCADEC4_onVal, ddd_customData["test_4"])
    print("\nEMscorecustomCADEC4_onVal:", EMscorecustomCADEC4_onVal)


    dd_EMpredictions_randCADEC0_onTrain = optimized_exact_matcher(dd_randCADEC_train0_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC0_onTrain = accuracy(dd_EMpredictions_randCADEC0_onTrain, ddd_randData["AskAPatient.fold-0.train"])
    print("\n\nEMscoreRandCADEC0_onTrain:", EMscoreRandCADEC0_onTrain)
    dd_EMpredictions_randCADEC1_onTrain = optimized_exact_matcher(dd_randCADEC_train1_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC1_onTrain = accuracy(dd_EMpredictions_randCADEC1_onTrain, ddd_randData["AskAPatient.fold-1.train"])
    print("\nEMscoreRandCADEC1_onTrain:", EMscoreRandCADEC1_onTrain)
    dd_EMpredictions_randCADEC2_onTrain = optimized_exact_matcher(dd_randCADEC_train2_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC2_onTrain = accuracy(dd_EMpredictions_randCADEC2_onTrain, ddd_randData["AskAPatient.fold-2.train"])
    print("\nEMscoreRandCADEC2_onTrain:", EMscoreRandCADEC2_onTrain)
    dd_EMpredictions_randCADEC3_onTrain = optimized_exact_matcher(dd_randCADEC_train3_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC3_onTrain = accuracy(dd_EMpredictions_randCADEC3_onTrain, ddd_randData["AskAPatient.fold-3.train"])
    print("\nEMscoreRandCADEC3_onTrain:", EMscoreRandCADEC3_onTrain)
    dd_EMpredictions_randCADEC4_onTrain = optimized_exact_matcher(dd_randCADEC_train4_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC4_onTrain = accuracy(dd_EMpredictions_randCADEC4_onTrain, ddd_randData["AskAPatient.fold-4.train"])
    print("\nEMscoreRandCADEC4_onTrain:", EMscoreRandCADEC4_onTrain)
    dd_EMpredictions_randCADEC5_onTrain = optimized_exact_matcher(dd_randCADEC_train5_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC5_onTrain = accuracy(dd_EMpredictions_randCADEC5_onTrain, ddd_randData["AskAPatient.fold-5.train"])
    print("\nEMscoreRandCADEC5_onTrain:", EMscoreRandCADEC5_onTrain)
    dd_EMpredictions_randCADEC6_onTrain = optimized_exact_matcher(dd_randCADEC_train6_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC6_onTrain = accuracy(dd_EMpredictions_randCADEC6_onTrain, ddd_randData["AskAPatient.fold-6.train"])
    print("\nEMscoreRandCADEC6_onTrain:", EMscoreRandCADEC6_onTrain)
    dd_EMpredictions_randCADEC7_onTrain = optimized_exact_matcher(dd_randCADEC_train7_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC7_onTrain = accuracy(dd_EMpredictions_randCADEC7_onTrain, ddd_randData["AskAPatient.fold-7.train"])
    print("\nEMscoreRandCADEC7_onTrain:", EMscoreRandCADEC7_onTrain)
    dd_EMpredictions_randCADEC8_onTrain = optimized_exact_matcher(dd_randCADEC_train8_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC8_onTrain = accuracy(dd_EMpredictions_randCADEC8_onTrain, ddd_randData["AskAPatient.fold-8.train"])
    print("\nEMscoreRandCADEC8_onTrain:", EMscoreRandCADEC8_onTrain)
    dd_EMpredictions_randCADEC9_onTrain = optimized_exact_matcher(dd_randCADEC_train9_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC9_onTrain = accuracy(dd_EMpredictions_randCADEC9_onTrain, ddd_randData["AskAPatient.fold-9.train"])
    print("\nEMscoreRandCADEC9_onTrain:", EMscoreRandCADEC9_onTrain)


    dd_EMpredictions_randCADEC0_onVal = optimized_exact_matcher(dd_randCADEC_validation0_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC0_onVal = accuracy(dd_EMpredictions_randCADEC0_onVal, ddd_randData["AskAPatient.fold-0.validation"])
    print("\n\nEMscoreRandCADEC0_onVal:", EMscoreRandCADEC0_onVal)
    dd_EMpredictions_randCADEC1_onVal = optimized_exact_matcher(dd_randCADEC_validation1_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC1_onVal = accuracy(dd_EMpredictions_randCADEC1_onVal, ddd_randData["AskAPatient.fold-1.validation"])
    print("\nEMscoreRandCADEC1_onVal:", EMscoreRandCADEC1_onVal)
    dd_EMpredictions_randCADEC2_onVal = optimized_exact_matcher(dd_randCADEC_validation2_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC2_onVal = accuracy(dd_EMpredictions_randCADEC2_onVal, ddd_randData["AskAPatient.fold-2.validation"])
    print("\nEMscoreRandCADEC2_onVal:", EMscoreRandCADEC2_onVal)
    dd_EMpredictions_randCADEC3_onVal = optimized_exact_matcher(dd_randCADEC_validation3_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC3_onVal = accuracy(dd_EMpredictions_randCADEC3_onVal, ddd_randData["AskAPatient.fold-3.validation"])
    print("\nEMscoreRandCADEC3_onVal:", EMscoreRandCADEC3_onVal)
    dd_EMpredictions_randCADEC4_onVal = optimized_exact_matcher(dd_randCADEC_validation4_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC4_onVal = accuracy(dd_EMpredictions_randCADEC4_onVal, ddd_randData["AskAPatient.fold-4.validation"])
    print("\nEMscoreRandCADEC4_onVal:", EMscoreRandCADEC4_onVal)
    dd_EMpredictions_randCADEC5_onVal = optimized_exact_matcher(dd_randCADEC_validation5_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC5_onVal = accuracy(dd_EMpredictions_randCADEC5_onVal, ddd_randData["AskAPatient.fold-5.validation"])
    print("\nEMscoreRandCADEC5_onVal:", EMscoreRandCADEC5_onVal)
    dd_EMpredictions_randCADEC6_onVal = optimized_exact_matcher(dd_randCADEC_validation6_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC6_onVal = accuracy(dd_EMpredictions_randCADEC6_onVal, ddd_randData["AskAPatient.fold-6.validation"])
    print("\nEMscoreRandCADEC6_onVal:", EMscoreRandCADEC6_onVal)
    dd_EMpredictions_randCADEC7_onVal = optimized_exact_matcher(dd_randCADEC_validation7_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC7_onVal = accuracy(dd_EMpredictions_randCADEC7_onVal, ddd_randData["AskAPatient.fold-7.validation"])
    print("\nEMscoreRandCADEC7_onVal:", EMscoreRandCADEC7_onVal)
    dd_EMpredictions_randCADEC8_onVal = optimized_exact_matcher(dd_randCADEC_validation8_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC8_onVal = accuracy(dd_EMpredictions_randCADEC8_onVal, ddd_randData["AskAPatient.fold-8.validation"])
    print("\nEMscoreRandCADEC8_onVal:", EMscoreRandCADEC8_onVal)
    dd_EMpredictions_randCADEC9_onVal = optimized_exact_matcher(dd_randCADEC_validation9_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC9_onVal = accuracy(dd_EMpredictions_randCADEC9_onVal, ddd_randData["AskAPatient.fold-9.validation"])
    print("\nEMscoreRandCADEC9_onVal:", EMscoreRandCADEC9_onVal)


    dd_EMpredictions_randCADEC0_onTest = optimized_exact_matcher(dd_randCADEC_test0_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC0_onTest = accuracy(dd_EMpredictions_randCADEC0_onTest, ddd_randData["AskAPatient.fold-0.test"])
    print("\n\nEMscoreRandCADEC0_onTest:", EMscoreRandCADEC0_onTest)
    dd_EMpredictions_randCADEC1_onTest = optimized_exact_matcher(dd_randCADEC_test1_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC1_onTest = accuracy(dd_EMpredictions_randCADEC1_onTest, ddd_randData["AskAPatient.fold-1.test"])
    print("\nEMscoreRandCADEC1_onTest:", EMscoreRandCADEC1_onTest)
    dd_EMpredictions_randCADEC2_onTest = optimized_exact_matcher(dd_randCADEC_test2_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC2_onTest = accuracy(dd_EMpredictions_randCADEC2_onTest, ddd_randData["AskAPatient.fold-2.test"])
    print("\nEMscoreRandCADEC2_onTest:", EMscoreRandCADEC2_onTest)
    dd_EMpredictions_randCADEC3_onTest = optimized_exact_matcher(dd_randCADEC_test3_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC3_onTest = accuracy(dd_EMpredictions_randCADEC3_onTest, ddd_randData["AskAPatient.fold-3.test"])
    print("\nEMscoreRandCADEC3_onTest:", EMscoreRandCADEC3_onTest)
    dd_EMpredictions_randCADEC4_onTest = optimized_exact_matcher(dd_randCADEC_test4_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC4_onTest = accuracy(dd_EMpredictions_randCADEC4_onTest, ddd_randData["AskAPatient.fold-4.test"])
    print("\nEMscoreRandCADEC4_onTest:", EMscoreRandCADEC4_onTest)
    dd_EMpredictions_randCADEC5_onTest = optimized_exact_matcher(dd_randCADEC_test5_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC5_onTest = accuracy(dd_EMpredictions_randCADEC5_onTest, ddd_randData["AskAPatient.fold-5.test"])
    print("\nEMscoreRandCADEC5_onTest:", EMscoreRandCADEC5_onTest)
    dd_EMpredictions_randCADEC6_onTest = optimized_exact_matcher(dd_randCADEC_test6_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC6_onTest = accuracy(dd_EMpredictions_randCADEC6_onTest, ddd_randData["AskAPatient.fold-6.test"])
    print("\nEMscoreRandCADEC6_onTest:", EMscoreRandCADEC6_onTest)
    dd_EMpredictions_randCADEC7_onTest = optimized_exact_matcher(dd_randCADEC_test7_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC7_onTest = accuracy(dd_EMpredictions_randCADEC7_onTest, ddd_randData["AskAPatient.fold-7.test"])
    print("\nEMscoreRandCADEC7_onTest:", EMscoreRandCADEC7_onTest)
    dd_EMpredictions_randCADEC8_onTest = optimized_exact_matcher(dd_randCADEC_test8_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC8_onTest = accuracy(dd_EMpredictions_randCADEC8_onTest, ddd_randData["AskAPatient.fold-8.test"])
    print("\nEMscoreRandCADEC8_onTest:", EMscoreRandCADEC8_onTest)
    dd_EMpredictions_randCADEC9_onTest = optimized_exact_matcher(dd_randCADEC_test9_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC9_onTest = accuracy(dd_EMpredictions_randCADEC9_onTest, ddd_randData["AskAPatient.fold-9.test"])
    print("\nEMscoreRandCADEC9_onTest:", EMscoreRandCADEC9_onTest)


    dd_EMpredictions_BB4_onTrain = optimized_exact_matcher(dd_BB4habTrain_lowercased, dd_habObt_lowercased)
    EMscore_BB4_onTrain = accuracy(dd_EMpredictions_BB4_onTrain, dd_habTrain)
    print("\n\nEMscore_BB4_onTrain:", EMscore_BB4_onTrain)
    dd_EMpredictions_BB4_onVal = optimized_exact_matcher(dd_BB4habDev_lowercased, dd_habObt_lowercased)
    EMscore_BB4_onVal = accuracy(dd_EMpredictions_BB4_onVal, dd_habDev)
    print("\nEMscore_BB4_onVal:", EMscore_BB4_onVal)

    #Applied to BB4 test, but need a formater to a1/a2 files to evaluate.


    dd_EMpredictions_NCBI_onTrain = optimized_exact_matcher(dd_NCBITrainFixed_lowercased, dd_medic_lowercased)
    EMscore_NCBI_onTrain = accuracy(dd_EMpredictions_NCBI_onTrain, dd_NCBITrainFixed_lowercased)
    print("\n\nEMscore_NCBI_onTrain:", EMscore_NCBI_onTrain)
    dd_EMpredictions_NCBI_onVal = optimized_exact_matcher(dd_NCBIDevFixed_lowercased, dd_medic_lowercased)
    EMscore_NCBI_onVal = accuracy(dd_EMpredictions_NCBI_onVal, dd_NCBIDevFixed_lowercased)
    print("\nEMscore_NCBI_onVal:", EMscore_NCBI_onVal)
    dd_EMpredictions_NCBI_onTest = optimized_exact_matcher(dd_NCBITestFixed_lowercased, dd_medic_lowercased)
    EMscore_NCBI_onTest = accuracy(dd_EMpredictions_NCBI_onTest, dd_TestFixed)
    print("\nEMscore_NCBI_onTest:", EMscore_NCBI_onTest)

    """

    #######################
    print("\n\n\nExact Matching + By Heart method:\n")
    #######################
    """
    dd_predictions_customCADEC0_onVal = by_heart_and_exact_matching(dd_customCADEC_validation0_lowercased, dd_customCADEC_train0_lowercased, dd_subsubRef)
    BHEMscorecustomCADEC0_onVal = accuracy(dd_predictions_customCADEC0_onVal, ddd_customData["test_0"])
    print("\n\nBHEMscorecustomCADEC0_onVal:", BHEMscorecustomCADEC0_onVal)
    dd_predictions_customCADEC1_onVal = by_heart_and_exact_matching(dd_customCADEC_validation1_lowercased, dd_customCADEC_train1_lowercased, dd_subsubRef)
    BHEMscorecustomCADEC1_onVal = accuracy(dd_predictions_customCADEC1_onVal, ddd_customData["test_1"])
    print("\nBHEMscorecustomCADEC1_onVal:", BHEMscorecustomCADEC1_onVal)
    dd_predictions_customCADEC2_onVal = by_heart_and_exact_matching(dd_customCADEC_validation2_lowercased, dd_customCADEC_train2_lowercased, dd_subsubRef)
    BHEMscorecustomCADEC2_onVal = accuracy(dd_predictions_customCADEC2_onVal, ddd_customData["test_2"])
    print("\nBHEMscorecustomCADEC2_onVal:", BHEMscorecustomCADEC2_onVal)
    dd_predictions_customCADEC3_onVal = by_heart_and_exact_matching(dd_customCADEC_validation3_lowercased, dd_customCADEC_train3_lowercased, dd_subsubRef)
    BHEMscorecustomCADEC3_onVal = accuracy(dd_predictions_customCADEC3_onVal, ddd_customData["test_3"])
    print("\nBHEMscorecustomCADEC3_onVal:", BHEMscorecustomCADEC3_onVal)
    dd_predictions_customCADEC4_onVal = by_heart_and_exact_matching(dd_customCADEC_validation4_lowercased, dd_customCADEC_train4_lowercased, dd_subsubRef)
    BHEMscorecustomCADEC4_onVal = accuracy(dd_predictions_customCADEC4_onVal, ddd_customData["test_4"])
    print("\nBHEMscorecustomCADEC4_onVal:", BHEMscorecustomCADEC4_onVal)


    dd_BHEMpredictions_randCADEC0_onTest = by_heart_and_exact_matching(dd_randCADEC_test0_lowercased, dd_randCADEC_train0_lowercased, dd_subsubRef_lowercased)
    BHEMscoreRandCADEC0_onTest = accuracy(dd_BHEMpredictions_randCADEC0_onTest, ddd_randData["AskAPatient.fold-0.test"])
    print("\n\nBHEMscoreRandCADEC0_onTest:", BHEMscoreRandCADEC0_onTest)
    dd_BHEMpredictions_randCADEC1_onTest = by_heart_and_exact_matching(dd_randCADEC_test1_lowercased, dd_randCADEC_train1_lowercased, dd_subsubRef_lowercased)
    BHEMscoreRandCADEC1_onTest = accuracy(dd_BHEMpredictions_randCADEC1_onTest, ddd_randData["AskAPatient.fold-1.test"])
    print("\nBHEMscoreRandCADEC1_onTest:", BHEMscoreRandCADEC1_onTest)
    dd_BHEMpredictions_randCADEC2_onTest = by_heart_and_exact_matching(dd_randCADEC_test2_lowercased, dd_randCADEC_train2_lowercased, dd_subsubRef_lowercased)
    BHEMscoreRandCADEC2_onTest = accuracy(dd_BHEMpredictions_randCADEC2_onTest, ddd_randData["AskAPatient.fold-2.test"])
    print("\nBHEMscoreRandCADEC2_onTest:", BHEMscoreRandCADEC2_onTest)
    dd_BHEMpredictions_randCADEC3_onTest = by_heart_and_exact_matching(dd_randCADEC_test3_lowercased, dd_randCADEC_train3_lowercased, dd_subsubRef_lowercased)
    BHEMscoreRandCADEC3_onTest = accuracy(dd_BHEMpredictions_randCADEC3_onTest, ddd_randData["AskAPatient.fold-3.test"])
    print("\nBHEMscoreRandCADEC3_onTest:", BHEMscoreRandCADEC3_onTest)
    dd_BHEMpredictions_randCADEC4_onTest = by_heart_and_exact_matching(dd_randCADEC_test4_lowercased, dd_randCADEC_train4_lowercased, dd_subsubRef_lowercased)
    BHEMscoreRandCADEC4_onTest = accuracy(dd_BHEMpredictions_randCADEC4_onTest, ddd_randData["AskAPatient.fold-4.test"])
    print("\nBHEMscoreRandCADEC4_onTest:", BHEMscoreRandCADEC4_onTest)
    dd_BHEMpredictions_randCADEC5_onTest = by_heart_and_exact_matching(dd_randCADEC_test5_lowercased, dd_randCADEC_train5_lowercased, dd_subsubRef_lowercased)
    BHEMscoreRandCADEC5_onTest = accuracy(dd_BHEMpredictions_randCADEC5_onTest, ddd_randData["AskAPatient.fold-5.test"])
    print("\nBHEMscoreRandCADEC5_onTest:", BHEMscoreRandCADEC5_onTest)
    dd_BHEMpredictions_randCADEC6_onTest = by_heart_and_exact_matching(dd_randCADEC_test6_lowercased, dd_randCADEC_train6_lowercased, dd_subsubRef_lowercased)
    BHEMscoreRandCADEC6_onTest = accuracy(dd_BHEMpredictions_randCADEC6_onTest, ddd_randData["AskAPatient.fold-6.test"])
    print("\nBHEMscoreRandCADEC6_onTest:", BHEMscoreRandCADEC6_onTest)
    dd_BHEMpredictions_randCADEC7_onTest = by_heart_and_exact_matching(dd_randCADEC_test7_lowercased, dd_randCADEC_train7_lowercased, dd_subsubRef_lowercased)
    BHEMscoreRandCADEC7_onTest = accuracy(dd_BHEMpredictions_randCADEC7_onTest, ddd_randData["AskAPatient.fold-7.test"])
    print("\nBHEMscoreRandCADEC7_onTest:", BHEMscoreRandCADEC7_onTest)
    dd_BHEMpredictions_randCADEC8_onTest = by_heart_and_exact_matching(dd_randCADEC_test8_lowercased, dd_randCADEC_train8_lowercased, dd_subsubRef_lowercased)
    BHEMscoreRandCADEC8_onTest = accuracy(dd_BHEMpredictions_randCADEC8_onTest, ddd_randData["AskAPatient.fold-8.test"])
    print("\nBHEMscoreRandCADEC8_onTest:", BHEMscoreRandCADEC8_onTest)
    dd_BHEMpredictions_randCADEC9_onTest = by_heart_and_exact_matching(dd_randCADEC_test9_lowercased, dd_randCADEC_train9_lowercased, dd_subsubRef_lowercased)
    BHEMscoreRandCADEC9_onTest = accuracy(dd_BHEMpredictions_randCADEC9_onTest, ddd_randData["AskAPatient.fold-9.test"])
    print("\nBHEMscoreRandCADEC9_onTest:", BHEMscoreRandCADEC9_onTest)


    dd_predictions_BB4_onVal = by_heart_and_exact_matching(dd_BB4habDev_lowercased, dd_BB4habTrain_lowercased, dd_habObt_lowercased)
    BHEMscore_BB4_onVal = accuracy(dd_predictions_BB4_onVal, dd_habDev)
    print("\n\nBHEMscore_BB4_onVal:", BHEMscore_BB4_onVal)


    """
    #######################
    print("\n\n\nStatic distance between label/mention embeddings:\n")
    #######################

    """
    word_vectors = KeyedVectors.load_word2vec_format('../PubMed-w2v.bin', binary=True)
    #word_vectors = Word2Vec.load('../VST_count0_size100_iter50.model')


    dd_predictions_customCADEC0_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation0_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscorecustomCADEC0_onVal = accuracy(dd_predictions_customCADEC0_onVal, ddd_customData["test_0"])
    print("\n\nSEscorecustomCADEC0_onVal:", SEscorecustomCADEC0_onVal)
    dd_predictions_customCADEC1_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation1_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscorecustomCADEC1_onVal = accuracy(dd_predictions_customCADEC1_onVal, ddd_customData["test_1"])
    print("\nSEscorecustomCADEC1_onVal:", SEscorecustomCADEC1_onVal)
    dd_predictions_customCADEC2_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation2_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscorecustomCADEC2_onVal = accuracy(dd_predictions_customCADEC2_onVal, ddd_customData["test_2"])
    print("\nSEscorecustomCADEC2_onVal:", SEscorecustomCADEC2_onVal)
    dd_predictions_customCADEC3_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation3_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscorecustomCADEC3_onVal = accuracy(dd_predictions_customCADEC3_onVal, ddd_customData["test_3"])
    print("\nSEscorecustomCADEC3_onVal:", SEscorecustomCADEC3_onVal)
    dd_predictions_customCADEC4_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation4_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscorecustomCADEC4_onVal = accuracy(dd_predictions_customCADEC4_onVal, ddd_customData["test_4"])
    print("\nSEscorecustomCADEC4_onVal:", SEscorecustomCADEC4_onVal)


    dd_SEpredictions_randCADEC0_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test0_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscoreRandCADEC0_onTest = accuracy(dd_SEpredictions_randCADEC0_onTest, ddd_randData["AskAPatient.fold-0.test"])
    print("\n\nSEscoreRandCADEC0_onTest:", SEscoreRandCADEC0_onTest)
    dd_SEpredictions_randCADEC1_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test1_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscoreRandCADEC1_onTest = accuracy(dd_SEpredictions_randCADEC1_onTest, ddd_randData["AskAPatient.fold-1.test"])
    print("\nSEscoreRandCADEC1_onTest:", SEscoreRandCADEC1_onTest)
    dd_SEpredictions_randCADEC2_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test2_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscoreRandCADEC2_onTest = accuracy(dd_SEpredictions_randCADEC2_onTest, ddd_randData["AskAPatient.fold-2.test"])
    print("\nSEscoreRandCADEC2_onTest:", SEscoreRandCADEC2_onTest)
    dd_SEpredictions_randCADEC3_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test3_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscoreRandCADEC3_onTest = accuracy(dd_SEpredictions_randCADEC3_onTest, ddd_randData["AskAPatient.fold-3.test"])
    print("\nSEscoreRandCADEC3_onTest:", SEscoreRandCADEC3_onTest)
    dd_SEpredictions_randCADEC4_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test4_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscoreRandCADEC4_onTest = accuracy(dd_SEpredictions_randCADEC4_onTest, ddd_randData["AskAPatient.fold-4.test"])
    print("\nSEscoreRandCADEC4_onTest:", SEscoreRandCADEC4_onTest)
    dd_SEpredictions_randCADEC5_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test5_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscoreRandCADEC5_onTest = accuracy(dd_SEpredictions_randCADEC5_onTest, ddd_randData["AskAPatient.fold-5.test"])
    print("\nSEscoreRandCADEC5_onTest:", SEscoreRandCADEC5_onTest)
    dd_SEpredictions_randCADEC6_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test6_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscoreRandCADEC6_onTest = accuracy(dd_SEpredictions_randCADEC6_onTest, ddd_randData["AskAPatient.fold-6.test"])
    print("\nSEscoreRandCADEC6_onTest:", SEscoreRandCADEC6_onTest)
    dd_SEpredictions_randCADEC7_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test7_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscoreRandCADEC7_onTest = accuracy(dd_SEpredictions_randCADEC7_onTest, ddd_randData["AskAPatient.fold-7.test"])
    print("\nSEscoreRandCADEC7_onTest:", SEscoreRandCADEC7_onTest)
    dd_SEpredictions_randCADEC8_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test8_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscoreRandCADEC8_onTest = accuracy(dd_SEpredictions_randCADEC8_onTest, ddd_randData["AskAPatient.fold-8.test"])
    print("\nSEscoreRandCADEC8_onTest:", SEscoreRandCADEC8_onTest)
    dd_SEpredictions_randCADEC9_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test9_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscoreRandCADEC9_onTest = accuracy(dd_SEpredictions_randCADEC9_onTest, ddd_randData["AskAPatient.fold-9.test"])
    print("\nSEscoreRandCADEC9_onTest:", SEscoreRandCADEC9_onTest)


    dd_predictions_BB4_onVal = embeddings_similarity_method_with_tags(dd_BB4habDev_lowercased, dd_habObt_lowercased, word_vectors)
    SEscore_BB4_onVal = accuracy(dd_predictions_BB4_onVal, dd_habDev)
    print("\n\nSEscore_BB4_onVal:", SEscore_BB4_onVal)


    dd_EMpredictions_NCBI_onVal = embeddings_similarity_method_with_tags(dd_NCBIDevFixed_lowercased, dd_medic_lowercased, word_vectors)
    SEscore_NCBI_onVal = accuracy(dd_EMpredictions_NCBI_onVal, dd_NCBIDevFixed_lowercased)
    print("\nSEscore_NCBI_onVal (with tags):", SEscore_NCBI_onVal)
    dd_EMpredictions_NCBI_onTest = embeddings_similarity_method_with_tags(dd_NCBITestFixed_lowercased, dd_medic_lowercased, word_vectors)
    SEscore_NCBI_onTest = accuracy(dd_EMpredictions_NCBI_onTest, dd_TestFixed)
    print("\nSEscore_NCBI_onTest (with tags):", SEscore_NCBI_onTest)
    """


    #######################
    print("\n\n\nML distance between label/mention embeddings:\n")
    #######################
    """
    word_vectors = KeyedVectors.load_word2vec_format('../PubMed-w2v.bin', binary=True)
    # word_vectors = Word2Vec.load('../VST_count0_size100_iter50.model')


    dd_predictions_customCADEC0_onVal = dense_layer_method(dd_customCADEC_train0_lowercased, dd_customCADEC_validation0_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscorecustomCADEC0_onVal = accuracy(dd_predictions_customCADEC0_onVal, ddd_customData["test_0"])
    print("\n\nMLEscorecustomCADEC0_onVal:", MLEscorecustomCADEC0_onVal)
    dd_predictions_customCADEC1_onVal = dense_layer_method(dd_customCADEC_train1_lowercased, dd_customCADEC_validation1_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscorecustomCADEC1_onVal = accuracy(dd_predictions_customCADEC1_onVal, ddd_customData["test_1"])
    print("\nMLEscorecustomCADEC1_onVal:", MLEscorecustomCADEC1_onVal)
    dd_predictions_customCADEC2_onVal = dense_layer_method(dd_customCADEC_train2_lowercased, dd_customCADEC_validation2_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscorecustomCADEC2_onVal = accuracy(dd_predictions_customCADEC2_onVal, ddd_customData["test_2"])
    print("\nMLEscorecustomCADEC2_onVal:", MLEscorecustomCADEC2_onVal)
    dd_predictions_customCADEC3_onVal = dense_layer_method(dd_customCADEC_train3_lowercased, dd_customCADEC_validation3_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscorecustomCADEC3_onVal = accuracy(dd_predictions_customCADEC3_onVal, ddd_customData["test_3"])
    print("\nMLEscorecustomCADEC3_onVal:", MLEscorecustomCADEC3_onVal)
    dd_predictions_customCADEC4_onVal = dense_layer_method(dd_customCADEC_train4_lowercased, dd_customCADEC_validation4_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscorecustomCADEC4_onVal = accuracy(dd_predictions_customCADEC4_onVal, ddd_customData["test_4"])
    print("\nMLEscorecustomCADEC4_onVal:", MLEscorecustomCADEC4_onVal)


    dd_MLEpredictions_randCADEC0_onTest = dense_layer_method(dd_randCADEC_train0_lowercased, dd_randCADEC_test0_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC0_onTest = accuracy(dd_MLEpredictions_randCADEC0_onTest, ddd_randData["AskAPatient.fold-0.test"])
    print("\n\nMLEscoreRandCADEC0_onTest:", MLEscoreRandCADEC0_onTest)
    dd_MLEpredictions_randCADEC1_onTest = dense_layer_method(dd_randCADEC_train1_lowercased, dd_randCADEC_test1_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC1_onTest = accuracy(dd_MLEpredictions_randCADEC1_onTest, ddd_randData["AskAPatient.fold-1.test"])
    print("\nMLEscoreRandCADEC1_onTest:", MLEscoreRandCADEC1_onTest)
    dd_MLEpredictions_randCADEC2_onTest = dense_layer_method(dd_randCADEC_train2_lowercased, dd_randCADEC_test2_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC2_onTest = accuracy(dd_MLEpredictions_randCADEC2_onTest, ddd_randData["AskAPatient.fold-2.test"])
    print("\nMLEscoreRandCADEC2_onTest:", MLEscoreRandCADEC2_onTest)
    dd_MLEpredictions_randCADEC3_onTest = dense_layer_method(dd_randCADEC_train3_lowercased, dd_randCADEC_test3_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC3_onTest = accuracy(dd_MLEpredictions_randCADEC3_onTest, ddd_randData["AskAPatient.fold-3.test"])
    print("\nMLEscoreRandCADEC3_onTest:", MLEscoreRandCADEC3_onTest)
    dd_MLEpredictions_randCADEC4_onTest = dense_layer_method(dd_randCADEC_train4_lowercased, dd_randCADEC_test4_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC4_onTest = accuracy(dd_MLEpredictions_randCADEC4_onTest, ddd_randData["AskAPatient.fold-4.test"])
    print("\nMLEscoreRandCADEC4_onTest:", MLEscoreRandCADEC4_onTest)
    dd_MLEpredictions_randCADEC5_onTest = dense_layer_method(dd_randCADEC_train5_lowercased, dd_randCADEC_test5_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC5_onTest = accuracy(dd_MLEpredictions_randCADEC5_onTest, ddd_randData["AskAPatient.fold-5.test"])
    print("\nMLEscoreRandCADEC5_onTest:", MLEscoreRandCADEC5_onTest)
    dd_MLEpredictions_randCADEC6_onTest = dense_layer_method(dd_randCADEC_train6_lowercased, dd_randCADEC_test6_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC6_onTest = accuracy(dd_MLEpredictions_randCADEC6_onTest, ddd_randData["AskAPatient.fold-6.test"])
    print("\nMLEscoreRandCADEC6_onTest:", MLEscoreRandCADEC6_onTest)
    dd_MLEpredictions_randCADEC7_onTest = dense_layer_method(dd_randCADEC_train7_lowercased, dd_randCADEC_test7_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC7_onTest = accuracy(dd_MLEpredictions_randCADEC7_onTest, ddd_randData["AskAPatient.fold-7.test"])
    print("\nMLEscoreRandCADEC7_onTest:", MLEscoreRandCADEC7_onTest)
    dd_MLEpredictions_randCADEC8_onTest = dense_layer_method(dd_randCADEC_train8_lowercased, dd_randCADEC_test8_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC8_onTest = accuracy(dd_MLEpredictions_randCADEC8_onTest, ddd_randData["AskAPatient.fold-8.test"])
    print("\nMLEscoreRandCADEC8_onTest:", MLEscoreRandCADEC8_onTest)
    dd_MLEpredictions_randCADEC9_onTest = dense_layer_method(dd_randCADEC_train9_lowercased, dd_randCADEC_test9_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC9_onTest = accuracy(dd_MLEpredictions_randCADEC9_onTest, ddd_randData["AskAPatient.fold-9.test"])
    print("\nMLEscoreRandCADEC9_onTest:", MLEscoreRandCADEC9_onTest)


    dd_predictions_BB4_onVal = dense_layer_method(dd_BB4habTrain_lowercased, dd_BB4habDev_lowercased, dd_habObt_lowercased, word_vectors)
    MLEscore_BB4_onVal = accuracy(dd_predictions_BB4_onVal, dd_habDev)
    print("\n\nBHEMscore_BB4_onVal:", MLEscore_BB4_onVal)


    dd_MLEpredictions_NCBI_onTest = dense_layer_method(dd_NCBITrainFixed_lowercased, dd_NCBITestFixed_lowercased, dd_medic_lowercased, word_vectors)
    MLEscore_NCBI_onTest = accuracy(dd_MLEpredictions_NCBI_onTest, dd_TestFixed)
    print("\nMLEscore_NCBI_onTest (with tags):", MLEscore_NCBI_onTest)
    dd_MLEpredictions_NCBI_onTestWithTrainDev = dense_layer_method(dd_NCBITrainDevFixed_lowercased, dd_NCBITestFixed_lowercased, dd_medic_lowercased, word_vectors)
    MLEscore_NCBI_onTestWithTrainDev = accuracy(dd_MLEpredictions_NCBI_onTestWithTrainDev, dd_TestFixed)
    print("\nMLEscore_NCBI_onTestWithTrainDev (with tags):", MLEscore_NCBI_onTestWithTrainDev)
    """


    #######################
    print("\n\n\nSieve (BHEM->MLE):\n")
    #######################
    """
    word_vectors = KeyedVectors.load_word2vec_format('../PubMed-w2v.bin', binary=True)
    # word_vectors = Word2Vec.load('../VST_count0_size100_iter50.model')


    dd_predictions_customCADEC0_onVal = sieve(dd_customCADEC_train0_lowercased, dd_customCADEC_validation0_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscorecustomCADEC0_onVal = accuracy(dd_predictions_customCADEC0_onVal, ddd_customData["test_0"])
    print("\n\nSIEVEscorecustomCADEC0_onVal:", SIEVEscorecustomCADEC0_onVal)
    dd_predictions_customCADEC1_onVal = sieve(dd_customCADEC_train1_lowercased, dd_customCADEC_validation1_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscorecustomCADEC1_onVal = accuracy(dd_predictions_customCADEC1_onVal, ddd_customData["test_1"])
    print("\nSIEVEscorecustomCADEC1_onVal:", SIEVEscorecustomCADEC1_onVal)
    dd_predictions_customCADEC2_onVal = sieve(dd_customCADEC_train2_lowercased, dd_customCADEC_validation2_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscorecustomCADEC2_onVal = accuracy(dd_predictions_customCADEC2_onVal, ddd_customData["test_2"])
    print("\nSIEVEscorecustomCADEC2_onVal:", SIEVEscorecustomCADEC2_onVal)
    dd_predictions_customCADEC3_onVal = sieve(dd_customCADEC_train3_lowercased, dd_customCADEC_validation3_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscorecustomCADEC3_onVal = accuracy(dd_predictions_customCADEC3_onVal, ddd_customData["test_3"])
    print("\nSIEVEscorecustomCADEC3_onVal:", SIEVEscorecustomCADEC3_onVal)
    dd_predictions_customCADEC4_onVal = sieve(dd_customCADEC_train4_lowercased, dd_customCADEC_validation4_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscorecustomCADEC4_onVal = accuracy(dd_predictions_customCADEC4_onVal, ddd_customData["test_4"])
    print("\nSIEVEscorecustomCADEC4_onVal:", SIEVEscorecustomCADEC4_onVal)


    dd_SIEVEpredictions_randCADEC0_onTest = sieve(dd_randCADEC_train0_lowercased, dd_randCADEC_test0_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscoreRandCADEC0_onTest = accuracy(dd_SIEVEpredictions_randCADEC0_onTest, ddd_randData["AskAPatient.fold-0.test"])
    print("\n\nSIEVEscoreRandCADEC0_onTest:", SIEVEscoreRandCADEC0_onTest)
    dd_SIEVEpredictions_randCADEC1_onTest = sieve(dd_randCADEC_train1_lowercased, dd_randCADEC_test1_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscoreRandCADEC1_onTest = accuracy(dd_SIEVEpredictions_randCADEC1_onTest, ddd_randData["AskAPatient.fold-1.test"])
    print("\nSIEVEscoreRandCADEC1_onTest:", SIEVEscoreRandCADEC1_onTest)
    dd_SIEVEpredictions_randCADEC2_onTest = sieve(dd_randCADEC_train2_lowercased, dd_randCADEC_test2_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscoreRandCADEC2_onTest = accuracy(dd_SIEVEpredictions_randCADEC2_onTest, ddd_randData["AskAPatient.fold-2.test"])
    print("\nSIEVEscoreRandCADEC2_onTest:", SIEVEscoreRandCADEC2_onTest)
    dd_SIEVEpredictions_randCADEC3_onTest = sieve(dd_randCADEC_train3_lowercased, dd_randCADEC_test3_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscoreRandCADEC3_onTest = accuracy(dd_SIEVEpredictions_randCADEC3_onTest, ddd_randData["AskAPatient.fold-3.test"])
    print("\nSIEVEscoreRandCADEC3_onTest:", SIEVEscoreRandCADEC3_onTest)
    dd_SIEVEpredictions_randCADEC4_onTest = sieve(dd_randCADEC_train4_lowercased, dd_randCADEC_test4_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscoreRandCADEC4_onTest = accuracy(dd_SIEVEpredictions_randCADEC4_onTest, ddd_randData["AskAPatient.fold-4.test"])
    print("\nSIEVEscoreRandCADEC4_onTest:", SIEVEscoreRandCADEC4_onTest)
    dd_SIEVEpredictions_randCADEC5_onTest = sieve(dd_randCADEC_train5_lowercased, dd_randCADEC_test5_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscoreRandCADEC5_onTest = accuracy(dd_SIEVEpredictions_randCADEC5_onTest, ddd_randData["AskAPatient.fold-5.test"])
    print("\nSIEVEscoreRandCADEC5_onTest:", SIEVEscoreRandCADEC5_onTest)
    dd_SIEVEpredictions_randCADEC6_onTest = sieve(dd_randCADEC_train6_lowercased, dd_randCADEC_test6_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscoreRandCADEC6_onTest = accuracy(dd_SIEVEpredictions_randCADEC6_onTest, ddd_randData["AskAPatient.fold-6.test"])
    print("\nSIEVEscoreRandCADEC6_onTest:", SIEVEscoreRandCADEC6_onTest)
    dd_SIEVEpredictions_randCADEC7_onTest = sieve(dd_randCADEC_train7_lowercased, dd_randCADEC_test7_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscoreRandCADEC7_onTest = accuracy(dd_SIEVEpredictions_randCADEC7_onTest, ddd_randData["AskAPatient.fold-7.test"])
    print("\nSIEVEscoreRandCADEC7_onTest:", SIEVEscoreRandCADEC7_onTest)
    dd_SIEVEpredictions_randCADEC8_onTest = sieve(dd_randCADEC_train8_lowercased, dd_randCADEC_test8_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscoreRandCADEC8_onTest = accuracy(dd_SIEVEpredictions_randCADEC8_onTest, ddd_randData["AskAPatient.fold-8.test"])
    print("\nSIEVEscoreRandCADEC8_onTest:", SIEVEscoreRandCADEC8_onTest)
    dd_SIEVEpredictions_randCADEC9_onTest = sieve(dd_randCADEC_train9_lowercased, dd_randCADEC_test9_lowercased, dd_subsubRef_lowercased, word_vectors)
    SIEVEscoreRandCADEC9_onTest = accuracy(dd_SIEVEpredictions_randCADEC9_onTest, ddd_randData["AskAPatient.fold-9.test"])
    print("\nSIEVEscoreRandCADEC9_onTest:", SIEVEscoreRandCADEC9_onTest)

    dd_predictions_BB4_onVal = sieve(dd_BB4habTrain_lowercased, dd_BB4habDev_lowercased, dd_habObt_lowercased, word_vectors)
    SIEVEscore_BB4_onVal = accuracy(dd_predictions_BB4_onVal, dd_habDev)
    print("\n\nBHEMscore_BB4_onVal:", SIEVEscore_BB4_onVal)

    dd_SIEVEpredictions_NCBI_onTest = sieve(dd_NCBITrainFixed_lowercased, dd_NCBITestFixed_lowercased, dd_medic_lowercased, word_vectors)
    SIEVEscore_NCBI_onTest = accuracy(dd_SIEVEpredictions_NCBI_onTest, dd_TestFixed)
    print("\nSIEVEscore_NCBI_onTest (with tags):", SIEVEscore_NCBI_onTest)
    dd_SIEVEpredictions_NCBI_onTestWithTrainDev = sieve(dd_NCBITrainDevFixed_lowercased, dd_NCBITestFixed_lowercased, dd_medic_lowercased, word_vectors)
    SIEVEscore_NCBI_onTestWithTrainDev = accuracy(dd_SIEVEpredictions_NCBI_onTestWithTrainDev, dd_TestFixed)
    print("\nSIEVEscore_NCBI_onTestWithTrainDev (with tags):", SIEVEscore_NCBI_onTestWithTrainDev)
    """


    #######################
    print("\n\n\nStatic distance between label/mention embeddings (only local ref):\n")
    #######################
    """
    from loaders import get_cuis_set_from_corpus, get_subref_from_cui_set

    print("\n\nLoading cuis set in corpus...")
    s_cuisInRandCadec = get_cuis_set_from_corpus(dd_randCadec)
    dd_localRandCadecRef = get_subref_from_cui_set(s_cuisInRandCadec, dd_subsubRef_lowercased)
    s_cuisInCustomCadec = get_cuis_set_from_corpus(dd_customCadec)
    dd_localCustomCadecRef = get_subref_from_cui_set(s_cuisInCustomCadec, dd_subsubRef_lowercased)
    print("Loaded.(Nb of distinct used concepts in rand/custom =", len(s_cuisInRandCadec), len(s_cuisInCustomCadec),")")
    print("Nb concepts in local ref (custom/rand):", len(dd_localCustomCadecRef.keys()), "/", len(dd_localRandCadecRef.keys()))


    print("\nLoading cuis set in corpus...")
    s_cuisHabTrain = get_cuis_set_from_corpus(dd_habTrain)
    s_cuisHabDev = get_cuis_set_from_corpus(dd_habDev)
    s_cuisHabTrainDev = get_cuis_set_from_corpus(dd_habTrainDev)
    dd_localBB4HabTrainDevRef = get_subref_from_cui_set(s_cuisHabTrainDev, dd_habObt_lowercased)
    dd_localBB4HabDevRef = get_subref_from_cui_set(s_cuisHabDev, dd_habObt_lowercased)
    print("Loaded.(Nb of distinct used concepts in train/dev/train+dev hab corpora =", len(s_cuisHabTrain),len(s_cuisHabDev),len(s_cuisHabTrainDev),")")
    print("Nb concepts in local ref (train+dev):", len(dd_localBB4HabTrainDevRef.keys()))
    print("Nb concepts in local ref (dev):", len(dd_localBB4HabDevRef.keys()))


    print("\nLoading cuis set in corpus...")
    s_cuisNCBIFull = get_cuis_set_from_corpus(dd_FullFixed)
    dd_localMedicFull = get_subref_from_cui_set(s_cuisNCBIFull, dd_medic_lowercased)
    s_cuisNCBITrain = get_cuis_set_from_corpus(dd_TrainFixed)
    s_cuisNCBIDev = get_cuis_set_from_corpus(dd_DevFixed)
    dd_localMedicDev = get_subref_from_cui_set(s_cuisNCBIDev, dd_medic_lowercased)
    s_cuisNCBITrainDev = get_cuis_set_from_corpus(dd_TrainDevFixed)
    s_cuisNCBITest = get_cuis_set_from_corpus(dd_TestFixed)
    dd_localMedicTest = get_subref_from_cui_set(s_cuisNCBITest, dd_medic_lowercased)
    print("Loaded.(Nb of distinct used concepts in Full/train/dev/train+dev/test NCBI folds =", len(s_cuisNCBIFull),len(s_cuisNCBITrain),len(s_cuisNCBIDev),len(s_cuisNCBITrainDev),len(s_cuisNCBITest),")")
    print("Nb concepts in local ref (full):", len(dd_localMedicFull.keys()))
    print("Nb concepts in local ref (dev):", len(dd_localMedicDev.keys()))
    print("Nb concepts in local ref (test):", len(dd_localMedicTest.keys()))


    word_vectors = KeyedVectors.load_word2vec_format('../PubMed-w2v.bin', binary=True)
    #word_vectors = Word2Vec.load('../VST_count0_size100_iter50.model')


    dd_predictions_customCADEC0_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation0_lowercased, dd_localCustomCadecRef, word_vectors)
    SEscorecustomCADEC0_onVal = accuracy(dd_predictions_customCADEC0_onVal, ddd_customData["test_0"])
    print("\n\nSEscorecustomCADEC0_onVal:", SEscorecustomCADEC0_onVal)
    dd_predictions_customCADEC1_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation1_lowercased, dd_localCustomCadecRef, word_vectors)
    SEscorecustomCADEC1_onVal = accuracy(dd_predictions_customCADEC1_onVal, ddd_customData["test_1"])
    print("\nSEscorecustomCADEC1_onVal:", SEscorecustomCADEC1_onVal)
    dd_predictions_customCADEC2_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation2_lowercased, dd_localCustomCadecRef, word_vectors)
    SEscorecustomCADEC2_onVal = accuracy(dd_predictions_customCADEC2_onVal, ddd_customData["test_2"])
    print("\nSEscorecustomCADEC2_onVal:", SEscorecustomCADEC2_onVal)
    dd_predictions_customCADEC3_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation3_lowercased, dd_localCustomCadecRef, word_vectors)
    SEscorecustomCADEC3_onVal = accuracy(dd_predictions_customCADEC3_onVal, ddd_customData["test_3"])
    print("\nSEscorecustomCADEC3_onVal:", SEscorecustomCADEC3_onVal)
    dd_predictions_customCADEC4_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation4_lowercased, dd_localCustomCadecRef, word_vectors)
    SEscorecustomCADEC4_onVal = accuracy(dd_predictions_customCADEC4_onVal, ddd_customData["test_4"])
    print("\nSEscorecustomCADEC4_onVal:", SEscorecustomCADEC4_onVal)


    dd_SEpredictions_randCADEC0_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test0_lowercased, dd_localRandCadecRef, word_vectors)
    SEscoreRandCADEC0_onTest = accuracy(dd_SEpredictions_randCADEC0_onTest, ddd_randData["AskAPatient.fold-0.test"])
    print("\n\nSEscoreRandCADEC0_onTest:", SEscoreRandCADEC0_onTest)
    dd_SEpredictions_randCADEC1_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test1_lowercased, dd_localRandCadecRef, word_vectors)
    SEscoreRandCADEC1_onTest = accuracy(dd_SEpredictions_randCADEC1_onTest, ddd_randData["AskAPatient.fold-1.test"])
    print("\nSEscoreRandCADEC1_onTest:", SEscoreRandCADEC1_onTest)
    dd_SEpredictions_randCADEC2_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test2_lowercased, dd_localRandCadecRef, word_vectors)
    SEscoreRandCADEC2_onTest = accuracy(dd_SEpredictions_randCADEC2_onTest, ddd_randData["AskAPatient.fold-2.test"])
    print("\nSEscoreRandCADEC2_onTest:", SEscoreRandCADEC2_onTest)
    dd_SEpredictions_randCADEC3_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test3_lowercased, dd_localRandCadecRef, word_vectors)
    SEscoreRandCADEC3_onTest = accuracy(dd_SEpredictions_randCADEC3_onTest, ddd_randData["AskAPatient.fold-3.test"])
    print("\nSEscoreRandCADEC3_onTest:", SEscoreRandCADEC3_onTest)
    dd_SEpredictions_randCADEC4_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test4_lowercased, dd_localRandCadecRef, word_vectors)
    SEscoreRandCADEC4_onTest = accuracy(dd_SEpredictions_randCADEC4_onTest, ddd_randData["AskAPatient.fold-4.test"])
    print("\nSEscoreRandCADEC4_onTest:", SEscoreRandCADEC4_onTest)
    dd_SEpredictions_randCADEC5_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test5_lowercased, dd_localRandCadecRef, word_vectors)
    SEscoreRandCADEC5_onTest = accuracy(dd_SEpredictions_randCADEC5_onTest, ddd_randData["AskAPatient.fold-5.test"])
    print("\nSEscoreRandCADEC5_onTest:", SEscoreRandCADEC5_onTest)
    dd_SEpredictions_randCADEC6_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test6_lowercased, dd_localRandCadecRef, word_vectors)
    SEscoreRandCADEC6_onTest = accuracy(dd_SEpredictions_randCADEC6_onTest, ddd_randData["AskAPatient.fold-6.test"])
    print("\nSEscoreRandCADEC6_onTest:", SEscoreRandCADEC6_onTest)
    dd_SEpredictions_randCADEC7_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test7_lowercased, dd_localRandCadecRef, word_vectors)
    SEscoreRandCADEC7_onTest = accuracy(dd_SEpredictions_randCADEC7_onTest, ddd_randData["AskAPatient.fold-7.test"])
    print("\nSEscoreRandCADEC7_onTest:", SEscoreRandCADEC7_onTest)
    dd_SEpredictions_randCADEC8_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test8_lowercased, dd_localRandCadecRef, word_vectors)
    SEscoreRandCADEC8_onTest = accuracy(dd_SEpredictions_randCADEC8_onTest, ddd_randData["AskAPatient.fold-8.test"])
    print("\nSEscoreRandCADEC8_onTest:", SEscoreRandCADEC8_onTest)
    dd_SEpredictions_randCADEC9_onTest = embeddings_similarity_method_with_tags(dd_randCADEC_test9_lowercased, dd_localRandCadecRef, word_vectors)
    SEscoreRandCADEC9_onTest = accuracy(dd_SEpredictions_randCADEC9_onTest, ddd_randData["AskAPatient.fold-9.test"])
    print("\nSEscoreRandCADEC9_onTest:", SEscoreRandCADEC9_onTest)


    dd_predictions_BB4_onVal = embeddings_similarity_method_with_tags(dd_BB4habDev_lowercased, dd_localBB4HabTrainDevRef, word_vectors)
    SEscore_BB4_onVal = accuracy(dd_predictions_BB4_onVal, dd_habDev)
    print("\n\nSEscore_BB4_onVal (local full=train+dev):", SEscore_BB4_onVal)
    dd_predictions_BB4_onVal = embeddings_similarity_method_with_tags(dd_BB4habDev_lowercased, dd_localBB4HabDevRef, word_vectors)
    SEscore_BB4_onVal = accuracy(dd_predictions_BB4_onVal, dd_habDev)
    print("\n\nSEscore_BB4_onVal (local dev):", SEscore_BB4_onVal)


    dd_EMpredictions_NCBI_onVal = embeddings_similarity_method_with_tags(dd_NCBIDevFixed_lowercased, dd_localMedicFull, word_vectors)
    SEscore_NCBI_onVal = accuracy(dd_EMpredictions_NCBI_onVal, dd_NCBIDevFixed_lowercased)
    print("\nSEscore_NCBI_onVal (with tags):", SEscore_NCBI_onVal)
    dd_EMpredictions_NCBI_onTest = embeddings_similarity_method_with_tags(dd_NCBITestFixed_lowercased, dd_localMedicFull, word_vectors)
    SEscore_NCBI_onTest = accuracy(dd_EMpredictions_NCBI_onTest, dd_TestFixed)
    print("\nSEscore_NCBI_onTest (with tags):", SEscore_NCBI_onTest)
    dd_EMpredictions_NCBI_onVal = embeddings_similarity_method_with_tags(dd_NCBIDevFixed_lowercased, dd_localMedicDev, word_vectors)
    SEscore_NCBI_onVal = accuracy(dd_EMpredictions_NCBI_onVal, dd_NCBIDevFixed_lowercased)
    print("\nSEscore_NCBI_onVal (with tags - only dev CUIs):", SEscore_NCBI_onVal)
    dd_EMpredictions_NCBI_onTest = embeddings_similarity_method_with_tags(dd_NCBITestFixed_lowercased, dd_localMedicTest, word_vectors)
    SEscore_NCBI_onTest = accuracy(dd_EMpredictions_NCBI_onTest, dd_TestFixed)
    print("\nSEscore_NCBI_onTest (with tags - only test CUIs):", SEscore_NCBI_onTest)

    """

    #######################
    print("\n\n\nML distance between label/mention embeddings (only local ref):\n")
    #######################
    """
    from loaders import get_cuis_set_from_corpus, get_subref_from_cui_set

    print("\n\nLoading cuis set in corpus...")
    s_cuisInRandCadec = get_cuis_set_from_corpus(dd_randCadec)
    dd_localRandCadecRef = get_subref_from_cui_set(s_cuisInRandCadec, dd_subsubRef_lowercased)
    s_cuisInCustomCadec = get_cuis_set_from_corpus(dd_customCadec)
    dd_localCustomCadecRef = get_subref_from_cui_set(s_cuisInCustomCadec, dd_subsubRef_lowercased)
    print("Loaded.(Nb of distinct used concepts in rand/custom =", len(s_cuisInRandCadec), len(s_cuisInCustomCadec), ")")
    print("Nb concepts in local ref (custom/rand):", len(dd_localCustomCadecRef.keys()), "/", len(dd_localRandCadecRef.keys()))

    print("\nLoading cuis set in corpus...")
    s_cuisHabTrain = get_cuis_set_from_corpus(dd_habTrain)
    s_cuisHabDev = get_cuis_set_from_corpus(dd_habDev)
    s_cuisHabTrainDev = get_cuis_set_from_corpus(dd_habTrainDev)
    dd_localBB4HabTrainDevRef = get_subref_from_cui_set(s_cuisHabTrainDev, dd_habObt_lowercased)
    dd_localBB4HabDevRef = get_subref_from_cui_set(s_cuisHabDev, dd_habObt_lowercased)
    print("Loaded.(Nb of distinct used concepts in train/dev/train+dev hab corpora =", len(s_cuisHabTrain), len(s_cuisHabDev), len(s_cuisHabTrainDev), ")")
    print("Nb concepts in local ref (train+dev):", len(dd_localBB4HabTrainDevRef.keys()))
    print("Nb concepts in local ref (dev):", len(dd_localBB4HabDevRef.keys()))

    print("\nLoading cuis set in corpus...")
    s_cuisNCBIFull = get_cuis_set_from_corpus(dd_FullFixed)
    dd_localMedicFull = get_subref_from_cui_set(s_cuisNCBIFull, dd_medic_lowercased)
    s_cuisNCBITrain = get_cuis_set_from_corpus(dd_TrainFixed)
    s_cuisNCBIDev = get_cuis_set_from_corpus(dd_DevFixed)
    dd_localMedicDev = get_subref_from_cui_set(s_cuisNCBIDev, dd_medic_lowercased)
    s_cuisNCBITrainDev = get_cuis_set_from_corpus(dd_TrainDevFixed)
    s_cuisNCBITest = get_cuis_set_from_corpus(dd_TestFixed)
    dd_localMedicTest = get_subref_from_cui_set(s_cuisNCBITest, dd_medic_lowercased)
    print("Loaded.(Nb of distinct used concepts in Full/train/dev/train+dev/test NCBI folds =", len(s_cuisNCBIFull), len(s_cuisNCBITrain), len(s_cuisNCBIDev), len(s_cuisNCBITrainDev), len(s_cuisNCBITest), ")")
    print("Nb concepts in local ref (full):", len(dd_localMedicFull.keys()))
    print("Nb concepts in local ref (dev):", len(dd_localMedicDev.keys()))
    print("Nb concepts in local ref (test):", len(dd_localMedicTest.keys()))



    word_vectors = KeyedVectors.load_word2vec_format('../PubMed-w2v.bin', binary=True)
    # word_vectors = Word2Vec.load('../VST_count0_size100_iter50.model')


    dd_predictions_customCADEC0_onVal = dense_layer_method(dd_customCADEC_train0_lowercased, dd_customCADEC_validation0_lowercased, dd_localCustomCadecRef, word_vectors)
    MLEscorecustomCADEC0_onVal = accuracy(dd_predictions_customCADEC0_onVal, ddd_customData["test_0"])
    print("\n\nMLEscorecustomCADEC0_onVal:", MLEscorecustomCADEC0_onVal)
    dd_predictions_customCADEC1_onVal = dense_layer_method(dd_customCADEC_train1_lowercased, dd_customCADEC_validation1_lowercased, dd_localCustomCadecRef, word_vectors)
    MLEscorecustomCADEC1_onVal = accuracy(dd_predictions_customCADEC1_onVal, ddd_customData["test_1"])
    print("\nMLEscorecustomCADEC1_onVal:", MLEscorecustomCADEC1_onVal)
    dd_predictions_customCADEC2_onVal = dense_layer_method(dd_customCADEC_train2_lowercased, dd_customCADEC_validation2_lowercased, dd_localCustomCadecRef, word_vectors)
    MLEscorecustomCADEC2_onVal = accuracy(dd_predictions_customCADEC2_onVal, ddd_customData["test_2"])
    print("\nMLEscorecustomCADEC2_onVal:", MLEscorecustomCADEC2_onVal)
    dd_predictions_customCADEC3_onVal = dense_layer_method(dd_customCADEC_train3_lowercased, dd_customCADEC_validation3_lowercased, dd_localCustomCadecRef, word_vectors)
    MLEscorecustomCADEC3_onVal = accuracy(dd_predictions_customCADEC3_onVal, ddd_customData["test_3"])
    print("\nMLEscorecustomCADEC3_onVal:", MLEscorecustomCADEC3_onVal)
    dd_predictions_customCADEC4_onVal = dense_layer_method(dd_customCADEC_train4_lowercased, dd_customCADEC_validation4_lowercased, dd_localCustomCadecRef, word_vectors)
    MLEscorecustomCADEC4_onVal = accuracy(dd_predictions_customCADEC4_onVal, ddd_customData["test_4"])
    print("\nMLEscorecustomCADEC4_onVal:", MLEscorecustomCADEC4_onVal)

    dd_MLEpredictions_randCADEC0_onTest = dense_layer_method(dd_randCADEC_train0_lowercased, dd_randCADEC_test0_lowercased, dd_localRandCadecRef, word_vectors)
    MLEscoreRandCADEC0_onTest = accuracy(dd_MLEpredictions_randCADEC0_onTest, ddd_randData["AskAPatient.fold-0.test"])
    print("\n\nMLEscoreRandCADEC0_onTest:", MLEscoreRandCADEC0_onTest)
    dd_MLEpredictions_randCADEC1_onTest = dense_layer_method(dd_randCADEC_train1_lowercased, dd_randCADEC_test1_lowercased, dd_localRandCadecRef, word_vectors)
    MLEscoreRandCADEC1_onTest = accuracy(dd_MLEpredictions_randCADEC1_onTest, ddd_randData["AskAPatient.fold-1.test"])
    print("\nMLEscoreRandCADEC1_onTest:", MLEscoreRandCADEC1_onTest)
    dd_MLEpredictions_randCADEC2_onTest = dense_layer_method(dd_randCADEC_train2_lowercased, dd_randCADEC_test2_lowercased, dd_localRandCadecRef, word_vectors)
    MLEscoreRandCADEC2_onTest = accuracy(dd_MLEpredictions_randCADEC2_onTest, ddd_randData["AskAPatient.fold-2.test"])
    print("\nMLEscoreRandCADEC2_onTest:", MLEscoreRandCADEC2_onTest)
    dd_MLEpredictions_randCADEC3_onTest = dense_layer_method(dd_randCADEC_train3_lowercased, dd_randCADEC_test3_lowercased, dd_localRandCadecRef, word_vectors)
    MLEscoreRandCADEC3_onTest = accuracy(dd_MLEpredictions_randCADEC3_onTest, ddd_randData["AskAPatient.fold-3.test"])
    print("\nMLEscoreRandCADEC3_onTest:", MLEscoreRandCADEC3_onTest)
    dd_MLEpredictions_randCADEC4_onTest = dense_layer_method(dd_randCADEC_train4_lowercased, dd_randCADEC_test4_lowercased, dd_localRandCadecRef, word_vectors)
    MLEscoreRandCADEC4_onTest = accuracy(dd_MLEpredictions_randCADEC4_onTest, ddd_randData["AskAPatient.fold-4.test"])
    print("\nMLEscoreRandCADEC4_onTest:", MLEscoreRandCADEC4_onTest)
    dd_MLEpredictions_randCADEC5_onTest = dense_layer_method(dd_randCADEC_train5_lowercased, dd_randCADEC_test5_lowercased, dd_localRandCadecRef, word_vectors)
    MLEscoreRandCADEC5_onTest = accuracy(dd_MLEpredictions_randCADEC5_onTest, ddd_randData["AskAPatient.fold-5.test"])
    print("\nMLEscoreRandCADEC5_onTest:", MLEscoreRandCADEC5_onTest)
    dd_MLEpredictions_randCADEC6_onTest = dense_layer_method(dd_randCADEC_train6_lowercased, dd_randCADEC_test6_lowercased, dd_localRandCadecRef, word_vectors)
    MLEscoreRandCADEC6_onTest = accuracy(dd_MLEpredictions_randCADEC6_onTest, ddd_randData["AskAPatient.fold-6.test"])
    print("\nMLEscoreRandCADEC6_onTest:", MLEscoreRandCADEC6_onTest)
    dd_MLEpredictions_randCADEC7_onTest = dense_layer_method(dd_randCADEC_train7_lowercased, dd_randCADEC_test7_lowercased, dd_localRandCadecRef, word_vectors)
    MLEscoreRandCADEC7_onTest = accuracy(dd_MLEpredictions_randCADEC7_onTest, ddd_randData["AskAPatient.fold-7.test"])
    print("\nMLEscoreRandCADEC7_onTest:", MLEscoreRandCADEC7_onTest)
    dd_MLEpredictions_randCADEC8_onTest = dense_layer_method(dd_randCADEC_train8_lowercased, dd_randCADEC_test8_lowercased, dd_localRandCadecRef, word_vectors)
    MLEscoreRandCADEC8_onTest = accuracy(dd_MLEpredictions_randCADEC8_onTest, ddd_randData["AskAPatient.fold-8.test"])
    print("\nMLEscoreRandCADEC8_onTest:", MLEscoreRandCADEC8_onTest)
    dd_MLEpredictions_randCADEC9_onTest = dense_layer_method(dd_randCADEC_train9_lowercased, dd_randCADEC_test9_lowercased, dd_localRandCadecRef, word_vectors)
    MLEscoreRandCADEC9_onTest = accuracy(dd_MLEpredictions_randCADEC9_onTest, ddd_randData["AskAPatient.fold-9.test"])
    print("\nMLEscoreRandCADEC9_onTest:", MLEscoreRandCADEC9_onTest)

    dd_predictions_BB4_onVal = dense_layer_method(dd_BB4habTrain_lowercased, dd_BB4habDev_lowercased, dd_localBB4HabTrainDevRef, word_vectors)
    MLEscore_BB4_onVal = accuracy(dd_predictions_BB4_onVal, dd_habDev)
    print("\n\nMLEscore_BB4_onVal:", MLEscore_BB4_onVal)
    dd_predictions_BB4_onVal = dense_layer_method(dd_BB4habTrain_lowercased, dd_BB4habDev_lowercased, dd_localBB4HabDevRef, word_vectors)
    MLEscore_BB4_onVal = accuracy(dd_predictions_BB4_onVal, dd_habDev)
    print("\n\nMLEscore_BB4_onVal:", MLEscore_BB4_onVal)

    dd_MLEpredictions_NCBI_onTest = dense_layer_method(dd_NCBITrainFixed_lowercased, dd_NCBITestFixed_lowercased, dd_localMedicFull, word_vectors)
    MLEscore_NCBI_onTest = accuracy(dd_MLEpredictions_NCBI_onTest, dd_TestFixed)
    print("\nMLEscore_NCBI_onTest (with tags):", MLEscore_NCBI_onTest)
    dd_MLEpredictions_NCBI_onTestWithTrainDev = dense_layer_method(dd_NCBITrainDevFixed_lowercased, dd_NCBITestFixed_lowercased, dd_localMedicFull, word_vectors)
    MLEscore_NCBI_onTestWithTrainDev = accuracy(dd_MLEpredictions_NCBI_onTestWithTrainDev, dd_TestFixed)
    print("\nMLEscore_NCBI_onTestWithTrainDev (with tags):", MLEscore_NCBI_onTestWithTrainDev)
    dd_MLEpredictions_NCBI_onTest = dense_layer_method(dd_NCBITrainFixed_lowercased, dd_NCBITestFixed_lowercased, dd_localMedicTest, word_vectors)
    MLEscore_NCBI_onTest = accuracy(dd_MLEpredictions_NCBI_onTest, dd_TestFixed)
    print("\nMLEscore_NCBI_onTest (with tags):", MLEscore_NCBI_onTest)
    dd_MLEpredictions_NCBI_onTestWithTrainDev = dense_layer_method(dd_NCBITrainDevFixed_lowercased, dd_NCBITestFixed_lowercased, dd_localMedicTest, word_vectors)
    MLEscore_NCBI_onTestWithTrainDev = accuracy(dd_MLEpredictions_NCBI_onTestWithTrainDev, dd_TestFixed)
    print("\nMLEscore_NCBI_onTestWithTrainDev (with tags):", MLEscore_NCBI_onTestWithTrainDev)
    """


    #######################
    print("\n\n\nML distance between label/mention embeddings (only local ref):\n")
    #######################
    """
    from loaders import get_cuis_set_from_corpus, get_subref_from_cui_set

    print("\n\nLoading cuis set in corpus...")
    s_cuisInRandCadec = get_cuis_set_from_corpus(dd_randCadec)
    dd_localRandCadecRef = get_subref_from_cui_set(s_cuisInRandCadec, dd_subsubRef_lowercased)
    s_cuisInCustomCadec = get_cuis_set_from_corpus(dd_customCadec)
    dd_localCustomCadecRef = get_subref_from_cui_set(s_cuisInCustomCadec, dd_subsubRef_lowercased)
    print("Loaded.(Nb of distinct used concepts in rand/custom =", len(s_cuisInRandCadec), len(s_cuisInCustomCadec),")")
    print("Nb concepts in local ref (custom/rand):", len(dd_localCustomCadecRef.keys()), "/", len(dd_localRandCadecRef.keys()))

    print("\nLoading cuis set in corpus...")
    s_cuisHabTrain = get_cuis_set_from_corpus(dd_habTrain)
    s_cuisHabDev = get_cuis_set_from_corpus(dd_habDev)
    s_cuisHabTrainDev = get_cuis_set_from_corpus(dd_habTrainDev)
    dd_localBB4HabTrainDevRef = get_subref_from_cui_set(s_cuisHabTrainDev, dd_habObt_lowercased)
    dd_localBB4HabDevRef = get_subref_from_cui_set(s_cuisHabDev, dd_habObt_lowercased)
    print("Loaded.(Nb of distinct used concepts in train/dev/train+dev hab corpora =", len(s_cuisHabTrain), len(s_cuisHabDev), len(s_cuisHabTrainDev), ")")
    print("Nb concepts in local ref (train+dev):", len(dd_localBB4HabTrainDevRef.keys()))
    print("Nb concepts in local ref (dev):", len(dd_localBB4HabDevRef.keys()))

    print("\nLoading cuis set in corpus...")
    s_cuisNCBIFull = get_cuis_set_from_corpus(dd_FullFixed)
    dd_localMedicFull = get_subref_from_cui_set(s_cuisNCBIFull, dd_medic_lowercased)
    s_cuisNCBITrain = get_cuis_set_from_corpus(dd_TrainFixed)
    s_cuisNCBIDev = get_cuis_set_from_corpus(dd_DevFixed)
    dd_localMedicDev = get_subref_from_cui_set(s_cuisNCBIDev, dd_medic_lowercased)
    s_cuisNCBITrainDev = get_cuis_set_from_corpus(dd_TrainDevFixed)
    s_cuisNCBITest = get_cuis_set_from_corpus(dd_TestFixed)
    dd_localMedicTest = get_subref_from_cui_set(s_cuisNCBITest, dd_medic_lowercased)
    print("Loaded.(Nb of distinct used concepts in Full/train/dev/train+dev/test NCBI folds =", len(s_cuisNCBIFull), len(s_cuisNCBITrain), len(s_cuisNCBIDev), len(s_cuisNCBITrainDev), len(s_cuisNCBITest), ")")
    print("Nb concepts in local ref (full):", len(dd_localMedicFull.keys()))
    print("Nb concepts in local ref (dev):", len(dd_localMedicDev.keys()))
    print("Nb concepts in local ref (test):", len(dd_localMedicTest.keys()))

    word_vectors = KeyedVectors.load_word2vec_format('../PubMed-w2v.bin', binary=True)
    # word_vectors = Word2Vec.load('../VST_count0_size100_iter50.model')


    dd_predictions_customLocalCADEC0_onVal = sieve(dd_customCADEC_train0_lowercased, dd_customCADEC_validation0_lowercased, dd_localCustomCadecRef, word_vectors)
    SIEVEscorecustomLocalCADEC0_onVal = accuracy(dd_predictions_customLocalCADEC0_onVal, ddd_customData["test_0"])
    print("\n\nSIEVEscorecustomLocalCADEC0_onVal:", SIEVEscorecustomLocalCADEC0_onVal)
    dd_predictions_customLocalCADEC1_onVal = sieve(dd_customCADEC_train1_lowercased, dd_customCADEC_validation1_lowercased, dd_localCustomCadecRef, word_vectors)
    SIEVEscorecustomLocalCADEC1_onVal = accuracy(dd_predictions_customLocalCADEC1_onVal, ddd_customData["test_1"])
    print("\nSIEVEscorecustomLocalCADEC1_onVal:", SIEVEscorecustomLocalCADEC1_onVal)
    dd_predictions_customLocalCADEC2_onVal = sieve(dd_customCADEC_train2_lowercased, dd_customCADEC_validation2_lowercased, dd_localCustomCadecRef, word_vectors)
    SIEVEscorecustomLocalCADEC2_onVal = accuracy(dd_predictions_customLocalCADEC2_onVal, ddd_customData["test_2"])
    print("\nSIEVEscorecustomLocalCADEC2_onVal:", SIEVEscorecustomLocalCADEC2_onVal)
    dd_predictions_customLocalCADEC3_onVal = sieve(dd_customCADEC_train3_lowercased, dd_customCADEC_validation3_lowercased, dd_localCustomCadecRef, word_vectors)
    SIEVEscorecustomLocalCADEC3_onVal = accuracy(dd_predictions_customLocalCADEC3_onVal, ddd_customData["test_3"])
    print("\nSIEVEscorecustomLocalCADEC3_onVal:", SIEVEscorecustomLocalCADEC3_onVal)
    dd_predictions_customLocalCADEC4_onVal = sieve(dd_customCADEC_train4_lowercased, dd_customCADEC_validation4_lowercased, dd_localCustomCadecRef, word_vectors)
    SIEVEscorecustomLocalCADEC4_onVal = accuracy(dd_predictions_customLocalCADEC4_onVal, ddd_customData["test_4"])
    print("\nSIEVEscorecustomLocalCADEC4_onVal:", SIEVEscorecustomLocalCADEC4_onVal)


    dd_LocalSIEVEpredictions_randCADEC0_onTest = sieve(dd_randCADEC_train0_lowercased, dd_randCADEC_test0_lowercased, dd_localRandCadecRef, word_vectors)
    LocalSIEVEscoreRandCADEC0_onTest = accuracy(dd_LocalSIEVEpredictions_randCADEC0_onTest, ddd_randData["AskAPatient.fold-0.test"])
    print("\n\nLocalSIEVEscoreRandCADEC0_onTest:", LocalSIEVEscoreRandCADEC0_onTest)
    dd_LocalSIEVEpredictions_randCADEC1_onTest = sieve(dd_randCADEC_train1_lowercased, dd_randCADEC_test1_lowercased, dd_localRandCadecRef, word_vectors)
    LocalSIEVEscoreRandCADEC1_onTest = accuracy(dd_LocalSIEVEpredictions_randCADEC1_onTest, ddd_randData["AskAPatient.fold-1.test"])
    print("\nLocalSIEVEscoreRandCADEC1_onTest:", LocalSIEVEscoreRandCADEC1_onTest)
    dd_LocalSIEVEpredictions_randCADEC2_onTest = sieve(dd_randCADEC_train2_lowercased, dd_randCADEC_test2_lowercased, dd_localRandCadecRef, word_vectors)
    LocalSIEVEscoreRandCADEC2_onTest = accuracy(dd_LocalSIEVEpredictions_randCADEC2_onTest, ddd_randData["AskAPatient.fold-2.test"])
    print("\nLocalSIEVEscoreRandCADEC2_onTest:", LocalSIEVEscoreRandCADEC2_onTest)
    dd_LocalSIEVEpredictions_randCADEC3_onTest = sieve(dd_randCADEC_train3_lowercased, dd_randCADEC_test3_lowercased, dd_localRandCadecRef, word_vectors)
    LocalSIEVEscoreRandCADEC3_onTest = accuracy(dd_LocalSIEVEpredictions_randCADEC3_onTest, ddd_randData["AskAPatient.fold-3.test"])
    print("\nLocalSIEVEscoreRandCADEC3_onTest:", LocalSIEVEscoreRandCADEC3_onTest)
    dd_LocalSIEVEpredictions_randCADEC4_onTest = sieve(dd_randCADEC_train4_lowercased, dd_randCADEC_test4_lowercased, dd_localRandCadecRef, word_vectors)
    LocalSIEVEscoreRandCADEC4_onTest = accuracy(dd_LocalSIEVEpredictions_randCADEC4_onTest, ddd_randData["AskAPatient.fold-4.test"])
    print("\nLocalSIEVEscoreRandCADEC4_onTest:", LocalSIEVEscoreRandCADEC4_onTest)
    dd_LocalSIEVEpredictions_randCADEC5_onTest = sieve(dd_randCADEC_train5_lowercased, dd_randCADEC_test5_lowercased, dd_localRandCadecRef, word_vectors)
    LocalSIEVEscoreRandCADEC5_onTest = accuracy(dd_LocalSIEVEpredictions_randCADEC5_onTest, ddd_randData["AskAPatient.fold-5.test"])
    print("\nLocalSIEVEscoreRandCADEC5_onTest:", LocalSIEVEscoreRandCADEC5_onTest)
    dd_LocalSIEVEpredictions_randCADEC6_onTest = sieve(dd_randCADEC_train6_lowercased, dd_randCADEC_test6_lowercased, dd_localRandCadecRef, word_vectors)
    LocalSIEVEscoreRandCADEC6_onTest = accuracy(dd_LocalSIEVEpredictions_randCADEC6_onTest, ddd_randData["AskAPatient.fold-6.test"])
    print("\nLocalSIEVEscoreRandCADEC6_onTest:", LocalSIEVEscoreRandCADEC6_onTest)
    dd_LocalSIEVEpredictions_randCADEC7_onTest = sieve(dd_randCADEC_train7_lowercased, dd_randCADEC_test7_lowercased, dd_localRandCadecRef, word_vectors)
    LocalSIEVEscoreRandCADEC7_onTest = accuracy(dd_LocalSIEVEpredictions_randCADEC7_onTest, ddd_randData["AskAPatient.fold-7.test"])
    print("\nLocalSIEVEscoreRandCADEC7_onTest:", LocalSIEVEscoreRandCADEC7_onTest)
    dd_LocalSIEVEpredictions_randCADEC8_onTest = sieve(dd_randCADEC_train8_lowercased, dd_randCADEC_test8_lowercased, dd_localRandCadecRef, word_vectors)
    LocalSIEVEscoreRandCADEC8_onTest = accuracy(dd_LocalSIEVEpredictions_randCADEC8_onTest, ddd_randData["AskAPatient.fold-8.test"])
    print("\nLocalSIEVEscoreRandCADEC8_onTest:", LocalSIEVEscoreRandCADEC8_onTest)
    dd_LocalSIEVEpredictions_randCADEC9_onTest = sieve(dd_randCADEC_train9_lowercased, dd_randCADEC_test9_lowercased, dd_localRandCadecRef, word_vectors)
    LocalSIEVEscoreRandCADEC9_onTest = accuracy(dd_LocalSIEVEpredictions_randCADEC9_onTest, ddd_randData["AskAPatient.fold-9.test"])
    print("\nLocalSIEVEscoreRandCADEC9_onTest:", LocalSIEVEscoreRandCADEC9_onTest)


    dd_predictions_BB4_onVal = sieve(dd_BB4habTrain_lowercased, dd_BB4habDev_lowercased, dd_localBB4HabTrainDevRef, word_vectors)
    LocalSIEVEscore_BB4_onVal = accuracy(dd_predictions_BB4_onVal, dd_habDev)
    print("\n\nLocalSIEVEscore_BB4_onVal:", LocalSIEVEscore_BB4_onVal)
    dd_predictions_BB4_onVal = sieve(dd_BB4habTrain_lowercased, dd_BB4habDev_lowercased, dd_localBB4HabDevRef, word_vectors)
    LocalSIEVEscore_BB4_onVal = accuracy(dd_predictions_BB4_onVal, dd_habDev)
    print("\n\nLocalSIEVEscore_BB4_onVal (only dev CUIs):", LocalSIEVEscore_BB4_onVal)


    dd_LocalSIEVEpredictions_NCBI_onTest = sieve(dd_NCBITrainFixed_lowercased, dd_NCBITestFixed_lowercased, dd_localMedicFull, word_vectors)
    LocalSIEVEscore_NCBI_onTest = accuracy(dd_LocalSIEVEpredictions_NCBI_onTest, dd_TestFixed)
    print("\nLocalSIEVEscore_NCBI_onTest (with tags):", LocalSIEVEscore_NCBI_onTest)
    dd_LocalSIEVEpredictions_NCBI_onTestWithTrainDev = sieve(dd_NCBITrainDevFixed_lowercased, dd_NCBITestFixed_lowercased, dd_localMedicFull, word_vectors)
    LocalSIEVEscore_NCBI_onTestWithTrainDev = accuracy(dd_LocalSIEVEpredictions_NCBI_onTestWithTrainDev, dd_TestFixed)
    print("\nLocalSIEVEscore_NCBI_onTestWithTrainDev (with tags):", LocalSIEVEscore_NCBI_onTestWithTrainDev)
    dd_LocalSIEVEpredictions_NCBI_onTest = sieve(dd_NCBITrainFixed_lowercased, dd_NCBITestFixed_lowercased, dd_localMedicTest, word_vectors)
    LocalSIEVEscore_NCBI_onTest = accuracy(dd_LocalSIEVEpredictions_NCBI_onTest, dd_TestFixed)
    print("\nLocalSIEVEscore_NCBI_onTest (with tags - only test CUIs):", LocalSIEVEscore_NCBI_onTest)
    dd_LocalSIEVEpredictions_NCBI_onTestWithTrainDev = sieve(dd_NCBITrainDevFixed_lowercased, dd_NCBITestFixed_lowercased, dd_localMedicTest, word_vectors)
    LocalSIEVEscore_NCBI_onTestWithTrainDev = accuracy(dd_LocalSIEVEpredictions_NCBI_onTestWithTrainDev, dd_TestFixed)
    print("\nLocalSIEVEscore_NCBI_onTestWithTrainDev (with tags - only test CUIs):", LocalSIEVEscore_NCBI_onTestWithTrainDev)
    """



    from evaluators import global_wang_score


    dd_EMpredictions_BB4_onTrain = optimized_exact_matcher(dd_BB4habTrain_lowercased, dd_habObt_lowercased)
    EMscore_BB4_onTrain = accuracy(dd_EMpredictions_BB4_onTrain, dd_habTrain)
    print("Wang Score calculation...")
    EMWangScore_BB4_onTrain = global_wang_score(dd_EMpredictions_BB4_onTrain, dd_habTrain, dd_habObt_lowercased, 0.65)
    print("\n\nEMscore_BB4_onTrain:", EMscore_BB4_onTrain, EMWangScore_BB4_onTrain)
    dd_EMpredictions_BB4_onVal = optimized_exact_matcher(dd_BB4habDev_lowercased, dd_habObt_lowercased)
    EMscore_BB4_onVal = accuracy(dd_EMpredictions_BB4_onVal, dd_habDev)
    EMWangScore_BB4_onVal = global_wang_score(dd_EMpredictions_BB4_onVal, dd_habDev, dd_habObt_lowercased, 0.65)
    print("Wang Score calculation...")
    print("\nEMscore_BB4_onVal:", EMscore_BB4_onVal, EMWangScore_BB4_onVal)