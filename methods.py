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

import sys
import numpy
from scipy.spatial.distance import cosine, euclidean




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
                    if cui in dd_count[mention].keys():
                        dd_count[mention][cui] += 1
                    else:
                        dd_count[mention][cui] = 0

        # Print progression:
        currentProgression = round(100 * (i / nbMentions))
        if currentProgression > progresssion:
            print(str(currentProgression) + "%")
            progresssion = currentProgression


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

    return dd_predictions


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
                break # First match gives the prediction


        #Print progression:
        currentProgression = round(100*(i/nbMentions))
        if currentProgression > progresssion:
            print(currentProgression, "%")
            progresssion = currentProgression

    return dd_predictions


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
def embeddings_similarity_method(dd_mentions, dd_ref, embeddings):

    nbMentions = len(dd_mentions.keys())
    progresssion = -1

    dd_predictions = dict()
    for id in dd_mentions.keys():
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []

    vocabSize = len(embeddings.wv.vocab)
    sizeVST = embeddings.wv.vector_size
    print("vocabSize:", vocabSize, "sizeVST:", sizeVST)

    i=0
    s_knownTokensInMentions = set()
    s_unknownTokensInMentions = set()
    d_mentionVectors = dict()
    dd_score = dict()
    for id in dd_mentions.keys():
        d_mentionVectors[id] = numpy.zeros(sizeVST)
        l_tokens = dd_mentions[id]["mention"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                d_mentionVectors[id] += ( embeddings[token] / numpy.linalg.norm(embeddings[token]) )
                s_knownTokensInMentions.add(token)
                dd_score[id] = dict()
            else:
                s_unknownTokensInMentions.add(token)
        d_mentionVectors[id] = d_mentionVectors[id] / len(l_tokens)

        if not numpy.any(d_mentionVectors[id]):
            i+=1
    print("Nb of null mentions:", i, "(/", len(dd_mentions.keys()), ").")
    print("Unknown tokens in Mentions:", len(s_unknownTokensInMentions), "(/", len(s_knownTokensInMentions), ").")


    d_conceptVectors = dict()
    s_unkwnownConcepts = set()
    for cui in dd_ref.keys():
        d_conceptVectors[cui] = numpy.zeros(sizeVST)
        l_tokens = dd_ref[cui]["label"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                d_conceptVectors[cui] += ( embeddings[token] / numpy.linalg.norm(embeddings[token]) )
        d_conceptVectors[cui] = d_conceptVectors[cui] / (len(l_tokens)) #+ nbTokenInAllTags)

        if not d_conceptVectors[cui].any():
            s_unkwnownConcepts.add(cui)
    print("Nb concepts nuls:", len(s_unkwnownConcepts), "\n\n\n")

    del embeddings


    for i, id in enumerate(dd_score.keys()):
        for cui in d_conceptVectors.keys():
            if d_mentionVectors[id].any() and d_conceptVectors[cui].any():
                dd_score[id][cui] = 1 - cosine(d_mentionVectors[id], d_conceptVectors[cui])

        #Print progression:
        currentProgression = round(100*(i/len(dd_score.keys()) ))
        if currentProgression > progresssion:
            print(currentProgression, "%")
            progresssion = currentProgression


    del d_mentionVectors
    del d_conceptVectors

    dl_maxScore = dict()
    for id in dd_score.keys():
        dl_maxScore[id] = list()
        for cui in dd_score[id].keys():
            dl_maxScore[id].append(dd_score[id][cui])
    for id in dd_score.keys():
        maximumScore = max(dl_maxScore[id])
        for cui in dd_score[id].keys():
            if dd_score[id][cui] == maximumScore:
                dd_predictions[id]["pred_cui"] = [cui]
                break

    return dd_predictions




# A static method with embeddings
def embeddings_similarity_method_with_tags(dd_mentions, dd_ref, embeddings):

    progresssion = -1
    dd_predictions = dict()
    for id in dd_mentions.keys():
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []

    vocabSize = len(embeddings.wv.vocab)
    sizeVST = embeddings.wv.vector_size
    print("vocabSize:", vocabSize, "sizeVST:", sizeVST)

    i=0
    s_knownTokensInMentions = set()
    s_unknownTokensInMentions = set()
    d_mentionVectors = dict()
    dd_score = dict()
    for id in dd_mentions.keys():
        d_mentionVectors[id] = numpy.zeros(sizeVST)
        l_tokens = dd_mentions[id]["mention"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                d_mentionVectors[id] += ( embeddings[token] / numpy.linalg.norm(embeddings[token]) )
                s_knownTokensInMentions.add(token)
                dd_score[id] = dict()
            else:
                s_unknownTokensInMentions.add(token)
        d_mentionVectors[id] = d_mentionVectors[id] / len(l_tokens)

        if not numpy.any(d_mentionVectors[id]):
            i+=1
    print("Nb of null mentions:", i, "(/", len(dd_mentions.keys()), ").")
    print("Unknown tokens in Mentions:", len(s_unknownTokensInMentions), "(/", len(s_knownTokensInMentions), ").")


    dd_conceptVectors = dict()
    s_unkwnownConcepts = set()
    for cui in dd_ref.keys():
        dd_conceptVectors[cui] = dict()
        dd_conceptVectors[cui][dd_ref[cui]["label"]] = numpy.zeros(sizeVST)
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
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


    for i, id in enumerate(dd_score.keys()):
        for cui in dd_conceptVectors.keys():
            l_scoreForCui = list()
            for labtag in dd_conceptVectors[cui].keys():
                if d_mentionVectors[id].any() and dd_conceptVectors[cui][labtag].any():
                    score = 1 - cosine(d_mentionVectors[id], dd_conceptVectors[cui][labtag])
                    l_scoreForCui.append(score)
            if len(l_scoreForCui) > 0:
                dd_score[id][cui] = max(l_scoreForCui)

        #Print progression:
        currentProgression = round(100*(i/len(dd_score.keys()) ))
        if currentProgression > progresssion:
            print(currentProgression, "%")
            progresssion = currentProgression


    del d_mentionVectors
    del dd_conceptVectors

    dl_maxScore = dict()
    for id in dd_score.keys():
        dl_maxScore[id] = list()
        for cui in dd_score[id].keys():
            dl_maxScore[id].append(dd_score[id][cui])
    for id in dd_score.keys():
        maximumScore = max(dl_maxScore[id])
        for cui in dd_score[id].keys():
            if dd_score[id][cui] == maximumScore:
                dd_predictions[id]["pred_cui"] = [cui]
                break

    return dd_predictions


##################################################
# Model too big for big reference...
def wordCNN(dd_train, dd_mentions, dd_ref, embeddings, phraseMaxSize=15):

    progresssion = -1
    dd_predictions = dict()
    for id in dd_mentions.keys():
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []


    # Prepare matrix for training data:
    nbMentions = len(dd_train.keys())
    vocabSize = len(embeddings.wv.vocab)
    sizeVST = embeddings.wv.vector_size
    sizeVSO = len(dd_ref.keys())
    print("vocabSize:", vocabSize, "sizeVST:", sizeVST, "sizeVSO:", sizeVSO)
    X_train = numpy.zeros((nbMentions, phraseMaxSize, sizeVST))
    Y_train = numpy.zeros((nbMentions, 1, sizeVSO))


    # Build training data:
    d_dimToCui = dict()
    for k, cui in enumerate(dd_ref.keys()):
        d_dimToCui[k] = cui
    for i, id in enumerate(dd_train.keys()):
        toPredCui = dd_train[id]["cui"][0]
        for dim in d_dimToCui.keys():
            if d_dimToCui[dim] == toPredCui:
                Y_train[i][0] = numpy.zeros((1, sizeVSO))
                Y_train[i][0][dim] = 1.0 # one-hot encoding
                for j, token in enumerate(dd_train[id]["mention"].split()):
                    if j < phraseMaxSize:
                        if token in embeddings.wv.vocab:
                            X_train[i][j] = embeddings[token] / numpy.linalg.norm(embeddings[token])
                break  # Because it' easier to keep only one concept per mention (mainly to calculate size of matrix).


    # Neural architecture:
    inputLayer = Input(shape=(phraseMaxSize, sizeVST))
    convLayer = (layers.Conv1D( sizeVSO, 1, strides=1, activation='relu', kernel_initializer='glorot_uniform'))(inputLayer)
    outputSize = phraseMaxSize - 1 + 1
    pool = (layers.MaxPool1D(pool_size=outputSize))(convLayer)
    # denseLayer = (layers.Dense(sizeVSO, activation='softmax', use_bias=True))(pool)
    denseLayer = (layers.Activation('softmax'))(pool)

    CNNmodel = Model(inputs=inputLayer, outputs=denseLayer)

    CNNmodel.summary()
    CNNmodel.compile(optimizer=optimizers.Nadam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Training
    callback = callbacks.EarlyStopping(monitor='categorical_accuracy', patience=10, min_delta=0.01)
    history = CNNmodel.fit(X_train, Y_train, epochs=100, batch_size=64, callbacks=[callback])
    print("\nTraining done.")


    # Mentions embeddings:
    X_test = numpy.zeros( (len(dd_mentions.keys()), phraseMaxSize, sizeVST) )
    for i, id in enumerate(dd_mentions.keys()):
        x_test = numpy.zeros((1, phraseMaxSize, sizeVST))
        for j, token in enumerate(dd_mentions[id]["mention"].split()):
            if j < phraseMaxSize:
                if token in embeddings.wv.vocab:
                    x_test[0][j] = embeddings[token]

        indice = numpy.argmax(CNNmodel.predict(x_test)[0][0])
        dd_predictions[id]["pred_cui"] = [d_dimToCui[indice]]

        #Print progression:
        currentProgression = round(100*(i/len(dd_mentions.keys()) ))
        if currentProgression > progresssion:
            print(currentProgression, "%")
            progresssion = currentProgression


    # Supprimer les mentions avec vecteur nul pour l'entrainement...?
    # ToDo: Rendre la methode capable de ne faire que le train, que la pred, ou les 2.

    return dd_predictions






# Embeddings+ML but tak less RAM with big ref:
def dense_layer_method(dd_train, dd_mentions, dd_ref, embeddings):

    progresssion = -1
    dd_predictions = dict()
    for id in dd_mentions.keys():
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []

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
    dd_conceptVectors = dict()
    for cui in dd_ref.keys():
        dd_conceptVectors[cui] = dict()
        dd_conceptVectors[cui][dd_ref[cui]["label"]] = numpy.zeros(sizeVST)
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
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
        for j, tag in enumerate(dd_conceptVectors[toPredCui].keys()):
            if dd_conceptVectors[toPredCui][tag].any() and j<(len(dd_conceptVectors[toPredCui].keys())-1):
                Y_train[i] = dd_conceptVectors[toPredCui][tag]
                break  # Just taking the first label
            elif j==(len(dd_conceptVectors[toPredCui].keys())-1):
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
    history = CNNmodel.fit(X_train, Y_train, epochs=200, batch_size=64, callbacks=[callback])
    # plt.plot(history.history['logcosh'], label='logcosh')
    # plt.show()
    print("\nTraining done.")


    # Mentions embeddings:
    d_mentionVectors = dict()
    for id in dd_mentions.keys():
        d_mentionVectors[id] = numpy.zeros(sizeVST)
        l_tokens = dd_mentions[id]["mention"].split()
        for token in l_tokens:
            if token in embeddings.wv.vocab:
                d_mentionVectors[id] += (embeddings[token] / numpy.linalg.norm(embeddings[token]))
        d_mentionVectors[id] = d_mentionVectors[id] / len(l_tokens)



    # Prediction:
    dd_score = dict()
    for i, id in enumerate(dd_mentions.keys()): #ValueError: Error when checking input: expected input_1 to have shape (100,) but got array with shape (1,)
        x_test = numpy.zeros((1, sizeVST))
        x_test[0] = d_mentionVectors[id]
        y_pred = CNNmodel.predict(x_test)[0]

        dd_score[id] = dict()
        for cui in dd_conceptVectors.keys():

            l_scoreForCui = list()
            for labtag in dd_conceptVectors[cui].keys():
                if y_pred.any() and dd_conceptVectors[cui][labtag].any():
                    score = 1 - cosine(y_pred, dd_conceptVectors[cui][labtag])
                    l_scoreForCui.append(score)

            if len(l_scoreForCui) > 0:
                dd_score[id][cui] = max(l_scoreForCui)

        #Print progression:
        currentProgression = round(100*(i/len(dd_mentions.keys()) ))
        if currentProgression > progresssion:
            print(currentProgression, "%")
            progresssion = currentProgression


    del dd_conceptVectors
    del d_mentionVectors


    dl_maxScore = dict()
    for id in dd_score.keys():
        dl_maxScore[id] = list()
        for cui in dd_score[id].keys():
            dl_maxScore[id].append(dd_score[id][cui])
    for id in dd_score.keys():
        maximumScore = max(dl_maxScore[id])
        for cui in dd_score[id].keys():
            if dd_score[id][cui] == maximumScore:
                dd_predictions[id]["pred_cui"] = [cui]
                break

    # Supprimer les mentions avec vecteur nul pour l'entrainement...?
    # ToDo: Rendre la methode capable de ne faire que le train, que la pred, ou les 2.

    return dd_predictions



##################################################




def sieve():
    """
    Description: Begin by by_heart_and_exact_matching(), then () on mentions without predictions.
    :return:
    """
    return None







###################################################
# Preprocessing tools:
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
    ps = PorterStemmer()
    for id in dd_mentions.keys():
        dd_mentions[id]["mention"] = ps.stem(dd_mentions[id]["mention"].lower())
    return dd_mentions

def stem_lowercase_ref(dd_ref):
    ps = PorterStemmer()
    for cui in dd_ref.keys():
        dd_ref[cui]["label"] = ps.stem(dd_ref[cui]["label"].lower())
        if "tags" in dd_ref[cui].keys():
            l_lowercasedTags = list()
            for tag in dd_ref[cui]["tags"]:
                l_lowercasedTags.append(ps.stem(tag.lower()))
            dd_ref[cui]["tags"] = l_lowercasedTags
    return dd_ref


def stopword_filtering_mentions(dd_mentions):
    ps = PorterStemmer()
    for id in dd_mentions.keys():
        dd_mentions[id]["mention"] = ps.stem(dd_mentions[id]["mention"].lower())


     #ToDo

    return dd_mentions


###################################################
# Preprocessing tools:
###################################################

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

    from loaders import loader_clinical_finding_file, loader_amt, select_subpart_with_patterns_in_label, get_tags_in_ref, fusion_ref, get_cui_list
    from loaders import extract_data_without_file, loader_all_initial_cadec_folds, loader_all_random_cadec_folds, loader_all_custom_cadec_folds
    from loaders import loader_ontobiotope, select_subpart_hierarchy, loader_one_bb4_fold
    from loaders import loader_medic, loader_one_ncbi_fold, extract_data
    from evaluators import accuracy



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
    dd_randCADEC_train0_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-0.train"])
    dd_randCADEC_train1_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-1.train"])
    dd_randCADEC_train2_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-2.train"])
    dd_randCADEC_train3_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-3.train"])
    dd_randCADEC_train4_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-4.train"])
    dd_randCADEC_train5_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-5.train"])
    dd_randCADEC_train6_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-6.train"])
    dd_randCADEC_train7_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-7.train"])
    dd_randCADEC_train8_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-8.train"])
    dd_randCADEC_train9_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-9.train"])

    dd_randCADEC_validation0_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-0.validation"])
    dd_randCADEC_validation1_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-1.validation"])
    dd_randCADEC_validation2_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-2.validation"])
    dd_randCADEC_validation3_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-3.validation"])
    dd_randCADEC_validation4_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-4.validation"])
    dd_randCADEC_validation5_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-5.validation"])
    dd_randCADEC_validation6_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-6.validation"])
    dd_randCADEC_validation7_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-7.validation"])
    dd_randCADEC_validation8_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-8.validation"])
    dd_randCADEC_validation9_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-9.validation"])

    dd_randCADEC_test0_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-0.test"])
    dd_randCADEC_test1_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-1.test"])
    dd_randCADEC_test2_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-2.test"])
    dd_randCADEC_test3_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-3.test"])
    dd_randCADEC_test4_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-4.test"])
    dd_randCADEC_test5_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-5.test"])
    dd_randCADEC_test6_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-6.test"])
    dd_randCADEC_test7_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-7.test"])
    dd_randCADEC_test8_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-8.test"])
    dd_randCADEC_test9_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-9.test"])

    dd_customCADEC_train0_lowercased = stem_lowercase_mentions(ddd_customData["train_0"])
    dd_customCADEC_train1_lowercased = stem_lowercase_mentions(ddd_customData["train_1"])
    dd_customCADEC_train2_lowercased = stem_lowercase_mentions(ddd_customData["train_2"])
    dd_customCADEC_train3_lowercased = stem_lowercase_mentions(ddd_customData["train_3"])
    dd_customCADEC_train4_lowercased = stem_lowercase_mentions(ddd_customData["train_4"])

    dd_customCADEC_validation0_lowercased = stem_lowercase_mentions(ddd_customData["test_0"])
    dd_customCADEC_validation1_lowercased = stem_lowercase_mentions(ddd_customData["test_1"])
    dd_customCADEC_validation2_lowercased = stem_lowercase_mentions(ddd_customData["test_2"])
    dd_customCADEC_validation3_lowercased = stem_lowercase_mentions(ddd_customData["test_3"])
    dd_customCADEC_validation4_lowercased = stem_lowercase_mentions(ddd_customData["test_4"])

    dd_BB4habTrain_lowercased = stem_lowercase_mentions(dd_habTrain)
    dd_BB4habDev_lowercased = stem_lowercase_mentions(dd_habDev)

    dd_NCBITrainFixed_lowercased = stem_lowercase_mentions(dd_TrainFixed)
    dd_NCBIDevFixed_lowercased = stem_lowercase_mentions(dd_DevFixed)
    dd_NCBITrainDevFixed_lowercased = stem_lowercase_mentions(dd_TrainDevFixed)
    dd_NCBITestFixed_lowercased = stem_lowercase_mentions(dd_TestFixed)


    print("Mentions lowercasing done.\n")


    print("Lowercase references...")
    dd_subsubRef_lowercased = stem_lowercase_ref(dd_subsubRef)
    dd_habObt_lowercased = stem_lowercase_ref(dd_habObt)
    dd_medic_lowercased = stem_lowercase_ref(dd_medic)
    print("Done.")



    ################################################
    print("\n\n\n\nPREDICTING:\n")
    ################################################

    #######################
    print("By heart learning method:")
    #######################

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



    #######################
    print("\n\n\nExact Matching method:\n")
    #######################

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



    #######################
    print("\n\n\nExact Matching + By Heart method:\n")
    #######################

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



    #######################
    print("\n\n\nStatic distance between label/mention embeddings:\n")
    #######################
    """
    word_vectors = KeyedVectors.load_word2vec_format('../PubMed-w2v.bin', binary=True)
    #word_vectors = Word2Vec.load('../VST_count0_size100_iter50.model')


    dd_predictions_customCADEC0_onVal = embeddings_similarity_method(dd_customCADEC_validation0_lowercased, dd_subsubRef, word_vectors)
    print(dd_predictions_customCADEC0_onVal)
    SEscorecustomCADEC0_onVal = accuracy(dd_predictions_customCADEC0_onVal, ddd_customData["test_0"])
    print("\n\nSEscorecustomCADEC0_onVal:", SEscorecustomCADEC0_onVal)
    dd_predictions_customCADEC1_onVal = embeddings_similarity_method(dd_customCADEC_validation1_lowercased, dd_subsubRef, word_vectors)
    SEscorecustomCADEC1_onVal = accuracy(dd_predictions_customCADEC1_onVal, ddd_customData["test_1"])
    print("\nSEscorecustomCADEC1_onVal:", SEscorecustomCADEC1_onVal)
    dd_predictions_customCADEC2_onVal = embeddings_similarity_method(dd_customCADEC_validation2_lowercased, dd_subsubRef, word_vectors)
    SEscorecustomCADEC2_onVal = accuracy(dd_predictions_customCADEC2_onVal, ddd_customData["test_2"])
    print("\nSEscorecustomCADEC2_onVal:", SEscorecustomCADEC2_onVal)
    dd_predictions_customCADEC3_onVal = embeddings_similarity_method(dd_customCADEC_validation3_lowercased, dd_subsubRef, word_vectors)
    SEscorecustomCADEC3_onVal = accuracy(dd_predictions_customCADEC3_onVal, ddd_customData["test_3"])
    print("\nSEscorecustomCADEC3_onVal:", SEscorecustomCADEC3_onVal)
    dd_predictions_customCADEC4_onVal = embeddings_similarity_method(dd_customCADEC_validation4_lowercased, dd_subsubRef, word_vectors)
    SEscorecustomCADEC4_onVal = accuracy(dd_predictions_customCADEC4_onVal, ddd_customData["test_4"])
    print("\nSEscorecustomCADEC4_onVal:", SEscorecustomCADEC4_onVal)


    dd_predictions_customCADEC0_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation0_lowercased, dd_subsubRef, word_vectors)
    SEscorecustomCADEC0_onVal = accuracy(dd_predictions_customCADEC0_onVal, ddd_customData["test_0"])
    print("\n\nSEscorecustomCADEC0_onVal:", SEscorecustomCADEC0_onVal)
    dd_predictions_customCADEC1_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation1_lowercased, dd_subsubRef, word_vectors)
    SEscorecustomCADEC1_onVal = accuracy(dd_predictions_customCADEC1_onVal, ddd_customData["test_1"])
    print("\nSEscorecustomCADEC1_onVal:", SEscorecustomCADEC1_onVal)
    dd_predictions_customCADEC2_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation2_lowercased, dd_subsubRef, word_vectors)
    SEscorecustomCADEC2_onVal = accuracy(dd_predictions_customCADEC2_onVal, ddd_customData["test_2"])
    print("\nSEscorecustomCADEC2_onVal:", SEscorecustomCADEC2_onVal)
    dd_predictions_customCADEC3_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation3_lowercased, dd_subsubRef, word_vectors)
    SEscorecustomCADEC3_onVal = accuracy(dd_predictions_customCADEC3_onVal, ddd_customData["test_3"])
    print("\nSEscorecustomCADEC3_onVal:", SEscorecustomCADEC3_onVal)
    dd_predictions_customCADEC4_onVal = embeddings_similarity_method_with_tags(dd_customCADEC_validation4_lowercased, dd_subsubRef, word_vectors)
    SEscorecustomCADEC4_onVal = accuracy(dd_predictions_customCADEC4_onVal, ddd_customData["test_4"])
    print("\nSEscorecustomCADEC4_onVal:", SEscorecustomCADEC4_onVal)


    dd_EMpredictions_NCBI_onVal = embeddings_similarity_method(dd_NCBIDevFixed_lowercased, dd_medic_lowercased, word_vectors)
    SEscore_NCBI_onVal = accuracy(dd_EMpredictions_NCBI_onVal, dd_NCBIDevFixed_lowercased)
    print("\nSEscore_NCBI_onVal:", SEscore_NCBI_onVal)
    dd_EMpredictions_NCBI_onTest = embeddings_similarity_method(dd_NCBITestFixed_lowercased, dd_medic_lowercased, word_vectors)
    SEscore_NCBI_onTest = accuracy(dd_EMpredictions_NCBI_onTest, dd_TestFixed)
    print("\nSEscore_NCBI_onTest:", SEscore_NCBI_onTest)

    dd_EMpredictions_NCBI_onVal = embeddings_similarity_method_with_tags(dd_NCBIDevFixed_lowercased, dd_medic_lowercased, word_vectors)
    SEscore_NCBI_onVal = accuracy(dd_EMpredictions_NCBI_onVal, dd_NCBIDevFixed_lowercased)
    print("\nSEscore_NCBI_onVal (with tags):", SEscore_NCBI_onVal)
    dd_EMpredictions_NCBI_onTest = embeddings_similarity_method_with_tags(dd_NCBITestFixed_lowercased, dd_medic_lowercased, word_vectors)
    SEscore_NCBI_onTest = accuracy(dd_EMpredictions_NCBI_onTest, dd_TestFixed)
    print("\nSEscore_NCBI_onTest (with tags):", SEscore_NCBI_onTest)


    dd_SEpredictions_randCADEC0_onTest = embeddings_similarity_method(dd_randCADEC_test0_lowercased, dd_subsubRef_lowercased, word_vectors)
    SEscoreRandCADEC0_onTest = accuracy(dd_SEpredictions_randCADEC0_onTest, ddd_randData["AskAPatient.fold-0.test"])
    print("\n\nSEscoreRandCADEC0_onTest (without tags):", SEscoreRandCADEC0_onTest)


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
    """


    #######################
    print("\n\n\nML distance between label/mention embeddings:\n")
    #######################

    word_vectors = KeyedVectors.load_word2vec_format('../PubMed-w2v.bin', binary=True)
    # word_vectors = Word2Vec.load('../VST_count0_size100_iter50.model')


    dd_predictions_customCADEC0_onVal = dense_layer_method(dd_customCADEC_validation0_lowercased, dd_customCADEC_train0_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscorecustomCADEC0_onVal = accuracy(dd_predictions_customCADEC0_onVal, ddd_customData["test_0"])
    print("\n\nMLEscorecustomCADEC0_onVal:", MLEscorecustomCADEC0_onVal)
    dd_predictions_customCADEC1_onVal = dense_layer_method(dd_customCADEC_validation1_lowercased, dd_customCADEC_train1_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscorecustomCADEC1_onVal = accuracy(dd_predictions_customCADEC1_onVal, ddd_customData["test_1"])
    print("\nMLEscorecustomCADEC1_onVal:", MLEscorecustomCADEC1_onVal)
    dd_predictions_customCADEC2_onVal = dense_layer_method(dd_customCADEC_validation2_lowercased, dd_customCADEC_train2_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscorecustomCADEC2_onVal = accuracy(dd_predictions_customCADEC2_onVal, ddd_customData["test_2"])
    print("\nMLEscorecustomCADEC2_onVal:", MLEscorecustomCADEC2_onVal)
    dd_predictions_customCADEC3_onVal = dense_layer_method(dd_customCADEC_validation3_lowercased, dd_customCADEC_train3_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscorecustomCADEC3_onVal = accuracy(dd_predictions_customCADEC3_onVal, ddd_customData["test_3"])
    print("\nMLEscorecustomCADEC3_onVal:", MLEscorecustomCADEC3_onVal)
    dd_predictions_customCADEC4_onVal = dense_layer_method(dd_customCADEC_validation4_lowercased, dd_customCADEC_train4_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscorecustomCADEC4_onVal = accuracy(dd_predictions_customCADEC4_onVal, ddd_customData["test_4"])
    print("\nMLEscorecustomCADEC4_onVal:", MLEscorecustomCADEC4_onVal)


    dd_MLEpredictions_randCADEC0_onTest = dense_layer_method(dd_randCADEC_test0_lowercased, dd_randCADEC_train0_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC0_onTest = accuracy(dd_MLEpredictions_randCADEC0_onTest, ddd_randData["AskAPatient.fold-0.test"])
    print("\n\nMLEscoreRandCADEC0_onTest:", MLEscoreRandCADEC0_onTest)
    dd_MLEpredictions_randCADEC1_onTest = dense_layer_method(dd_randCADEC_test1_lowercased, dd_randCADEC_train1_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC1_onTest = accuracy(dd_MLEpredictions_randCADEC1_onTest, ddd_randData["AskAPatient.fold-1.test"])
    print("\nMLEscoreRandCADEC1_onTest:", MLEscoreRandCADEC1_onTest)
    dd_MLEpredictions_randCADEC2_onTest = dense_layer_method(dd_randCADEC_test2_lowercased, dd_randCADEC_train2_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC2_onTest = accuracy(dd_MLEpredictions_randCADEC2_onTest, ddd_randData["AskAPatient.fold-2.test"])
    print("\nMLEscoreRandCADEC2_onTest:", MLEscoreRandCADEC2_onTest)
    dd_MLEpredictions_randCADEC3_onTest = dense_layer_method(dd_randCADEC_test3_lowercased, dd_randCADEC_train3_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC3_onTest = accuracy(dd_MLEpredictions_randCADEC3_onTest, ddd_randData["AskAPatient.fold-3.test"])
    print("\nMLEscoreRandCADEC3_onTest:", MLEscoreRandCADEC3_onTest)
    dd_MLEpredictions_randCADEC4_onTest = dense_layer_method(dd_randCADEC_test4_lowercased, dd_randCADEC_train4_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC4_onTest = accuracy(dd_MLEpredictions_randCADEC4_onTest, ddd_randData["AskAPatient.fold-4.test"])
    print("\nMLEscoreRandCADEC4_onTest:", MLEscoreRandCADEC4_onTest)
    dd_MLEpredictions_randCADEC5_onTest = dense_layer_method(dd_randCADEC_test5_lowercased, dd_randCADEC_train5_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC5_onTest = accuracy(dd_MLEpredictions_randCADEC5_onTest, ddd_randData["AskAPatient.fold-5.test"])
    print("\nMLEscoreRandCADEC5_onTest:", MLEscoreRandCADEC5_onTest)
    dd_MLEpredictions_randCADEC6_onTest = dense_layer_method(dd_randCADEC_test6_lowercased, dd_randCADEC_train6_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC6_onTest = accuracy(dd_MLEpredictions_randCADEC6_onTest, ddd_randData["AskAPatient.fold-6.test"])
    print("\nMLEscoreRandCADEC6_onTest:", MLEscoreRandCADEC6_onTest)
    dd_MLEpredictions_randCADEC7_onTest = dense_layer_method(dd_randCADEC_test7_lowercased, dd_randCADEC_train7_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC7_onTest = accuracy(dd_MLEpredictions_randCADEC7_onTest, ddd_randData["AskAPatient.fold-7.test"])
    print("\nMLEscoreRandCADEC7_onTest:", MLEscoreRandCADEC7_onTest)
    dd_MLEpredictions_randCADEC8_onTest = dense_layer_method(dd_randCADEC_test8_lowercased, dd_randCADEC_train8_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC8_onTest = accuracy(dd_MLEpredictions_randCADEC8_onTest, ddd_randData["AskAPatient.fold-8.test"])
    print("\nMLEscoreRandCADEC8_onTest:", MLEscoreRandCADEC8_onTest)
    dd_MLEpredictions_randCADEC9_onTest = dense_layer_method(dd_randCADEC_test9_lowercased, dd_randCADEC_train9_lowercased, dd_subsubRef_lowercased, word_vectors)
    MLEscoreRandCADEC9_onTest = accuracy(dd_MLEpredictions_randCADEC9_onTest, ddd_randData["AskAPatient.fold-9.test"])
    print("\nMLEscoreRandCADEC9_onTest:", MLEscoreRandCADEC9_onTest)


    dd_predictions_BB4_onVal = dense_layer_method(dd_BB4habDev_lowercased, dd_BB4habTrain_lowercased, dd_habObt_lowercased, word_vectors)
    MLEscore_BB4_onVal = accuracy(dd_predictions_BB4_onVal, dd_habDev)
    print("\n\nBHEMscore_BB4_onVal:", MLEscore_BB4_onVal)


    dd_MLEpredictions_NCBI_onTest = dense_layer_method(dd_NCBITestFixed_lowercased, dd_NCBITrainDevFixed_lowercased, dd_medic_lowercased, word_vectors)
    MLEscore_NCBI_onTest = accuracy(dd_MLEpredictions_NCBI_onTest, dd_TestFixed)
    print("\nMLEscore_NCBI_onTest (with tags):", MLEscore_NCBI_onTest)




    """
    # word_vectors = Word2Vec.load('../VST_count0_size100_iter50.model')
    word_vectors = KeyedVectors.load_word2vec_format('../PubMed-w2v.bin', binary=True)

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


    print("\nLoading random CADEC corpus...")
    ddd_randData = loader_all_random_cadec_folds("../CADEC/1_Random_folds_AskAPatient/")
    dd_randCADEC_test0_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-0.test"])
    dd_randCADEC_train0_lowercased = stem_lowercase_mentions(ddd_randData["AskAPatient.fold-0.train"])


    dd_EMpredictions_randCADEC0_onTest = wordCNN(dd_randCADEC_train0_lowercased, dd_randCADEC_test0_lowercased, dd_subsubRef, word_vectors, phraseMaxSize=15)
    #optimized_exact_matcher(dd_randCADEC_test0_lowercased, dd_subsubRef_lowercased)
    EMscoreRandCADEC0_onTest = accuracy(dd_EMpredictions_randCADEC0_onTest, ddd_randData["AskAPatient.fold-0.test"])
    print("\n\nEMscoreRandCADEC0_onTest:", EMscoreRandCADEC0_onTest)

    sys.exit(0)


    print("loading OntoBiotope...")
    dd_obt = loader_ontobiotope("../BB4/OntoBiotope_BioNLP-OST-2019.obo")
    print("loaded. (Nb of concepts in SCT =", len(dd_obt.keys()), ", Nb of tags =", len(get_tags_in_ref(dd_obt)), ")")
    print("\nExtracting Bacterial Habitat hierarchy:")
    dd_habObt = select_subpart_hierarchy(dd_obt, 'OBT:000001')
    print("Done. (Nb of concepts in this subpart of OBT =", len(dd_habObt.keys()), ", Nb of tags =",
          len(get_tags_in_ref(dd_habObt)), ")")

    ddd_dataTrain = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_train"])
    dd_habTrain = extract_data(ddd_dataTrain, l_type=["Habitat"])  # ["Habitat", "Phenotype", "Microorganism"]
    print("loaded.(Nb of mentions in train =", len(dd_habTrain.keys()), ")")
    dd_BB4habTrain_lowercased = lowercaser_mentions(dd_habTrain)

    ddd_dataDev = loader_one_bb4_fold(["../BB4/BioNLP-OST-2019_BB-norm_dev"])
    dd_habDev = extract_data(ddd_dataDev, l_type=["Habitat"])
    print("loaded.(Nb of mentions in dev =", len(dd_habDev.keys()), ")")
    dd_BB4habDev_lowercased = lowercaser_mentions(dd_habDev)

    dd_pred = dense_layer_method(dd_BB4habTrain_lowercased, dd_BB4habDev_lowercased, dd_habObt, word_vectors)

    BHscore_BB4_onVal = accuracy(dd_pred, dd_habDev)
    print("\nBHscore_BB4_onVal:", BHscore_BB4_onVal)

    sys.exit(0)
    """









