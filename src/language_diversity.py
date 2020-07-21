import spacy
from spacy.lang.xx import MultiLanguage
import numpy as np
import pandas as pd
import math

from .parameters import getParam

LANGUAGE = getParam('language')
LANGUAGE_MODELS = getParam('languageModels')


def getModel(language: str, language_models: dict = LANGUAGE_MODELS) -> spacy.lang:

    if language not in language_models.keys():
        print("language not supported. Running on MultiLanguage.")
        return MultiLanguage()

    return spacy.load(language_models[language], disable=["parser"])


def removeEntities(doc: spacy.tokens.Doc) -> list:
    entityRanges = np.concatenate(
        [np.arange(e.start, e.end) for e in doc.ents])

    return [token for token in doc if token.i not in entityRanges]


def removePunctuation(doc: list) -> list:
    return [token for token in doc if token.pos_ != "PUNCT"]


def removeNumbers(doc: list) -> list:
    return [token for token in doc if token.pos_ != "NUM"]


def calculateDiversity(doc: spacy.tokens.Doc, batch_size: int = 2000) -> float:
    noEnts = removeEntities(doc)
    noPunct = removePunctuation(noEnts)
    noNum = removeNumbers(noPunct)
    prep = np.array(noNum)

    if batch_size > len(prep):
        return None

    excess = len(prep) % batch_size

    if(excess == 0):
        batches = np.reshape(prep, (-1, batch_size))
    else:
        batches = np.reshape(prep[:-excess], (-1, batch_size))

    lemmaNumbers = list()
    for batch in batches:
        lemmas = set()
        for token in batch:
            lemmas.add(token.lemma_)
        lemmaNumbers.append(len(lemmas))

    return sum(lemmaNumbers)/len(lemmaNumbers)/batch_size


def getChatStrings(df: pd.DataFrame, chat: str, avg_batch_chars: int = 70000) -> dict:
    prep = df.dropna(subset=["content"])
    oneChat = prep[prep["chat_with"] == chat]

    chatStrings = dict()
    for direction in ["Sent", "Received"]:
        directedChat = oneChat[oneChat["message_direction"] == direction]
        messages = np.array(directedChat["content"].astype(str).values)

        totalLen = sum([len(message) for message in messages])
        batchNum = math.ceil(totalLen/avg_batch_chars)

        batches = np.array_split(messages, batchNum)

        batchStrings = [". ".join(batchMessages) for batchMessages in batches]

        chatStrings[direction] = batchStrings

    return chatStrings
