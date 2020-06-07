from collections import OrderedDict
import numpy as np
import os
import re
import pandas as pd
from corpy.udpipe import Model

DEFAULT_MODELS_FOLDER = "language_models"
DEFAULT_STOPWORDS_FOLDER = "stopwords"

POSSIBLE_KEYWORD_POS = ["NOUN", "PROPN"]


def getStopwords(language: str, stopwords_dir: str = DEFAULT_STOPWORDS_FOLDER) -> list:
    for root, _, files in os.walk(stopwords_dir):
        for file in files:
            if (file.endswith("stopwords.txt") and file.startswith(language)):
                with open(os.path.join(root, file), "r") as f:
                    return [x.strip() for x in f.readlines()]

    return []


def getUdpipeModelNames(dirname: str = DEFAULT_MODELS_FOLDER) -> list:
    modelNames = []
    pattern = r"[a-zA-Z]+"

    for root, _, files in os.walk(dirname):
        for file in files:
            if (file.endswith(".udpipe")):
                filePath = os.path.join(root, file)
                languageName = re.findall(pattern, file)[0]
                modelNames.append((languageName, filePath))

    return modelNames


def getModel(language: str = "english", models_dir: str = DEFAULT_MODELS_FOLDER):
    models = dict(getUdpipeModelNames(models_dir))

    if language not in models.keys():
        print("ERR: Language '{}' is not available. \n\nCurrently language packs installed locally are: ".format(language))
        for lang in models.keys():
            print(" - ", lang)

        print("\nIf your language is not listed here consider downloading language package from official UDpipe repository. Then just put the file in '{}' folder.".format(DEFAULT_MODELS_FOLDER))
        return None

    return Model(models[language])


class keywordExtractor():
    def __init__(self, language: str, stopwords_dir: str = DEFAULT_STOPWORDS_FOLDER, models_dir: str = DEFAULT_MODELS_FOLDER, pos: list = POSSIBLE_KEYWORD_POS):
        self.d = 0.85
        self.min_diff = 1e-5
        self.steps = 10
        self.node_weight = None
        self.language = language
        self.stopwords = getStopwords(
            language=language, stopwords_dir=stopwords_dir)
        self.lemmaModel = getModel(
            language=language, models_dir=models_dir)
        self.partsOfSpeech = pos

    def cleanText(self, text: str):
        return " ".join([word for word in text.split() if not word.startswith("http")])


def createCorpus(data: pd.DataFrame, names: str) -> list:
    corpus = []
    for name in names:
        oneChat = data[data["chat_with"] == name]
        chatString = ". ".join(oneChat["content"].astype(str).values)
        corpus.append(chatString)

    return corpus


data = pd.read_csv("all_messages.csv")
chats = 5
keywordNum = 15

noGroup = data[data["chat_with"] != "GROUP"]
plotDataSeries = noGroup["chat_with"].value_counts()[:chats]
names = [name[0] for name in plotDataSeries.items()]

corpus = createCorpus(data, names)

extractor = keywordExtractor(language="polish")

extractor.cleanText(corpus[0])
