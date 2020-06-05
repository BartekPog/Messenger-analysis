import os
import re
from corpy.udpipe import Model

DEFAULT_MODELS_FOLDER = "language_models"
DEFAULT_STOPWORDS_FOLDER = "stopwords"

POSSIBLE_KEYWORD_POS = ["NOUN", "ADJ"]


def getStopwords(language: str, stopwords_dir: str = DEFAULT_STOPWORDS_FOLDER) -> list:
    stopwordsFiles = []

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


def getModel(language: str = "english"):
    models = dict(getUdpipeModelNames(DEFAULT_MODELS_FOLDER))

    if language not in models.keys():
        print("ERR: Language '{}' is not available. \n\nCurrently language packs installed locally are: ".format(language))
        for lang in models.keys():
            print(" - ", lang)

        print("\nIf your language is not listed here consider downloading language package from official UDpipe repository. Then just put the file in '{}' folder.".format(DEFAULT_MODELS_FOLDER))

    return Model(models[language])


sampleText = '''Wzięła baba maku wór
Bo wyniknął z dziadem spór
Baba twierdzi, że bez dziada
Z siewem też da sobie radę
Bierze worek, jak to baba
I zaczyna siać

Czeka baba tydzień, dwa
Skoro świt na pole gna
Nic nie wzeszło, trudna rada
A to była wina dziada
Nic innego nie wypada
Jak ponowić siew

Widzisz, babo, to jest znak
Że tu dziada chyba brak
Cały rok byś siała tak
Dziad to ważna rzecz

Idzie baba na ten znak
Prosi dziada, choć nie w smak
To był sprytny fortel dziada
Sekret zdradzić tu wypada
Aby utrzeć nosa babie
Wsypał w worek piach

Widzisz, babo, to jest znak
Że tu dziada chyba brak
Cały rok byś siała tak
Dziad to ważna rzecz

Idzie baba na ten znak
Prosi dziada, choć nie w smak
To był sprytny fortel dziada
Sekret zdradzić tu wypada
Aby utrzeć nosa babie
Wsypał w worek piach'''

lang = "polish"

model = getModel(lang)

sentences = list(model.process(sampleText))

keyword_parts_of_speech = POSSIBLE_KEYWORD_POS
stopwords = getStopwords(lang)

tokens = []

for sentence in sentences:
    for word in sentence.words:
        if not isinstance(word.lemma, str):
            continue
        if word.form in stopwords or word.lemma in stopwords:
            continue
        if word.upostag not in keyword_parts_of_speech:
            continue
        tokens.append(word.lemma)

print(tokens)


list(sentences[0].words)[1].lemma
