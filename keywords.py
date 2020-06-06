import os
import re
from corpy.udpipe import Model
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_MODELS_FOLDER = "language_models"
DEFAULT_STOPWORDS_FOLDER = "stopwords"

POSSIBLE_KEYWORD_POS = ["NOUN", "VERB", "PROPN", "ADJ", "ADV"]


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


def getModel(language: str = "english"):
    models = dict(getUdpipeModelNames(DEFAULT_MODELS_FOLDER))

    if language not in models.keys():
        print("ERR: Language '{}' is not available. \n\nCurrently language packs installed locally are: ".format(language))
        for lang in models.keys():
            print(" - ", lang)

        print("\nIf your language is not listed here consider downloading language package from official UDpipe repository. Then just put the file in '{}' folder.".format(DEFAULT_MODELS_FOLDER))
        return None

    return Model(models[language])


def tokenize(text: str, language: str) -> list:
    model = getModel(language)

    if (model == None):
        return []

    sentences = list(model.process(text))

    keyword_parts_of_speech = POSSIBLE_KEYWORD_POS
    stopwords = getStopwords(language)

    tokens = []

    for sentence in sentences:
        for word in sentence.words:
            if not isinstance(word.lemma, str):
                continue
            if word.form in stopwords or word.lemma in stopwords:
                continue
            if word.upostag not in keyword_parts_of_speech:
                continue
            tokens.append(word.lemma.lower())

    return tokens


def preprocessText(text: str, language: str) -> str:
    return ' '.join(tokenize(text, language))


def getKeywords(corpus: list, language: str, keyword_number: int = 15) -> list:
    corpusPrep = [preprocessText(text, language) for text in corpus]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    X = vectorizer.fit_transform(corpusPrep).toarray()

    keywordsPerDoc = []
    for text, scores in zip(corpusPrep, X):
        scoredDoc = [(word, score)
                     for word, score in zip(text.split(), scores)]
        scoredDoc.sort(key=(lambda x: x[1]), reverse=True)

        keywords = []
        numberOfKwds = 0
        for word, _ in scoredDoc:
            if(word not in keywords):
                keywords.append(word)
                numberOfKwds = numberOfKwds+1

            if(numberOfKwds >= N):
                break

        keywordsPerDoc.append(keywords)

    return keywordsPerDoc


sampleText1 = '''Siała baba mak
Nie wiedziała jak
A dziad wiedział
Nie powiedział
A to było tak

Wzięła baba maku wór
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

sampleText2 = '''Stoi na stacji lokomotywa,
Ciężka, ogromna i pot z niej spływa:
Tłusta oliwa.

Stoi i sapie, dyszy i dmucha,
Żar z rozgrzanego jej brzucha bucha:
Uch - jak gorąco!
Puff - jak gorąco!
Uff - jak gorąco!
Już ledwo sapie, już ledwo zipie,
A jeszcze palacz węgiel w nią sypie.
Wagony do niej podoczepiali
Wielkie i ciężkie, z żelaza, stali,
I pełno ludzi w każdym wagonie,
A w jednym krowy, a w drugim konie,
A w trzecim siedzą same grubasy,
Siedzą i jedzą tłuste kiełbasy,
A czwarty wagon pełen bananów,
A w piątym stoi sześć fortepianów,
W szóstym armata - o! jaka wielka!
Pod każdym kołem żelazna belka!
W siódmym dębowe stoły i szafy,
W ósmym słoń, niedźwiedź i dwie żyrafy,
W dziewiątym - same tuczone świnie,
W dziesiątym - kufry, paki i skrzynie,
A tych wagonów jest ze czterdzieści,
Sam nie wiem, co się w nich jeszcze mieści.
Lecz choćby przyszło tysiąc atletów
I każdy zjadłby tysiąc kotletów,
I każdy nie wiem jak się wytężał,
To nie udźwigną, taki to ciężar.
Nagle - gwizd!
Nagle - świst!
Para - buch!
Koła - w ruch!

Najpierw -- powoli -- jak żółw -- ociężale,
Ruszyła -- maszyna -- po szynach -- ospale,
Szarpnęła wagony i ciągnie z mozołem,
I kręci się, kręci się koło za kołem,
I biegu przyspiesza, i gna coraz prędzej,
I dudni, i stuka, łomoce i pędzi,
A dokąd? A dokąd? A dokąd? Na wprost!
Po torze, po torze, po torze, przez most,
Przez góry, przez tunel, przez pola, przez las,
I spieszy się, spieszy, by zdążyć na czas,
Do taktu turkoce i puka, i stuka to:
Tak to to, tak to to , tak to to, tak to to.
Gładko tak, lekko tak toczy się w dal,
Jak gdyby to była piłeczka, nie stal,
Nie ciężka maszyna, zziajana, zdyszana,
Lecz fraszka, igraszka, zabawka blaszana.

A skądże to, jakże to, czemu tak gna?
A co to to, co to to, kto to tak pcha,
Że pędzi, że wali, że bucha buch, buch?
To para gorąca wprawiła to w ruch,
To para, co z kotła rurami do tłoków,
A tłoki kołami ruszają z dwóch boków
I gnają, i pchają, i pociąg się toczy,
Bo para te tłoki wciąż tłoczy i tłoczy,
I koła turkocą, i puka, i stuka to:
Tak to to, tak to to, tak to to, tak to to!...'''

language = "polish"
N = 10

corpus = [sampleText1, sampleText2]

getKeywords(corpus, language, N)
