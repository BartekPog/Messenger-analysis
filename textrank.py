from collections import OrderedDict
import numpy as np
# import keywords
import os
import re
import pandas as pd
from corpy.udpipe import Model
from scipy.sparse import csr_matrix


DEFAULT_MODELS_FOLDER = "language_models"
DEFAULT_STOPWORDS_FOLDER = "stopwords"

POSSIBLE_KEYWORD_POS = ["NOUN", "PROPN"]  # "PROPN"


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


class TextRank4Keyword():
    """Extract keywords from text"""

    def __init__(self, language: str, lemmatize: bool = False, verbose: bool = True, stopwords_dir: str = DEFAULT_STOPWORDS_FOLDER, models_dir: str = DEFAULT_MODELS_FOLDER, pos: list = POSSIBLE_KEYWORD_POS, following_character_limit: int = 2):
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 300  # iteration steps
        self.node_weight = None  # save keywords and its weight
        self.language = language
        self.stopwords = getStopwords(
            language=language, stopwords_dir=stopwords_dir)
        if(lemmatize):
            self.lemmaModel = getModel(
                language=language, models_dir=models_dir)
        else:
            self.lemmaModel = None
        self.partsOfSpeech = pos
        self.verbose = verbose
        self.lemmatize = lemmatize
        self.followingCharacterLimit = following_character_limit

    def clean_text(self, text: str):
        noLinks = " ".join([word.lower() for word in text.split()
                            if not word.startswith("http")])
        if(self.followingCharacterLimit == None):
            return noLinks

        noRepetitions = []
        chars = [char for char in noLinks]

        counter = 0
        currchar = '\n'

        for char in chars:
            if char != currchar:
                currchar = char
                counter = 1
                noRepetitions.append(char)
                continue
            if (counter <= self.followingCharacterLimit):
                counter = counter+1
                noRepetitions.append(char)

        return "".join(noRepetitions)
        # return ' '.join(re.sub(r"(@[A-Za-z0-9]+) | ([ ^ 0-9A-Za-z \t]) | (\w+: \/\/\S+)", " ", text).split())

    def get_n_grams(self, sentence: list, grams: int):
        tokens = []
        for i in range(len(sentence)-grams):
            token = " ".join(sentence[i:i+grams])
            tokens.append(token)
        return tokens

    def get_sentence_tokens(self, text: str, n_gram: int):
        # cleanText = self.clean_text(text)
        cleanText = self.clean_text(text)

        if self.lemmatize:
            modelSentences = list(self.lemmaModel.process(cleanText))

            processedSentences = []

            for sentence in modelSentences:
                sentenceTokens = []
                for word in sentence.words:
                    if not isinstance(word.lemma, str):
                        continue
                    if word.form in self.stopwords or word.lemma in self.stopwords:
                        continue
                    if word.upostag not in self.partsOfSpeech:
                        continue
                    if (len(word.lemma) < 3):
                        continue

                    sentenceTokens.append(word.lemma.lower())

                tokens = self.get_n_grams(sentenceTokens, n_gram)
                processedSentences.append(tokens)

        else:
            processedSentences = []
            for sentence in cleanText.split('.'):
                words = [word for word in sentence.split(
                ) if word not in self.stopwords and len(word) > 2 and word.isalpha()]

                tokens = self.get_n_grams(words, n_gram)
                processedSentences.append(tokens)

        return processedSentences

    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        # token_pairs = list()
        token_pairs = set()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        # token_pairs.append(pair)
                        token_pairs.add(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            if i == j:
                continue
            g[j][i] = g[j][i] + 1
            g[i][j] = g[i][j] + 1

        print("Normalizing")
        g_sparse = csr_matrix(g)

        norm = np.sum(g, axis=0)
        norm[norm == 0] = 1

        print("Dividing")
        g_sp_norm = g_sparse._divide(norm)

        return csr_matrix(g_sp_norm)

    def print_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(
            sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            print(key + ' - ' + str(value))
            if i > number:
                break

    def get_keywords(self, number=10):
        node_weight = OrderedDict(
            sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        keywords = []
        for i, (key, _) in enumerate(node_weight.items()):
            # print(key + ' - ' + str(value))
            keywords.append(key)
            if i > number:
                break
        return keywords

    def analyze(self, text,
                window_size=2, n_gram: int = 2):
        """Main function to analyze text"""

        # Pare text
        if self.verbose:
            print("Transforming text")
        sentences = self.get_sentence_tokens(text, n_gram)

        # Build vocabulary
        vocab = self.get_vocab(sentences)
        if self.verbose:
            print("Vocab length: {}".format(len(vocab)))

        # Get token_pairs from windows
        if self.verbose:
            print("Getting neighbouring words")
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        if self.verbose:
            print("Setting up graph")

        g = self.get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        if self.verbose:
            print("Iterating")
        # previous_pr = 0
        for epoch in range(self.steps):

            pr = (1-self.d) + self.d * g.dot(pr)
            # if abs(previous_pr - pr.sum()) < self.min_diff:
            #     print(pr.sum(), abs(previous_pr - pr.sum()), self.min_diff)
            #     print("breaking at {}".format(epoch))
            #     break
            # else:
            # previous_pr = pr.sum()

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight
