from collections import OrderedDict
import numpy as np
import os
import re
import pandas as pd
from corpy.udpipe import Model
from scipy.sparse import csr_matrix

from .parameters import getParam

DEFAULT_STOPWORDS_FOLDER = getParam('stopwordsDirectory')


def getStopwords(language: str, stopwords_dir: str = DEFAULT_STOPWORDS_FOLDER) -> list:
    for root, _, files in os.walk(stopwords_dir):
        for file in files:
            if (file.endswith("stopwords.txt") and file.startswith(language)):
                with open(os.path.join(root, file), "r") as f:
                    return [x.strip() for x in f.readlines()]

    return []


class NGramExtractor():
    def __init__(self, language: str, IDFCorpus: list = None, max_n: int = 4, stopwords_dir: str = DEFAULT_STOPWORDS_FOLDER, following_character_limit: int = 2):
        self.d = 0.85  # damping coefficient, usually is .85
        self.steps = 400  # iteration steps
        self.batchTokens = 1400  # tokens in one extraction batch
        self.max_n = max_n  # max n_gram
        self.language = language
        self.stopwords = getStopwords(
            language=language, stopwords_dir=stopwords_dir)

        self.followingCharacterLimit = following_character_limit
        self.node_weight = None
        self.idfCorpus = None
        if(IDFCorpus != None):
            self.initIDFCorpus(IDFCorpus)

    def initIDFCorpus(self, corpus: list):
        idfs = list()

        for n_gram in range(1, self.max_n + 1):
            nGramOccurences = dict()

            for doc in corpus:
                sentences = self.get_sentence_tokens(text=doc, n_gram=n_gram)
                tokensInDoc = set()

                for sentence in sentences:
                    for token in sentence:
                        tokensInDoc.add(token)

                for token in tokensInDoc:
                    if token in nGramOccurences.keys():
                        nGramOccurences[token] = nGramOccurences[token] + 1
                    else:
                        nGramOccurences[token] = 1

            numberOfDocs = len(corpus)

            nGramIDF = {k: np.log(numberOfDocs/float(v))
                        for k, v in nGramOccurences.items()}
            idfs.append(nGramIDF)
        # print(idfs)
        self.idfCorpus = idfs

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
            if (counter < self.followingCharacterLimit):
                counter = counter+1
                noRepetitions.append(char)

        return "".join(noRepetitions)

    def get_n_grams(self, sentence: list, grams: int):
        tokens = []
        for i in range(len(sentence)-grams):
            tokenSequence = sentence[i:i+grams]
            if (tokenSequence[0] in self.stopwords or tokenSequence[-1] in self.stopwords):
                continue

            token = " ".join(tokenSequence)
            tokens.append(token)
        return tokens

    def get_sentence_tokens(self, text: str, n_gram: int):
        cleanText = self.clean_text(text)

        processedSentences = []
        for sentence in cleanText.split('.'):
            words = [word for word in sentence.split(
            ) if word.isalpha()]

            tokens = self.get_n_grams(words, n_gram)
            processedSentences.append(tokens)

        return processedSentences

    def chunkize(self, sentences):
        chunks = []
        chunk = []
        tokenCount = 0
        for sentence in sentences:
            if len(sentence) + tokenCount < self.batchTokens:
                chunk.append(sentence)
                tokenCount = tokenCount + len(sentence)
            else:
                tokenCount = 0
                chunks.append(chunk)
                chunk = [sentence]

        chunks.append(chunk)
        return chunks

    def get_vocab(self, sentences):
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        token_pairs = set()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.add(pair)
        return token_pairs

    def get_matrix(self, vocab, token_pairs):
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            if i == j:
                continue
            g[j][i] = g[j][i] + 1
            g[i][j] = g[i][j] + 1

        g_sparse = csr_matrix(g)

        norm = np.sum(g, axis=0)
        norm[norm == 0] = 1

        g_sp_norm = g_sparse._divide(norm)

        return csr_matrix(g_sp_norm)

    def get_keywords_with_values(self, node_weights, part=0.7):
        node_weight = OrderedDict(
            sorted(node_weights.items(), key=lambda t: t[1], reverse=True))
        keywords = []

        number = len(node_weight.items())*part

        for i, (key, value) in enumerate(node_weight.items()):
            keywords.append((key, value))
            if i > number:
                break
        return keywords

    def get_keywords_for_chunk(self, sentences, window_size=2):
        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        for _ in range(self.steps):
            pr = (1-self.d) + self.d * g.dot(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        keywords = self.get_keywords_with_values(node_weight, part=0.7)
        return keywords

    def analyze(self, text,
                window_size=2, n_gram: int = 2, keywords_number: int = 10):
        """Main function to analyze text"""

        # Parse text
        allSentences = self.get_sentence_tokens(text, n_gram)

        chunks = self.chunkize(allSentences)

        keywordValue = dict()

        for chunk in chunks:
            chunkKeywords = self.get_keywords_for_chunk(
                chunk, window_size)

            for kwd in chunkKeywords:
                if kwd[0] in keywordValue.keys():
                    keywordValue[kwd[0]] = keywordValue[kwd[0]] + kwd[1]
                else:
                    keywordValue[kwd[0]] = kwd[1]

        if self.idfCorpus != None:
            finalKwdValue = {
                k: v*self.idfCorpus[n_gram-1][k] for k, v in keywordValue.items()}

        else:
            finalKwdValue = keywordValue

        keywordsList = sorted(finalKwdValue.items(),
                              key=lambda item: item[1], reverse=True)

        return keywordsList[:keywords_number]
