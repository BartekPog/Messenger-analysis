import spacy
import numpy as np
# from spacy.parts_of_speech import PROPN

LANGUAGE = "polish"

text = "Pan Tadeusz jest najsłynniejszym dziełem Adama Mickiewicza. To spisana trzynastozgłoskowcem, zawarta w dwunastu księgach opowieść o szlachcie polskiej początku XIX wieku"
text2 = "amerykańskie przedsiębiorstwo przemysłu kosmicznego, założone w roku 2002 przez Elona Muska. Jego 21 celem jest budowa silników rakietowych i rakiet nośnych oraz statków kosmicznych, w tym także załogowych. Kluczem do osiągnięcia sukcesu ma być znaczne zmniejszenie kosztów wynoszenia ładunku na orbitę. Przede wszystkim, SpaceX projektuje i buduje serię rakiet orbitalnych Falcon i statków kosmicznych Dragon."

language = LANGUAGE

langDict = {
    "polish": "pl_core_news_md",
    "english": "en_core_web_sm"
}

print("importing")
nlp = spacy.load(langDict[language], disable=["parser"])
print("The rest")

doc = nlp(text2)
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

for token in doc:
    print(token, token.pos_, token.tag_,
          token.dep_, token.shape_, token.is_stop)


def removeEntities(doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
    entityRanges = np.concatenate(
        [np.arange(e.start, e.end) for e in doc.ents])

    return [token for token in doc if token.i not in entityRanges]


def removePunctuation(doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
    return [token for token in doc if token.pos_ != "PUNCT"]


def removeNumbers(doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
    return [token for token in doc if token.pos_ != "NUM"]


for entity in doc.ents:
    print(entity.text, entity.label_, entity.start, entity.end)


# TODO REMEMBER ABOUT BATCHES
