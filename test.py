import pandas as pd
from textrank import TextRank4Keyword


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


extractor = TextRank4Keyword(language="polish", verbose=True, lemmatize=False)


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


extractor.analyze(sampleText1)
print(extractor.get_keywords(keywordNum))
# zip(names, chatKeywords)a

chatKeywords = []

for chat in corpus[2:5]:
    extractor.analyze(chat)
    chatKeywords.append(extractor.get_keywords(keywordNum))


print("done")
print(list(zip(names, chatKeywords)))

extractor.get_keywords(keywordNum)
