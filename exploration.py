import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = 'inbox/'
USER = "Bartek Pogod"


def getFileDirs(rootDir: str) -> list:
    fileDirs = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if (file.endswith(".json") and file.startswith("fixed_")):
                fileDirs.append(os.path.join(root, file))
    return fileDirs


fileDirs = getFileDirs(DATA_DIR)


def readOne(fileName: str):
    with open(fileName, "r") as f:
        conversation = json.load(f)

    participants = list(conversation["participants"])

    coParticipantsList = [x["name"] for x in participants if x["name"] != USER]

    coParticipant = coParticipantsList[0] if len(
        coParticipantsList) == 1 else "GROUP"

    messages = pd.DataFrame(conversation["messages"])
    messages["chat_with"] = coParticipant

    return messages


def getAllConversations(root: str):
    fileDirs = getFileDirs(root)
    allConversations = pd.concat([readOne(f)for f in fileDirs])

    return allConversations


data = getAllConversations(DATA_DIR)

# All activity over time
sns.kdeplot(data['timestamp_ms'], shade=True)
plt.show()

# my activity over time vs others
withoutUser = data[~data['sender_name'].isin([USER])]
withUser = data[data['sender_name'].isin([USER])]
ax = sns.kdeplot(withoutUser['timestamp_ms'], legend=False, shade=True)
ax = sns.kdeplot(withUser['timestamp_ms'], legend=False, shade=True, ax=ax)
ax.set(xlabel="Time (timestamps)", ylabel="messages frequency")
# change legend texts

# leg.set_title(new_title)
# new_labels = ['label 1', 'label 2']
# for t, l in zip(leg.texts, new_labels): t.set_text(l)

# sns.plt.show()
plt.show()


data.head()
# plot values in time


# TODO
# withoutUser = data[~data['sender_name'].isin([USER])]

# g = sns.FacetGrid(withoutUser, col="sender_name", col_wrap=5,
#                   margin_titles=True, height=2.5)
# g.map(sns.kdeplot, shade=True)
# # g.map(sns.regplot, "timestamp_ms", "score", order=1, color="green")
# plt.show()
