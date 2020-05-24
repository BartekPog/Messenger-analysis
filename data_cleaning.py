import json
import re
import os
import pandas as pd
from functools import partial

DATA_DIR = "inbox/"
GENERATE_OUTPUT_CSV = True
OUTPUT_CSV_NAME = "all_messages.csv"
USER = "Bartek Pogod"
TIMEZONE = "Europe/Warsaw"


def remakeFile(inFile, outFile):
    fix_mojibake_escapes = partial(
        re.compile(rb'\\u00([\da-f]{2})').sub,
        lambda m: bytes.fromhex(m.group(1).decode()))

    with open(inFile, 'rb') as binary_data:
        repaired = fix_mojibake_escapes(binary_data.read())

    data = json.loads(repaired.decode('utf8'))

    with open(outFile, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def fixFiles(dir):
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if (file.endswith(".json") and not file.startswith("fixed_")):
                inFile = os.path.join(root, file)
                outName = "fixed_"+file
                outFile = os.path.join(root, outName)
                remakeFile(inFile, outFile)


def getFileDirs(rootDir: str) -> list:
    fileDirs = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if (file.endswith(".json") and file.startswith("fixed_")):
                fileDirs.append(os.path.join(root, file))
    return fileDirs


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


if __name__ == "__main__":
    print("Fixing unicode encoding")
    fixFiles(DATA_DIR)

    print("Importing data")
    data = getAllConversations(DATA_DIR)

    print("Saving CSV")
    data.to_csv(OUTPUT_CSV_NAME, index=False)

    print("Done")
