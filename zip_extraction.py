import zipfile
import os
import pandas as pd
import re
from functools import partial
import json


DEFAULT_ZIP_FOLDER = "zips"
DEFAULT_OUTPUT_FILE = "all_messages.csv"
USER = "Bartek Pogod"
# USER = "Sara Zug"


def getZipPath(folderName: str = DEFAULT_ZIP_FOLDER, user: str = USER) -> str:
    prepName = ''.join(user.split()).lower()

    for fileName in os.listdir(folderName):
        if ((fileName.endswith(".zip")) and (str(prepName) in str(fileName))):
            return os.path.join(folderName, fileName)

    return None


def getFileData(zipF: zipfile.ZipFile, fileDir: str) -> dict:
    fix_mojibake_escapes = partial(
        re.compile(rb'\\u00([\da-f]{2})').sub,
        lambda m: bytes.fromhex(m.group(1).decode()))

    with zipF.open(fileDir) as binData:
        repaired = fix_mojibake_escapes(binData.read())

    return json.loads(repaired.decode('utf8'))


def extractOne(zipF: zipfile.ZipFile, fileDir: str, user: str = USER) -> pd.DataFrame:
    conversation = getFileData(zipF, fileDir)

    participants = list(conversation["participants"])

    coParticipantsList = [x["name"] for x in participants if x["name"] != user]

    coParticipant = coParticipantsList[0] if len(
        coParticipantsList) == 1 else "GROUP"

    messages = pd.DataFrame(conversation["messages"])
    messages["chat_with"] = coParticipant
    messages["participants_number"] = len(participants)

    return messages


def getDataFrame(folderName: str = DEFAULT_ZIP_FOLDER, user: str = USER) -> pd.DataFrame:
    zipPath = getZipPath(folderName=folderName, user=user)

    if zipPath == None:
        return None

    dataFrames = []

    with zipfile.ZipFile(zipPath) as zipF:
        for fileDir in zipF.namelist():
            if fileDir.endswith(".json"):
                dataFrames.append(extractOne(zipF, fileDir, user))

    return pd.concat(dataFrames, ignore_index=True)


def getMessagesCsv(folderName: str = DEFAULT_ZIP_FOLDER, outputName: str = DEFAULT_OUTPUT_FILE, user: str = USER):
    dataFrame = getDataFrame(folderName=folderName, user=user)
    if isinstance(outputName, str):
        dataFrame.to_csv(outputName, index=False)

    return dataFrame
