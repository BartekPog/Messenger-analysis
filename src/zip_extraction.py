import zipfile
import os
import pandas as pd
import re
from functools import partial
import json

from .parameters import getParam
from .anonymize import changeNames

DEFAULT_ZIP_FOLDER = getParam('dataZipDirectory')
DEFAULT_OUTPUT_FILE = getParam('allMessagesFile')
USER = getParam('user')
TIMEZONE = getParam('timezone')
ANONYMIZE = getParam('anonymize')


def getZipPath(folderName: str = DEFAULT_ZIP_FOLDER, user: str = USER) -> str:
    prepName = ''.join(user.split()).lower()

    filesNames = [file for file in os.listdir(folderName) if file.endswith(".zip") ]

    if (len(filesNames) == 0):
        print("ERROR: put the zip file in ", folderName)
        return None 
    
    if (len(filesNames) == 1):
        return os.path.join(folderName, filesNames[0])
    
    for fileName in os.listdir(folderName):
        if ((fileName.endswith(".zip")) and (str(prepName) in str(fileName))):
            return os.path.join(folderName, fileName)

    print("ERROR: No zip file for user ", USER)
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


def getDataFrame(folderName: str = DEFAULT_ZIP_FOLDER, user: str = USER, isAnonymous: bool = ANONYMIZE) -> pd.DataFrame:
    zipPath = getZipPath(folderName=folderName, user=user)

    if zipPath == None:
        return None

    dataFrames = []

    with zipfile.ZipFile(zipPath) as zipF:
        for fileDir in zipF.namelist():
            if fileDir.endswith(".json"):
                try:
                    newChat = extractOne(zipF, fileDir, user)
                    dataFrames.append(newChat)
                except:
                    print("WARNING: Wrong chat syntax in "+ fileDir)

    fullDataFrame = pd.concat(dataFrames, ignore_index=True)

    if isAnonymous:
        fullDataFrame = changeNames(fullDataFrame)

    return fullDataFrame


def getDates(data: pd.DataFrame, timezone: str = TIMEZONE) -> pd.DataFrame:
    data["date"] = pd.to_datetime(data["timestamp_ms"]*int(
        1e6), errors="ignore").dt.tz_localize('UTC').dt.tz_convert(timezone).dt.strftime('%Y-%m-%d')

    data["weekday"] = pd.to_datetime(data["timestamp_ms"]*int(
        1e6)).dt.tz_localize('UTC').dt.tz_convert(timezone).dt.strftime('%A')

    data["yearday"] = pd.to_datetime(data["timestamp_ms"]*int(
        1e6)).dt.tz_localize('UTC').dt.tz_convert(timezone).dt.strftime('%j')

    data["hour"] = pd.to_datetime(data["timestamp_ms"]*int(
        1e6)).dt.tz_localize('UTC').dt.tz_convert(timezone).dt.strftime('%H')

    data["minute"] = pd.to_datetime(data["timestamp_ms"]*int(
        1e6)).dt.tz_localize('UTC').dt.tz_convert(timezone).dt.strftime('%M')

    return data


def getMessages(folderName: str = DEFAULT_ZIP_FOLDER, outputName: str = DEFAULT_OUTPUT_FILE, user: str = USER, timezone: str = TIMEZONE, isAnonymous: bool = ANONYMIZE):
    dataFrame = getDataFrame(folderName=folderName,
                             user=user, isAnonymous=isAnonymous)
    if isinstance(outputName, str):
        dataFrame.to_csv(outputName, index=False)

    if isinstance(timezone, str):
        return getDates(dataFrame, timezone)
    else:
        print("WARNING: No timezone provided")
        return dataFrame
