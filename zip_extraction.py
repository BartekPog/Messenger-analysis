import zipfile
import os
import pandas as pd
import re
from functools import partial
import json

from parameters import getParam

DEFAULT_ZIP_FOLDER = getParam('dataZipDirectory')
DEFAULT_OUTPUT_FILE = getParam('allMessagesFile')
USER = getParam('user')
TIMEZONE = getParam('timezone')


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


def getDates(data: pd.DataFrame, timezone: str = TIMEZONE) -> pd.DataFrame:
    data["date"] = pd.to_datetime(data["timestamp_ms"]*int(
        1e6)).dt.tz_localize('UTC').dt.tz_convert(timezone).dt.strftime('%Y-%m-%d')

    data["weekday"] = pd.to_datetime(data["timestamp_ms"]*int(
        1e6)).dt.tz_localize('UTC').dt.tz_convert(timezone).dt.strftime('%A')

    data["yearday"] = pd.to_datetime(data["timestamp_ms"]*int(
        1e6)).dt.tz_localize('UTC').dt.tz_convert(timezone).dt.strftime('%j')

    data["hour"] = pd.to_datetime(data["timestamp_ms"]*int(
        1e6)).dt.tz_localize('UTC').dt.tz_convert(timezone).dt.strftime('%H')

    data["minute"] = pd.to_datetime(data["timestamp_ms"]*int(
        1e6)).dt.tz_localize('UTC').dt.tz_convert(timezone).dt.strftime('%M')

    return data


def getMessages(folderName: str = DEFAULT_ZIP_FOLDER, outputName: str = DEFAULT_OUTPUT_FILE, user: str = USER, timezone: str = TIMEZONE):
    dataFrame = getDataFrame(folderName=folderName, user=user)
    if isinstance(outputName, str):
        dataFrame.to_csv(outputName, index=False)

    if(timezone != None):
        return getDates(dataFrame, timezone)
    else:
        return dataFrame
