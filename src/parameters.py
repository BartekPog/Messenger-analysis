import json

PARAM_FILE = 'params.json'


def getParams(paramFile: str = PARAM_FILE) -> dict:
    with open(paramFile, "r", encoding='utf-8') as file:
        params = json.load(file)

    return params


def getParam(paramKey: str, paramFile: str = PARAM_FILE):
    params = getParams(paramFile=paramFile)

    return params[paramKey]
