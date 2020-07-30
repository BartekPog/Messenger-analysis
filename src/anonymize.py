import pandas as pd
import names


def getNamesDict(data: pd.DataFrame) -> dict:
    realNames = set(data["chat_with"].value_counts().index)
    realNames.remove("GROUP")

    namesMap = dict()

    for realName in realNames:

        newName = names.get_full_name()
        while newName in namesMap.values():
            newName = names.get_full_name()

        namesMap[realName] = newName

    return namesMap


def changeName(name: str, namesMap: dict) -> str:
    if name in namesMap.keys():
        return namesMap[name]

    return name


def changeRow(row: pd.Series, namesMap: dict) -> pd.Series:
    newRow = row.copy()
    newRow['sender_name'] = changeName(newRow['sender_name'], namesMap)
    newRow['chat_with'] = changeName(newRow['chat_with'], namesMap)

    return newRow


def changeNames(data: pd.DataFrame) -> pd.DataFrame:
    namesMap = getNamesDict(data)

    return data.apply(changeRow, axis=1, args=(namesMap,))
