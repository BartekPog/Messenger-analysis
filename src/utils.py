import re
import os
import pandas as pd


def countWords(text: str) -> int:
    pattern = r"[^\W|\d]+"
    words = re.findall(pattern, text)
    return len(words)


def assertDir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def printExecutionTime(startTime: float, endTime: float):
    hours, rem = divmod(endTime-startTime, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExecution time: {:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))
