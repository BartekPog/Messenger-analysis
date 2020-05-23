import json
import re
import os
from functools import partial

DATA_DIR = "inbox/"


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
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if (file.endswith(".json") and not file.startswith("fixed_")):
                inFile = os.path.join(root, file)
                outName = "fixed_"+file
                outFile = os.path.join(root, outName)
                remakeFile(inFile, outFile)


if __name__ == "__main__":
    fixFiles(DATA_DIR)
    print("Done")
