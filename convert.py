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
        # repaired =

    data = json.loads(repaired.decode('utf8'))

    # with open(outFile, 'wb') as fixed_binary:
    with open(outFile, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def remakeByHand(inFile, outFile):
    with open(inFile, 'r') as f:
        data = f.read().encode('latin_1').decode('utf-8')

    print(data.encode('latin1').decode('utf8'))

    with open(outFile, 'w') as f:
        f.write(data)

    # import json
    # data = r'"Rados\u00c5\u0082aw"'
    # json.loads(data).encode('latin1').decode('utf8')


def fixFiles(dir):
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if (file.endswith(".json") and not file.startswith("fixed_")):
                inFile = os.path.join(root, file)
                outName = "fixed_"+file
                outFile = os.path.join(root, outName)
                remakeFile(inFile, outFile)
                # remakeByHand(inFile, outFile)


if __name__ == "__main__":
    fixFiles(DATA_DIR)
    print("Done")

# data = r'"Rados\u00c5\u0082aw"'

# data = r'"Kasia Wro\u00c5\u0084ska"'
# json.loads(data).encode('latin_1').decode('utf-8')
