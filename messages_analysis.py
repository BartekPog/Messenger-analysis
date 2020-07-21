import time

from src.utils import printExecutionTime
from src.zip_extraction import getMessages
from src.plots_generation import generatePlots


def runFullAnalysis():
    startTime = time.time()

    print("Parsing data")
    data = getMessages()

    print("Generating plots")
    generatePlots(data)

    print("Done")
    endTime = time.time()
    printExecutionTime(startTime, endTime)


if __name__ == "__main__":
    runFullAnalysis()
