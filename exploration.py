import time

from parameters import getParam
from utils import printExecutionTime
from zip_extraction import getMessages
import plots


TIMEZONE = getParam('timezone')
USER = getParam('user')
LANGUAGE = getParam('language')
ZIP_DIR = getParam('dataZipDirectory')
MESSAGES_FILE = getParam('allMessagesFile')
PLOTS_DIR = getParam('plotsDirectory')
WORDCLOUDS_SUBDIR = getParam('wordClouds')['subDirectory']

startTime = time.time()

# Read data
data = getMessages(ZIP_DIR, outputName=MESSAGES_FILE,
                   user=USER, timezone=TIMEZONE)

# Generate plots
plots.plotMessagesInChats(data, chats=15, user=USER, save_dir=PLOTS_DIR)
plots.plotActivityOverTime(data, user=USER, save_dir=PLOTS_DIR, order=6)
plots.plotActivityForMostFrequentNonGroupChats(
    data, chats=4, order=3, save_dir=PLOTS_DIR)
plots.plotActivityOverWeek(data, user=USER, save_dir=PLOTS_DIR)
plots.plotActivityOverDay(data, user=USER, save_dir=PLOTS_DIR)
plots.plotMessageLengthDistributionPerChat(data, user=USER, save_dir=PLOTS_DIR)
plots.plotAverageMessageLength(
    data, user=USER, chats=20, messages_treshold=0.1, save_dir=PLOTS_DIR)
plots.generateKeywordClouds(
    data, user=USER, language=LANGUAGE, chats=5, save_dir=PLOTS_DIR, clouds_subdir=WORDCLOUDS_SUBDIR, background_color="white")
plots.plotLanguageDiversityRank(
    data, user=USER, language=LANGUAGE, save_dir=PLOTS_DIR, batch_size=500)

# Calculate execution time
endTime = time.time()
# print("done")
printExecutionTime(startTime, endTime)
