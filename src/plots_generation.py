import pandas as pd

from .plots import plotMessagesInChats, plotActivityOverTime, plotActivityForMostFrequentNonGroupChats, plotActivityOverWeek, plotActivityOverDay, plotMessageLengthDistributionPerChat, plotAverageMessageLength, generateKeywordClouds, plotLanguageDiversityRank
from .parameters import getParam

USER = getParam('user')
LANGUAGE = getParam('language')
PLOTS_DIR = getParam('plotsDirectory')
WORDCLOUDS_SUBDIR = getParam('wordClouds')['subDirectory']


def generatePlots(data: pd.DataFrame):
    plotMessagesInChats(data, chats=15, user=USER, save_dir=PLOTS_DIR)

    plotActivityOverTime(data, user=USER, save_dir=PLOTS_DIR, order=6)

    plotActivityForMostFrequentNonGroupChats(
        data, chats=4, order=3, save_dir=PLOTS_DIR)

    plotActivityOverWeek(data, user=USER, save_dir=PLOTS_DIR)

    plotActivityOverDay(data, user=USER, save_dir=PLOTS_DIR)

    plotMessageLengthDistributionPerChat(
        data, user=USER, save_dir=PLOTS_DIR)

    plotAverageMessageLength(
        data, user=USER, chats=20, messages_treshold=0.1, save_dir=PLOTS_DIR)

    generateKeywordClouds(
        data, user=USER, language=LANGUAGE, chats=10, save_dir=PLOTS_DIR, clouds_subdir=WORDCLOUDS_SUBDIR, background_color="white")

    plotLanguageDiversityRank(
        data, user=USER, language=LANGUAGE, save_dir=PLOTS_DIR, batch_size=500)
