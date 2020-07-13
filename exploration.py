import random
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels
import datetime
import re
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from datetime import date

import plots
from n_gram_extractor import NGramExtractor

TIMEZONE = "Europe/Warsaw"
USER = "Bartek Pogod"
# USER = "Sara Zug"

LANGUAGE = "polish"

MESSAGES_FILE = "all_messages.csv"
PLOTS_DIR = "figures"
WORDCLOUDS_SUBDIR = "wordclouds"

# Reading data
startTime = time.time()

data = pd.read_csv(MESSAGES_FILE)


data["date"] = pd.to_datetime(data["timestamp_ms"]*int(
    1e6)).dt.tz_localize('UTC').dt.tz_convert(TIMEZONE).dt.strftime('%Y-%m-%d')

data["weekday"] = pd.to_datetime(data["timestamp_ms"]*int(
    1e6)).dt.tz_localize('UTC').dt.tz_convert(TIMEZONE).dt.strftime('%A')

data["yearday"] = pd.to_datetime(data["timestamp_ms"]*int(
    1e6)).dt.tz_localize('UTC').dt.tz_convert(TIMEZONE).dt.strftime('%j')

data["hour"] = pd.to_datetime(data["timestamp_ms"]*int(
    1e6)).dt.tz_localize('UTC').dt.tz_convert(TIMEZONE).dt.strftime('%H')

data["minute"] = pd.to_datetime(data["timestamp_ms"]*int(
    1e6)).dt.tz_localize('UTC').dt.tz_convert(TIMEZONE).dt.strftime('%M')


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

endTime = time.time()
hours, rem = divmod(endTime-startTime, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
