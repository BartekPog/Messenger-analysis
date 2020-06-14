import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels
import datetime
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from datetime import date

import plots

TIMEZONE = "Europe/Warsaw"
USER = "Bartek Pogod"
LANGUAGE = "polish"

MESSAGES_FILE = "all_messages.csv"
PLOTS_DIR = "figures"

# Reading data
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
plots.plotActivityOverTime(data, user=USER, save_dir=PLOTS_DIR)
plots.plotActivityForMostFrequentNonGroupChats(
    data, chats=4, order=3, save_dir=PLOTS_DIR)
plots.plotActivityOverWeek(data, user=USER, save_dir=PLOTS_DIR)
plots.plotActivityOverDay(data, user=USER, save_dir=PLOTS_DIR)
plots.plotMessageLengthDistributionPerChat(data, user=USER, save_dir=PLOTS_DIR)
plots.plotAverageMessageLength(
    data, user=USER, chats=20, messagesTreshold=0.1, save_dir=PLOTS_DIR)
plots.generateKeywordClouds(
    data, user=USER, language=LANGUAGE, chats=8, save_dir=PLOTS_DIR, background_color="white", lemmatize=False)

# Plot user activity in groups over time
