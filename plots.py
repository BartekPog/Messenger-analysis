import pandas as pd
import seaborn as sns
import statsmodels
import datetime
import os
import random
import re
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from datetime import date
from wordcloud import WordCloud

from textrank import KeywordExtractor

FONT_PATH = "fonts"
FONT_NAME = "Righteous-Regular.ttf"
COLORS = ["twilight", "cividis", "copper", "bone", "magma"]

sns.set()


def assertDir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plotMessagesInChats(data: pd.DataFrame, chats: int, user: str, save_dir: str = None):
    plotName = "messages-in-chats"

    noGroup = data[data["chat_with"] != "GROUP"]
    plotDataSeries = noGroup["chat_with"].value_counts()[:chats]

    plotData = pd.DataFrame(plotDataSeries)
    plotData["person"] = plotData.index
    plotData["messages_number"] = plotData["chat_with"]
    ax = sns.barplot(x=plotData["messages_number"],
                     y=plotData["person"], orient="h")
    ax.grid(True)
    ax.set_title("{} {}'s chats with the most messages".format(chats, user))
    ax.set_ylabel('Chat participant')
    ax.set_xlabel('Total number of messages')

    if save_dir == None:
        plt.show()
    else:
        assertDir(save_dir)
        fullPath = os.path.join(save_dir, plotName+".png")
        ax.figure.savefig(fullPath, bbox_inches='tight')


def plotActivityOverTime(data: pd.DataFrame, user: str, save_dir: str = None):
    plotName = "activity-over-time"

    noGroup = data[data["chat_with"] != "GROUP"]

    noGroup["sent_by_user"] = noGroup["sender_name"] == user

    byDates = noGroup.groupby(["date", "sent_by_user"], as_index=True).agg([
        'count'])

    dates = pd.date_range(min(data["date"]), max(data["date"]))

    plotting = byDates.reindex(dates, level=0).reset_index()

    plotting["messages_per_day"] = plotting[("sender_name", "count")]
    plotting["Message direction"] = plotting["sent_by_user"].apply(
        lambda x: "Sent" if x else "Received")

    plotting["date_float"] = plotting["date"].values.astype(float)

    g = sns.lmplot(data=plotting, x="date_float", y="messages_per_day",
                   hue="Message direction", scatter=False, order=5, legend_out=False, aspect=1.7, palette="Set1")

    g.set(xlim=(plotting["date_float"].min(), plotting["date_float"].max()))

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Average messages number over time")
    g.axes[0, 0].yaxis.set_major_locator(MultipleLocator(10))

    avgNanoSecondsInMonth = 2628288*int(1e9)
    numberOfMonths = 2

    g.axes[0, 0].xaxis.set_major_locator(
        MultipleLocator(avgNanoSecondsInMonth*numberOfMonths))
    g.axes[0, 0].set_xlabel('Time')
    g.axes[0, 0].set_ylabel('Messages per day')

    xticks = g.axes[0, 0].get_xticks()
    xticks_dates = [datetime.datetime.fromtimestamp(
        x/int(1e9)).strftime(' %b %Y') for x in xticks]
    g.axes[0, 0].set_xticklabels(
        xticks_dates,  rotation=45, horizontalalignment='right')

    if save_dir == None:
        plt.show()
    else:
        assertDir(save_dir)
        fullPath = os.path.join(save_dir, plotName+".png")
        g.savefig(fullPath)


def plotActivityForMostFrequentNonGroupChats(data: pd.DataFrame, chats: int, order: int, save_dir: str = None):
    plotName = "activity-for-most-frequent-non-group-chats"

    legendOut = True if chats > 5 else False

    noGroup = data[data["chat_with"] != "GROUP"]
    plotDataSeries = noGroup["chat_with"].value_counts()[:chats]

    names = [name[0] for name in plotDataSeries.items()]

    onlyChosen = noGroup[noGroup["chat_with"].isin(names)]

    byDates = onlyChosen.groupby(["date", "chat_with"], as_index=True).agg(
        'count')

    dates = pd.date_range(min(data["date"]), max(data["date"]))

    plotting = byDates.reindex(dates, level=0).reset_index()

    plotting["messages_per_day"] = plotting["sender_name"]

    plotting["date_float"] = plotting["date"].values.astype(float)

    cat_type = pd.api.types.CategoricalDtype(categories=names, ordered=True)

    plotting['chat_with'] = plotting['chat_with'].astype(cat_type)

    g = sns.lmplot(data=plotting, x="date_float", y="messages_per_day", hue="chat_with",
                   scatter=False, order=order, legend_out=legendOut, aspect=1.7, ci=None)

    g.set(xlim=(plotting["date_float"].min(),
                plotting["date_float"].max()), ylim=(0, None))

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Average messages number in most frequent chats")
    g.axes[0, 0].yaxis.set_major_locator(MultipleLocator(20))
    g.axes[0, 0].yaxis.set_minor_locator(MultipleLocator(10))

    avgSecondsInMonth = 2628288*int(1e9)
    numberOfMonths = 2

    g.axes[0, 0].xaxis.set_major_locator(
        MultipleLocator(avgSecondsInMonth*numberOfMonths))
    g.axes[0, 0].set_xlabel('Time')
    g.axes[0, 0].set_ylabel('Messages per day')
    g._legend.set_title("Chat")

    xticks = g.axes[0, 0].get_xticks()
    xticks_dates = [datetime.datetime.fromtimestamp(
        x/int(1e9)).strftime(' %b %Y') for x in xticks]
    g.axes[0, 0].set_xticklabels(
        xticks_dates,  rotation=45, horizontalalignment='right')

    if save_dir == None:
        plt.show()
    else:
        assertDir(save_dir)
        fullPath = os.path.join(save_dir, plotName+".png")
        g.savefig(fullPath)


def plotActivityOverWeek(data: pd.DataFrame, user: str, save_dir: str = None):
    plotName = "activity-over-week"

    noGroup = data[data["chat_with"] != "GROUP"]

    noGroup["sent_by_user"] = noGroup["sender_name"] == user

    plotting = noGroup.groupby(["weekday", "sent_by_user"], as_index=True).agg([
        'count']).reset_index()

    numberOfDays = len(pd.period_range(min(data["date"]), max(data["date"])))

    plotting["average_messages_per_day"] = plotting[(
        'sender_name', 'count')]/(numberOfDays/7)

    plotting["message_direction"] = plotting["sent_by_user"].apply(
        lambda x: 'Sent' if x else 'Received')

    cats = ['Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday']

    cat_type = pd.api.types.CategoricalDtype(categories=cats, ordered=True)

    plotting['weekday'] = plotting['weekday'].astype(cat_type)

    kwargs = {"saturation": 0.5}

    g = sns.catplot(x="weekday", y="average_messages_per_day", hue="message_direction", data=plotting,
                    height=6, kind="bar", palette="Set1", **kwargs)

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Average activity over week")
    g._legend.set_title("Message direction")
    g.axes[0, 0].set_xlabel('Day of the week')
    g.axes[0, 0].set_ylabel('Messages per day')
    g.axes[0, 0].set_xticklabels(
        g.axes[0, 0].get_xticklabels(), horizontalalignment='right',  rotation=45)

    if save_dir == None:
        plt.show()
    else:
        assertDir(save_dir)
        fullPath = os.path.join(save_dir, plotName+".png")
        g.savefig(fullPath)


def plotActivityOverDay(data: pd.DataFrame, user: str, save_dir: str = None):
    plotName = "activity-over-day"
    noGroup = data[data["chat_with"] != "GROUP"]

    noGroup["sent_by_user"] = noGroup["sender_name"] == user

    plotting = noGroup.groupby(
        ["hour", "sent_by_user"]).agg("count").reset_index()

    numberOfHours = len(pd.period_range(min(data["date"]), max(data["date"])))

    plotting["avg_messages_per_hour"] = plotting["sender_name"]/numberOfHours

    plotting["message_direction"] = plotting["sent_by_user"].apply(
        lambda x: 'Sent' if x else 'Received')

    plotting["hour_num"] = plotting["hour"].astype(int)

    g = sns.lmplot(data=plotting, x="hour_num", y="avg_messages_per_hour",
                   hue="message_direction", scatter=False, order=4, legend_out=False, aspect=1.7, palette="Set1")

    g.axes[0, 0].xaxis.set_major_locator(
        MultipleLocator(2))
    g.set(xlim=(0, 24))

    g.ax.legend(loc=2)

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Average messages number over day")
    g.axes[0, 0].set_xlabel('Hour')
    g.axes[0, 0].set_ylabel('Number of messages')

    if save_dir == None:
        plt.show()
    else:
        assertDir(save_dir)
        fullPath = os.path.join(save_dir, plotName+".png")
        g.savefig(fullPath)


def generateKeywordClouds(data: pd.DataFrame, user: str, language: str = "polish", chats: int = 6, keyword_numbers=(20, 17, 6, 3), save_dir: str = None, background_color: str = "white", lemmatize: bool = False):
    plotName = "keywordCloud"

    noGroup = data[data["chat_with"] != "GROUP"]
    plotDataSeries = noGroup["chat_with"].value_counts()[:chats]
    names = [name[0] for name in plotDataSeries.items()]

    extractor = KeywordExtractor(
        language=language, verbose=False, lemmatize=lemmatize)

    for idx, name in enumerate(names):
        oneChat = data[data["chat_with"] == name]
        chatString = ". ".join(oneChat["content"].astype(str).values)
        if(background_color == "black"):
            fig = plt.figure()
            fig.patch.set_facecolor('black')

        keywordFreq = dict()

        for n_gram, number in enumerate(keyword_numbers, start=1):
            newKeywords = extractor.analyze(
                chatString, keywords_number=number, n_gram=n_gram, window_size=4)

            for word, value in newKeywords:
                # n_gram/len(keyword_numbers)
                keywordFreq[word.capitalize()] = 100*value

        fontPath = os.path.abspath(os.path.join(FONT_PATH, FONT_NAME))

        wordcloud = WordCloud(background_color=background_color, font_path=fontPath, colormap=random.choice(COLORS), width=2000,
                              height=1200).generate_from_frequencies(keywordFreq)
        plt.figure(figsize=(30, 16))
        plt.axis("off")
        plt.title(name, fontdict={"fontsize": 50, "fontweight": 7}, pad=7)

        plt.imshow(wordcloud, interpolation="bilinear")

        if save_dir == None:
            plt.show()
        else:
            assertDir(save_dir)
            fullPath = os.path.join(
                save_dir, "-".join([plotName, str(idx), name])+".png")
            plt.savefig(fullPath)


def countWords(text: str) -> int:
    pattern = r"[^\W|\d]+"
    words = re.findall(pattern, text)
    return len(words)


def plotMessageLengthDistributionPerChat(data: pd.DataFrame, user: str, chats: int = 6, bins: int = 12, save_dir: str = None):
    plotName = "message-length-distribution-per-chat"

    generic = data[data["type"] == "Generic"]
    noGroup = generic[generic["chat_with"] != "GROUP"]

    plotDataSeries = noGroup["chat_with"].value_counts()[:chats]
    names = [name[0] for name in plotDataSeries.items()]

    plotting = noGroup.dropna(subset=["content"])

    plotting = plotting[plotting["chat_with"].isin(names)]

    plotting["message_length"] = plotting["content"].apply(countWords)
    plotting = plotting[plotting["message_length"] > 0]

    plotting["message_direction"] = plotting["sender_name"].apply(
        lambda x: "Sent" if x == user else "Received")

    cat_type = pd.api.types.CategoricalDtype(categories=names, ordered=True)

    plotting['chat_with'] = plotting['chat_with'].astype(cat_type)

    logMin = np.log10(plotting["message_length"].min())
    logMax = np.log10(plotting["message_length"].max())

    newBins = np.logspace(logMin, logMax, bins)

    g = sns.FacetGrid(plotting, col="chat_with", hue="message_direction",
                      col_wrap=2, aspect=1.4, sharex=True, sharey=True, margin_titles=True, palette="Set1")

    g.map(sns.distplot, "message_length", bins=newBins,
          hist=True, kde=False, norm_hist=True).set(xscale='log', yscale='log')
    g.set_titles("{col_name}")
    g.add_legend(title="Message direction")
    g.set_axis_labels(x_var="Number of words in message",
                      y_var="Percentage of messages")

    yTicks = g.axes[0].get_yticks()
    newYTicks = [str(tick*100)+'%' for tick in yTicks]
    g.set_yticklabels(newYTicks)

    xTicks = g.axes[0].get_xticks()
    newXTicks = [str(int(tick)) for tick in xTicks]
    g.set_xticklabels(newXTicks)

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Message length distribution per chat')

    if save_dir == None:
        plt.show()
    else:
        assertDir(save_dir)
        fullPath = os.path.join(save_dir, plotName+".png")
        g.savefig(fullPath)


def plotAverageMessageLength(data: pd.DataFrame, user: str, chats: int = 20, messagesTreshold: float = 0.1, save_dir: str = None):
    plotName = "average-message-length-in-significant-chats"

    noGroup = data[data["chat_with"] != "GROUP"]
    noGroup = noGroup[noGroup["type"] == "Generic"]
    allNames = noGroup["chat_with"].value_counts()

    namesNum = (int(len(allNames)*messagesTreshold)
                ) if (int(len(allNames)*messagesTreshold)) > chats else chats

    possibleNames = allNames.index[:namesNum]

    prep = noGroup[noGroup["chat_with"].isin(possibleNames)]
    prep = prep.dropna(subset=["content"])
    prep["message_length"] = prep["content"].apply(countWords)
    prep = prep[prep["message_length"] > 0]
    prep["message_direction"] = prep["sender_name"].apply(
        lambda x: "Sent" if x == user else "Received")

    names = prep.groupby("chat_with").mean().reset_index()

    names = names.sort_values(["message_length"], ascending=False)

    namesList = list(names["chat_with"])[:chats]

    catNames = pd.api.types.CategoricalDtype(
        categories=namesList, ordered=True)

    plotting = prep[prep["chat_with"].isin(namesList)]

    plotting["chat_with"] = plotting["chat_with"].astype(catNames)

    kwargs = {"alpha": 0.5}

    plt.figure(figsize=(9, 12))
    ax = sns.barplot(data=plotting, y="chat_with", hue="message_direction",
                     x="message_length", orient="h", palette="Set1", dodge=False, errwidth=0, **kwargs)
    ax.grid(True)
    ax.set_title("Average message length in significant chats")
    ax.set_ylabel('Chat')
    ax.set_xlabel('Average number of words in a message')
    ax.legend().set_title("Message direction")

    if save_dir == None:
        plt.show()
    else:
        assertDir(save_dir)
        fullPath = os.path.join(save_dir, plotName+".png")
        ax.figure.savefig(fullPath, bbox_inches='tight')
