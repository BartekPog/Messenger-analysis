import pandas as pd
import seaborn as sns
import statsmodels
import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from datetime import date

sns.set()

MESSAGES_FILE = "all_messages.csv"
TIMEZONE = "Europe/Warsaw"
USER = "Bartek Pogod"

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


def messagesInAChatBarPlot(data: pd.DataFrame, chats: int):
    noGroup = data[data["chat_with"] != "GROUP"]
    plotDataSeries = noGroup["chat_with"].value_counts()[:chats]

    plotData = pd.DataFrame(plotDataSeries)
    plotData["person"] = plotData.index
    plotData["messages_number"] = plotData["chat_with"]
    ax = sns.barplot(x=plotData["messages_number"],
                     y=plotData["person"], orient="h")
    ax.grid(True)
    ax.set_title("{} {}'s chats with the most messages".format(chats, USER))
    ax.set_ylabel('Chat participant')
    ax.set_xlabel('Total number of messages')
    plt.show()


messagesInAChatBarPlot(data, chats=15)


def plotActivityOverTime(data: pd.DataFrame):
    noGroup = data[data["chat_with"] != "GROUP"]

    noGroup["sent_by_user"] = noGroup["sender_name"] == USER

    byDates = noGroup.groupby(["date", "sent_by_user"], as_index=True).agg([
        'count'])

    dates = pd.date_range(min(data["date"]), max(data["date"]))

    plotting = byDates.reindex(dates, level=0).reset_index()

    plotting["messages_per_day"] = plotting[("sender_name", "count")]
    plotting["Message direction"] = plotting["sent_by_user"].apply(
        lambda x: "Sent" if x else "Received")

    plotting["date_float"] = plotting["date"].values.astype(float)

    g = sns.lmplot(data=plotting, x="date_float", y="messages_per_day",
                   hue="Message direction", scatter=False, order=5, legend_out=False, aspect=1.7)

    g.set(xlim=(plotting["date_float"].min(), plotting["date_float"].max()))

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Average messages number over time")
    g.axes[0, 0].yaxis.set_major_locator(MultipleLocator(10))

    avgSecondsInMonth = 2628288*int(1e9)
    numberOfMonths = 2

    g.axes[0, 0].xaxis.set_major_locator(
        MultipleLocator(avgSecondsInMonth*numberOfMonths))
    g.axes[0, 0].set_xlabel('Time')
    g.axes[0, 0].set_ylabel('Messages per day')

    xticks = g.axes[0, 0].get_xticks()
    xticks_dates = [datetime.datetime.fromtimestamp(
        x/int(1e9)).strftime(' %b %Y') for x in xticks]
    g.axes[0, 0].set_xticklabels(
        xticks_dates,  rotation=45, horizontalalignment='right')
    plt.show()


plotActivityOverTime(data)


def plotActivityForMostFrequentNonGroupChats(data: pd.DataFrame, chats: int, order: int):
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
    plt.show()


plotActivityForMostFrequentNonGroupChats(data, 4, 3)


def plotActivityOverWeek(data: pd.DataFrame):
    noGroup = data[data["chat_with"] != "GROUP"]

    noGroup["sent_by_user"] = noGroup["sender_name"] == USER

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

    g = sns.catplot(x="weekday", y="average_messages_per_day", hue="message_direction", data=plotting,
                    height=6, kind="bar", palette="muted")

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Average activity over week")
    g._legend.set_title("Message direction")
    g.axes[0, 0].set_xlabel('Day of the week')
    g.axes[0, 0].set_ylabel('Messages per day')
    g.axes[0, 0].set_xticklabels(
        g.axes[0, 0].get_xticklabels(), horizontalalignment='right',  rotation=45)
    plt.show()


plotActivityOverWeek(data)


def plotActivityOverDay(data: pd.DataFrame):
    noGroup = data[data["chat_with"] != "GROUP"]

    noGroup["sent_by_user"] = noGroup["sender_name"] == USER

    plotting = noGroup.groupby(
        ["hour", "sent_by_user"]).agg("count").reset_index()

    numberOfHours = len(pd.period_range(min(data["date"]), max(data["date"])))

    plotting["avg_messages_per_hour"] = plotting["sender_name"]/numberOfHours

    plotting["message_direction"] = plotting["sent_by_user"].apply(
        lambda x: 'Sent' if x else 'Received')

    plotting["hour_num"] = plotting["hour"].astype(int)

    g = sns.lmplot(data=plotting, x="hour_num", y="avg_messages_per_hour",
                   hue="message_direction", scatter=False, order=4, legend_out=False, aspect=1.7)

    g.axes[0, 0].xaxis.set_major_locator(
        MultipleLocator(2))
    g.set(xlim=(0, 23))

    g.ax.legend(loc=2)

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Average messages number over day")
    g.axes[0, 0].set_xlabel('Hour')
    g.axes[0, 0].set_ylabel('Number of messages')

    plt.show()


plotActivityOverDay(data)
