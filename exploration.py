import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from datetime import date

sns.set()

# DATA_DIR = 'inbox/'
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
    # ax.right_ax(False)
    ax.set_title("{} {}'s chats with the most messages".format(chats, USER))
    ax.set_ylabel('Chat participant')
    ax.set_xlabel('Total number of messages')
    plt.show()


messagesInAChatBarPlot(data, chats=15)


# overal activity over time
def plotActivityOverTime(data: pd.DataFrame):
    noGroup = data[data["chat_with"] != "GROUP"]

    sentByUser = noGroup[noGroup["sender_name"] == USER]
    userPlot = sentByUser.groupby(["date"]).agg(['count'])
    userPlot["messages_per_day"] = userPlot[userPlot.columns[0]]
    userPlot["date"] = userPlot.index
    userPlot["date_ordinal"] = pd.to_datetime(
        userPlot['date']).apply(lambda date: date.toordinal())

    sentToUser = noGroup[noGroup["sender_name"] != USER]
    toUserPlot = sentToUser.groupby(["date"]).agg(['count'])
    toUserPlot["messages_per_day"] = toUserPlot[toUserPlot.columns[0]]
    toUserPlot["date"] = toUserPlot.index
    toUserPlot["date_ordinal"] = pd.to_datetime(
        toUserPlot['date']).apply(lambda date: date.toordinal())

    ax = sns.regplot(data=userPlot, x="date_ordinal", y="messages_per_day",
                     order=3, scatter=False, scatter_kws={"alpha": 0.03}, label="Sent by user", color="green")
    ax = sns.regplot(data=toUserPlot, x="date_ordinal", y="messages_per_day",
                     order=3, scatter=False, scatter_kws={"alpha": 0.03}, label="Sent to user", color="red")

    # ax.set_xlim(userPlot['date_ordinal'].min() - 1, userPlot['date_ordinal'].max() + 1)
    # ax.set_ylim(0, userPlot['messages_per_day'].max() + 1)
    ax.grid(True)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel('Date')
    ax.set_ylabel('Average messages per day')
    new_labels = [date.fromordinal(int(item)) for item in ax.get_xticks()]
    ax.set_xticklabels(new_labels, horizontalalignment='right',  rotation=45)

    ax.set_title("Average activity over time")

    plt.legend(labels=['Sent by {}'.format(USER), 'Sent to {}'.format(USER)])
    plt.show()


plotActivityOverTime(data)


def plotActivityOverWeek(data: pd.DataFrame):

    noGroup = data[data["chat_with"] != "GROUP"]

    noGroup["sent_by_user"] = noGroup["sender_name"] == USER
    # noGroup[""] noGroup["sender_name"] == USER

    plotting = noGroup.groupby(["weekday", "sent_by_user"], as_index=True).agg([
        'count']).reset_index()

    # plotting["weekday"] = plotting.index[:0]

    plotting["message_direction"] = plotting["sent_by_user"].apply(
        lambda x: 'Sent' if x else 'Received')
    g = sns.catplot(x="weekday", y=('sender_name', 'count'), hue="message_direction", data=plotting,
                    height=6, kind="bar", palette="muted")

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Average activity over week")
    g.axes[0, 0].set_xlabel('Weekday')
    g.axes[0, 0].set_ylabel('Total Messages')
    g.axes[0, 0].set_xticklabels(
        g.axes[0, 0].get_xticklabels(), horizontalalignment='right',  rotation=45)
    plt.show()


plotActivityOverWeek(data)

# TODO
# Average the number of messages to be per day
# Activity over week (AVERAGE)
# Keywords per chat
# Activity over day
