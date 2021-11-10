import pandas as pd
import numpy as np
import io
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.pyplot import cm

# receiving travel data & formating date
url_lnd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/02_Preprocessing/London_cleaned_unpivoted.csv"
download_lnd = requests.get(url_lnd).content
df_lnd = pd.read_csv(io.StringIO(download_lnd.decode('utf-8')))
df_lnd["Date"] = pd.to_datetime(df_lnd["Date"])

# receiving look up table for boroughs
url_brh = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/02_Preprocessing/boroughs.csv"
download_brh = requests.get(url_brh).content
df_brh = pd.read_csv(io.StringIO(download_brh.decode('utf-8')))

# the 6 categorys of the travel data set
category_list = ["retail_recreation", "grocery_pharmacy", "parks", "transit", "workplaces", "residential"]

# creating a list of df for each borough
df_list = []
for brh_id in df_brh["id"]:
    temp_df = df_lnd.loc[df_lnd["BoroughID"] == brh_id]
    df_list.append(temp_df)

# plotting one plot for each category with a line for every single borough
# for detailed explanation of plt & ax functions look into file "Covid_Data_Plot"
for category in category_list:
    color = iter(cm.rainbow(np.linspace(0, 1, 33)))
    fig, ax = plt.subplots()
    for brh_id in df_brh["id"]:
        clr = next(color)
        ax.plot(df_list[brh_id - 1]["Date"], df_list[brh_id - 1][category],
                color=clr,
                label=str(df_brh.loc[df_brh["id"] == brh_id]["name"]),
                linewidth=0.4)
    fmt_half_year = mdates.MonthLocator(interval=3)
    ax.xaxis.set_major_locator(fmt_half_year)
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(fmt_month)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.format_xdata = mdates.DateFormatter('%Y-%m')
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel("Date")
    plt.ylabel("Amount Traveled (%)")
    plt.title(str(category))
plt.show()
