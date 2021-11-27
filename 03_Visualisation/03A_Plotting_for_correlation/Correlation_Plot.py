import pandas as pd
import numpy as np
import io
import requests
import matplotlib.pyplot as plt

# receiving travel data, formating date & grouping over the boroughs
url_lnd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/02_Preprocessing/London_cleaned_unpivoted.csv"
download_lnd = requests.get(url_lnd).content
df_lnd = pd.read_csv(io.StringIO(download_lnd.decode('utf-8')))
df_lnd["Date"] = pd.to_datetime(df_lnd["Date"])
df_lnd_grouped = df_lnd.groupby('Date').mean()

# receiving covid data, formating date & computing the rolling sums over last 14 days
url_cvd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/02_Preprocessing/Covid_cleaned.csv"
download_cvd = requests.get(url_cvd).content
df_cvd = pd.read_csv(io.StringIO(download_cvd.decode('utf-8')))
df_cvd["Date"] = pd.to_datetime(df_cvd["Date"])
df_cvd["newDeaths28DaysByDeathDate"] = df_cvd["newDeaths28DaysByDeathDate"].rolling(min_periods=1, window=14).sum()
df_cvd["newCasesBySpecimenDate"] = df_cvd["newCasesBySpecimenDate"].rolling(min_periods=1, window=14).sum()

# the 6 categorys of the travel data set
category_list = ["retail_recreation", "grocery_pharmacy", "parks", "transit", "workplaces", "residential"]

# plotting: new cases as independent and travel data as dependent variable
for category in category_list:
    fig, ax = plt.subplots()
    ax.scatter(df_cvd["newCasesBySpecimenDate"], df_lnd_grouped[category_list[category_list.index(category)]],
               marker=".",
               color="black",
               label=str(category) + " in relation to new cases")
    ax.grid(True)
    plt.xlabel("New Covid-Cases (rolling sum over 14 days)")
    plt.xlim(xmin=0)
    plt.ylabel("Amount Traveled (%)")
    plt.title(category_list[category_list.index(category)])
# plt.show()

# plotting: death cases as independent and travel data as dependent variable
for category in category_list:
    fig, ax = plt.subplots()
    ax.scatter(df_cvd["newDeaths28DaysByDeathDate"], df_lnd_grouped[category_list[category_list.index(category)]],
               marker=".",
               color="black",
               label=str(category) + " in relation to deaths")
    ax.grid(True)
    plt.xlabel("Covid-Deaths (rolling sum over 14 days)")
    plt.xlim(xmin=0)
    plt.ylabel("Amount Traveled (%)")
    plt.title(category_list[category_list.index(category)])
# plt.show()

# plotting: log(death cases) as independent and travel data as dependent variable
for category in category_list:
    fig, ax = plt.subplots()
    ax.scatter(np.log(df_cvd["newDeaths28DaysByDeathDate"]), df_lnd_grouped[category_list[category_list.index(category)]],
               marker=".",
               color="black",
               label=str(category) + " in relation to log deaths")
    ax.grid(True)
    plt.xlabel("Log of Covid-Deaths (rolling sum over 14 days)")
    plt.xlim(xmin=0)
    plt.ylabel("Amount Traveled (%)")
    plt.title(category_list[category_list.index(category)])
# plt.show()

# plotting: in log space: death cases as independent and travel data as dependent variable
# note: travel data can be up to -100 (not exactly -100 though) thus we need to add 100 before computing the log
for category in category_list:
    fig, ax = plt.subplots()
    ax.scatter(np.log(df_cvd["newDeaths28DaysByDeathDate"]), np.log(df_lnd_grouped[category_list[category_list.index(category)]] + 100),
               marker=".",
               color="black",
               label=str(category) + " in relation to log deaths")
    ax.grid(True)
    plt.xlabel("Log of Covid-Deaths (rolling sum over 14 days)")
    plt.xlim(xmin=0)
    plt.ylabel("Log of Amount Traveled (%)")
    plt.title(category_list[category_list.index(category)])
plt.show()
