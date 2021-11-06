import pandas as pd
import numpy as np
import io
import requests
import matplotlib.pyplot as plt

url_lnd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/data/London_cleaned_unpivoted.csv"
download_lnd = requests.get(url_lnd).content
df_lnd = pd.read_csv(io.StringIO(download_lnd.decode('utf-8')))

url_cvd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/data/London_Covid_301021.csv"
download_cvd = requests.get(url_cvd).content
df_cvd = pd.read_csv(io.StringIO(download_cvd.decode('utf-8')))
df_cvd["Date"] = df_cvd["Date"].str.replace("-", "")
df_cvd = df_cvd.rolling(min_periods=1, window=14).sum()
df_cvd = df_cvd.sort_values(by="Date").iloc[7:620, :]

df_lnd_grouped = df_lnd.groupby('DatumID').mean()
cluster_list = ["retail_recreation", "grocery_pharmacy_recreation", "parks", "transit", "workplaces", "residential"]

i = 0
for cluster in cluster_list:
    plt.figure(figsize=(16, 5))
    plt.style.use("ggplot")
    plt.scatter(np.log(df_cvd["newDeaths28DaysByDeathDate"]), df_lnd_grouped[cluster_list[i]],  # df_list.index(df)
                marker="o",
                color="black",
                label="Time Span")
    plt.xlabel("Log of deaths of past 14 days")
    plt.xlim(xmin=0)
    plt.ylabel("Amount Traveled")
    plt.title(cluster_list[i])
    i += 1
plt.show()


print('placeholder')
