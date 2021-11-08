import pandas as pd
import numpy as np
import io
import requests
import matplotlib.pyplot as plt

url_lnd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/02_Preprocessing/London_cleaned_unpivoted.csv"
download_lnd = requests.get(url_lnd).content
df_lnd = pd.read_csv(io.StringIO(download_lnd.decode('utf-8')))
df_lnd_grouped = df_lnd.groupby('DatumID').mean()  # .iloc[23:620, :]

url_cvd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/02_Preprocessing/Covid_cleaned.csv"
download_cvd = requests.get(url_cvd).content
df_cvd = pd.read_csv(io.StringIO(download_cvd.decode('utf-8')))
# df_cvd["Date"] = df_cvd["Date"].str.replace("-", "")
# df_cvd = df_cvd.sort_values(by="Date").iloc[30:620, :]
df_cvd["newDeaths28DaysByDeathDate"] = df_cvd["newDeaths28DaysByDeathDate"].rolling(min_periods=1, window=14).sum()

category_list = ["retail_recreation", "grocery_pharmacy_recreation", "parks", "transit", "workplaces", "residential"]

i = 0
for category in category_list:
    plt.figure(figsize=(16, 5))
    plt.style.use("ggplot")
    plt.scatter(np.log(df_cvd["newDeaths28DaysByDeathDate"]), df_lnd_grouped[category_list[i]],  # df_list.index(df)
                marker="o",
                color="black",
                label="Time Span")
    plt.xlabel("Log of deaths of past 14 days")
    plt.xlim(xmin=0)
    plt.ylabel("Amount Traveled")
    plt.title(category_list[i])
    i += 1
plt.show()
