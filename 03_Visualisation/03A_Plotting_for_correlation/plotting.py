import pandas as pd
import io
import requests
import matplotlib.pyplot as plt

url_lnd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/data/London_cleaned_unpivoted.csv"
download_lnd = requests.get(url_lnd).content
df_ldn = pd.read_csv(io.StringIO(download_lnd.decode('utf-8')))

borough_list = list(range(1, 34))

category_list = ["retail_recreation", "grocery_pharmacy", "parks", "transit", "workplaces", "residential"]
df_list = []

for borough in borough_list:
    temp_df = df_ldn.loc[df_ldn["BoroughID"] == borough]
    df_list.append(temp_df)

for category in category_list:
    plt.figure(figsize=(16, 5))
    plt.style.use("ggplot")
    for borough in borough_list:
        plt.plot(df_list[borough_list.index(borough)]["DatumID"], df_list[borough_list.index(borough)][category],
                 marker=".",
                 color="red",
                 label=str(borough))
    plt.xlabel("Date")
    plt.xlim(xmin=0)
    plt.ylabel("Amount Traveled (Indexed)")
    plt.title(str(category))
plt.show()
