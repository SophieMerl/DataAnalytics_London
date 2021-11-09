import pandas as pd
import io
import requests
import matplotlib.pyplot as plt

url_lnd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/02_Preprocessing/London_cleaned_unpivoted.csv"
download_lnd = requests.get(url_lnd).content
df_ldn = pd.read_csv(io.StringIO(download_lnd.decode('utf-8')))

url_brh = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/02_Preprocessing/boroughs.csv"
download_brh = requests.get(url_brh).content
df_brh = pd.read_csv(io.StringIO(download_brh.decode('utf-8')))

category_list = ["retail_recreation", "grocery_pharmacy", "parks", "transit", "workplaces", "residential"]
df_list = []

for brh_id in df_brh["id"]:
    temp_df = df_ldn.loc[df_ldn["BoroughID"] == brh_id]
    df_list.append(temp_df)

for category in category_list:
    plt.figure(figsize=(16, 5))
    plt.style.use("ggplot")
    for brh_id in df_brh["id"]:
        plt.plot(df_list[brh_id - 1]["Date"], df_list[brh_id - 1][category],
                 color="black",
                 label=str(df_brh.loc[df_brh["id"] == brh_id]["name"]))
    plt.xlabel("Date")
    plt.ylabel("Amount Traveled (Indexed)")
    plt.title(str(category))
plt.show()
