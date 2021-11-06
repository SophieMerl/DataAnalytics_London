import pandas as pd
import matplotlib.pyplot as plt


input = pd.read_csv("/Users/dennisblaufuss/Desktop/London_251021.csv")

hood_list = [column[6:] for column in input.columns if "parks_" in column]

cluster_list = ["retail_recreation_", "grocery_pharmacy_recreation_", "parks_", "transit_", "workplaces_", "residential_"]
df_list = []

for cluster in cluster_list:
    temp_df = input[[column for column in input.columns if cluster in column or column == 'Date']]
    df_list.append(temp_df)
    # Shape of every single one is 669 X 31

i = 0
for df in df_list:
    plt.figure(figsize=(16, 5))
    plt.style.use("ggplot")
    for hood in hood_list:
        plt.plot(df["Date"], df[cluster_list[i] + hood],  # df_list.index(df)
                 marker="o",
                 color="red",
                 label="Time Span")
    plt.xlabel("Time")
    plt.xlim(xmin=0)
    plt.ylabel("Amount Traveled")
    # plt.ylim(ymin=0)
    plt.title(cluster_list[i])
    i += 1
plt.show()
