import pandas as pd
import numpy as np
import io
import requests
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# selecting a device and using it for every storage is essential within pytorch
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# You may want to turn this one on if you don't have a mac with M1

# receiving travel data, formating date & grouping over the boroughs
url_lnd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/02_Preprocessing/London_cleaned_unpivoted.csv"
download_lnd = requests.get(url_lnd).content
df_lnd = pd.read_csv(io.StringIO(download_lnd.decode('utf-8')))
df_lnd["Date"] = pd.to_datetime(df_lnd["Date"])
df_lnd_grouped = df_lnd.groupby('Date').mean()

# receiving covid data, formating date & computing the rolling sum over last 14 days
url_cvd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/02_Preprocessing/Covid_cleaned.csv"
download_cvd = requests.get(url_cvd).content
df_cvd = pd.read_csv(io.StringIO(download_cvd.decode('utf-8')))
df_cvd["Date"] = pd.to_datetime(df_cvd["Date"])
df_cvd["newDeaths28DaysByDeathDate"] = df_cvd["newDeaths28DaysByDeathDate"].rolling(min_periods=1, window=14).sum()


# utilizing pytorch's conventions & initializing the observed log relationship as model
class LogCorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    def forward(self, x):
        return self.a + self.b * np.log(x)


def train_model(model, lr, x_train, y_train):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # i recon here is space for improvement. MSE is the most classic and thus basic loss function
    loss_fn = nn.MSELoss(reduction='mean')
    n_epochs = 1000
    # training the model
    for epoch in range(n_epochs):
        model.train()
        yhat = model(x_train)
        loss = loss_fn(y_train, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # printing the to computed paramters a and b of trained model
    print(model.state_dict())


def plot_model(model, category, x_data, y_data, log):
    # log can either be one or zero
    fig, ax = plt.subplots()
    with torch.no_grad():
        predictions = model(df_cvd["newDeaths28DaysByDeathDate"].values)
    if log == 1:
        x_data = np.log(x_data)
        plt.xlabel("Log of Covid-Deaths (rolling sum over 14 days)")
    elif log == 0:
        plt.xlabel("Covid-Deaths (rolling sum over 14 days)")
    else:
        raise Exception("Log input has to be eiteher 0 or 1")
    ax.scatter(x_data, y_data,
               marker=".",
               color="black",
               label="Real Data")
    ax.plot(x_data, predictions,
            color="red",
            label="Model Predictions")
    ax.grid(True)
    plt.xlim(xmin=0)
    plt.ylabel("Amount Traveled (%)")
    plt.legend(loc='best')
    plt.title(str(category))


# training Data is 80% of the data, x data stays the same since we are only using deaths as an independet variable
x_train = torch.tensor(df_cvd["newDeaths28DaysByDeathDate"][:int(len(df_cvd["newDeaths28DaysByDeathDate"]) * 0.8)].values).float().to(device)
x_test = torch.tensor(df_cvd["newDeaths28DaysByDeathDate"][int(len(df_cvd["newDeaths28DaysByDeathDate"]) * 0.8):].values).float().to(device)

# the 6 categorys of the travel data set & storing all y data in dictionarys
category_list = ["retail_recreation", "grocery_pharmacy", "parks", "transit", "workplaces", "residential"]
dic_y_train = {}
dic_y_test = {}
for category in category_list:
    temp_y_train = torch.tensor(df_lnd_grouped[category][:int(len(df_lnd_grouped[category]) * 0.8)].values).float().to(device)
    temp_y_test = torch.tensor(df_lnd_grouped[category][int(len(df_lnd_grouped[category]) * 0.8):].values).float().to(device)
    dic_y_train[category] = temp_y_train
    dic_y_test[category] = temp_y_test

# RETAIL RECREATION
# initializing the model
model_retail_recreation = LogCorModel().to(device)
# atm we are using a constant LR, I think one improvement here would be to use a dynamic one
lr = 3e-2
train_model(model_retail_recreation, lr, x_train, dic_y_train["retail_recreation"])
plot_model(model_retail_recreation, "retail_recreation", df_cvd["newDeaths28DaysByDeathDate"], df_lnd_grouped["retail_recreation"], 1)
plot_model(model_retail_recreation, "retail_recreation", df_cvd["newDeaths28DaysByDeathDate"], df_lnd_grouped["retail_recreation"], 0)
# plt.show()

# GROCERY PHARMACY
# initializing the model
model_grocery_pharmacy = LogCorModel().to(device)
# atm we are using a constant LR, I think one improvement here would be to use a dynamic one
lr = 3e-2
train_model(model_grocery_pharmacy, lr, x_train, dic_y_train["grocery_pharmacy"])
plot_model(model_grocery_pharmacy, "grocery_pharmacy", df_cvd["newDeaths28DaysByDeathDate"], df_lnd_grouped["grocery_pharmacy"], 0)
# plt.show()

# PARKS
# initializing the model
model_parks = LogCorModel().to(device)
# atm we are using a constant LR, I think one improvement here would be to use a dynamic one
lr = 3e-2
train_model(model_parks, lr, x_train, dic_y_train["parks"])
plot_model(model_parks, "parks", df_cvd["newDeaths28DaysByDeathDate"], df_lnd_grouped["parks"], 0)
# plt.show()

# TRANSIT
# initializing the model
model_transit = LogCorModel().to(device)
# atm we are using a constant LR, I think one improvement here would be to use a dynamic one
lr = 3e-2
train_model(model_transit, lr, x_train, dic_y_train["transit"])
plot_model(model_transit, "transit", df_cvd["newDeaths28DaysByDeathDate"], df_lnd_grouped["transit"], 0)
# plt.show()

# WORKPLACES
# initializing the model
model_workplaces = LogCorModel().to(device)
# atm we are using a constant LR, I think one improvement here would be to use a dynamic one
lr = 3e-2
train_model(model_workplaces, lr, x_train, dic_y_train["workplaces"])
plot_model(model_workplaces, "workplaces", df_cvd["newDeaths28DaysByDeathDate"], df_lnd_grouped["workplaces"], 0)
# plt.show()

# RESIDENTIAL
# initializing the model
model_residential = LogCorModel().to(device)
# atm we are using a constant LR, I think one improvement here would be to use a dynamic one
lr = 3e-2
train_model(model_residential, lr, x_train, dic_y_train["residential"])
plot_model(model_residential, "residential", df_cvd["newDeaths28DaysByDeathDate"], df_lnd_grouped["residential"], 0)
plt.show()
