import pandas as pd
import numpy as np
import io
import requests
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# selecting a device and using it for every storage is essential within pytorch
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# You may want to turn this one on if you don't have a mac with M1

# receiving travel data, formating date, grouping over the boroughs & computing log
url_lnd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/02_Preprocessing/London_cleaned_unpivoted.csv"
download_lnd = requests.get(url_lnd).content
df_lnd = pd.read_csv(io.StringIO(download_lnd.decode('utf-8')))
df_lnd["Date"] = pd.to_datetime(df_lnd["Date"])
df_lnd_grouped = df_lnd.groupby('Date').mean()
df_lnd_grouped_log = np.log(df_lnd_grouped + 100)

# receiving covid data, formating date & computing the rolling sum over last 14 days & log
url_cvd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/02_Preprocessing/Covid_cleaned.csv"
download_cvd = requests.get(url_cvd).content
df_cvd = pd.read_csv(io.StringIO(download_cvd.decode('utf-8')))
df_cvd["Date"] = pd.to_datetime(df_cvd["Date"])
df_cvd["newDeaths28DaysByDeathDate"] = df_cvd["newDeaths28DaysByDeathDate"].rolling(min_periods=1, window=14).sum()
df_cvd_log = copy.deepcopy(df_cvd)
df_cvd_log["newDeaths28DaysByDeathDate"] = np.log(df_cvd_log["newDeaths28DaysByDeathDate"])


# utilizing pytorch's conventions & initializing the observed log relationship as model
class LinRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.a + self.b * x


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


def plot_model(model, category):
    # we are plotting in log log space
    fig, ax = plt.subplots()
    with torch.no_grad():
        predictions = model(df_cvd_log["newDeaths28DaysByDeathDate"].values)
    ax.scatter(x_train, dic_y_train[category],
               marker=".",
               color="black",
               label="Real Data (Train)")
    ax.scatter(x_test, dic_y_test[category],
               marker=".",
               color="green",
               label="Real Data (Test)")
    ax.plot(df_cvd_log["newDeaths28DaysByDeathDate"], predictions,
            color="red",
            label="Model Predictions")
    ax.grid(True)
    plt.xlim(xmin=0)
    plt.xlabel("Log of Covid-Deaths (rolling sum over 14 days)")
    plt.ylabel("Log of Amount Traveled (%)")
    plt.legend(loc='best')
    plt.title(str(category))


# training Data is 80% of the data, x data stays the same since we are only using deaths as an independet variable
x_train = torch.tensor(df_cvd_log["newDeaths28DaysByDeathDate"][:int(len(df_cvd_log["newDeaths28DaysByDeathDate"]) * 0.8)].values).float().to(device)
x_test = torch.tensor(df_cvd_log["newDeaths28DaysByDeathDate"][int(len(df_cvd_log["newDeaths28DaysByDeathDate"]) * 0.8):].values).float().to(device)

# the 6 categorys of the travel data set & storing all y data in dictionarys
category_list = ["retail_recreation", "grocery_pharmacy", "parks", "transit", "workplaces", "residential"]
dic_y_train = {}
dic_y_test = {}
for category in category_list:
    temp_y_train = torch.tensor(df_lnd_grouped_log[category][:int(len(df_lnd_grouped_log[category]) * 0.8)].values).float().to(device)
    temp_y_test = torch.tensor(df_lnd_grouped_log[category][int(len(df_lnd_grouped_log[category]) * 0.8):].values).float().to(device)
    dic_y_train[category] = temp_y_train
    dic_y_test[category] = temp_y_test

# RETAIL RECREATION
# initializing the model
model_retail_recreation = LinRegModel().to(device)
# atm we are using a constant LR, I think one improvement here would be to use a dynamic one
lr = 3e-2
train_model(model_retail_recreation, lr, x_train, dic_y_train["retail_recreation"])
plot_model(model_retail_recreation, "retail_recreation")
# plt.show()

# GROCERY PHARMACY
# initializing the model
model_grocery_pharmacy = LinRegModel().to(device)
# atm we are using a constant LR, I think one improvement here would be to use a dynamic one
lr = 3e-2
train_model(model_grocery_pharmacy, lr, x_train, dic_y_train["grocery_pharmacy"])
plot_model(model_grocery_pharmacy, "grocery_pharmacy")
# plt.show()

# PARKS
# initializing the model
model_parks = LinRegModel().to(device)
# atm we are using a constant LR, I think one improvement here would be to use a dynamic one
lr = 3e-2
train_model(model_parks, lr, x_train, dic_y_train["parks"])
plot_model(model_parks, "parks")
# plt.show()

# TRANSIT
# initializing the model
model_transit = LinRegModel().to(device)
# atm we are using a constant LR, I think one improvement here would be to use a dynamic one
lr = 3e-2
train_model(model_transit, lr, x_train, dic_y_train["transit"])
plot_model(model_transit, "transit")
# plt.show()

# WORKPLACES
# initializing the model
model_workplaces = LinRegModel().to(device)
# atm we are using a constant LR, I think one improvement here would be to use a dynamic one
lr = 3e-2
train_model(model_workplaces, lr, x_train, dic_y_train["workplaces"])
plot_model(model_workplaces, "workplaces")
# plt.show()

# RESIDENTIAL
# initializing the model
model_residential = LinRegModel().to(device)
# atm we are using a constant LR, I think one improvement here would be to use a dynamic one
lr = 3e-2
train_model(model_residential, lr, x_train, dic_y_train["residential"])
plot_model(model_residential, "residential")
# plt.show()

data = []
for category in category_list:
    temp_list = []
    temp_list.append(category)
    temp_list.append(np.corrcoef(x_train, dic_y_train[category])[0, 1])
    temp_list.append(np.corrcoef(x_test, dic_y_test[category])[0, 1])
    data.append(temp_list)
R_df = pd.DataFrame(data, columns=["Category", "R^2 Train", "R^2 Test"])

plt.show()
