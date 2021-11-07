import pandas as pd
import numpy as np
import io
import requests
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchviz import make_dot

url_lnd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/data/London_cleaned_unpivoted.csv"
download_lnd = requests.get(url_lnd).content
df_lnd = pd.read_csv(io.StringIO(download_lnd.decode('utf-8')))
df_lnd_grouped = df_lnd.groupby('DatumID').mean().iloc[23:620, :]

url_cvd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/data/London_Covid_301021.csv"
download_cvd = requests.get(url_cvd).content
df_cvd = pd.read_csv(io.StringIO(download_cvd.decode('utf-8')))
df_cvd["Date"] = df_cvd["Date"].str.replace("-", "")
df_cvd = df_cvd.sort_values(by="Date").iloc[30:620, :]
df_cvd["newDeaths28DaysByDeathDate"] = df_cvd["newDeaths28DaysByDeathDate"].rolling(min_periods=1, window=14).sum()

# cluster_list = ["retail_recreation", "grocery_pharmacy_recreation", "parks", "transit", "workplaces", "residential"]

# i = 0
# for cluster in cluster_list:
#     plt.figure(figsize=(16, 5))
#     plt.style.use("ggplot")
#     plt.scatter(np.log(df_cvd["newDeaths28DaysByDeathDate"]), df_lnd_grouped[cluster_list[i]],  # df_list.index(df)
#                 marker="o",
#                 color="black",
#                 label="Time Span")
#     plt.xlabel("Log of deaths of past 14 days")
#     plt.xlim(xmin=0)
#     plt.ylabel("Amount Traveled")
#     plt.title(cluster_list[i])
#     i += 1
# plt.show()

device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# You may want to turn this one on if you don't have a mac wiht M1

x_train = torch.tensor(df_cvd["newDeaths28DaysByDeathDate"][:int(len(df_cvd["newDeaths28DaysByDeathDate"]) * 0.8)].values).float().to(device)
x_test = torch.tensor(df_cvd["newDeaths28DaysByDeathDate"][int(len(df_cvd["newDeaths28DaysByDeathDate"]) * 0.8):].values).float().to(device)

y_train = torch.tensor(df_lnd_grouped["retail_recreation"][:int(len(df_lnd_grouped["retail_recreation"]) * 0.8)].values).float().to(device)
y_test = torch.tensor(df_lnd_grouped["retail_recreation"][int(len(df_lnd_grouped["retail_recreation"]) * 0.8):].values).float().to(device)


class ModelClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.a + self.b * np.log(x)


model = ModelClass().to(device)
lr = 3e-2
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss(reduction='mean')
n_epochs = 1000

for epoch in range(n_epochs):
    model.train()
    yhat = model(x_train)
    loss = loss_fn(y_train, yhat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(model.state_dict())

# a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
# b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
# n_epochs = 1000
# lr = 3e-2

# for epoch in range(n_epochs):
#     yhat = a + b * x_train
#     error = y_train - yhat
#     loss = (error ** 2).mean()
#     loss.backward()
#     with torch.no_grad():
#         a -= lr * a.grad
#         b -= lr * b.grad
#     a.grad.zero_()
#     b.grad.zero_()

from sklearn.linear_model import LinearRegression
linr = LinearRegression()
linr.fit(np.log(df_cvd["newDeaths28DaysByDeathDate"])[:int(len(df_cvd["newDeaths28DaysByDeathDate"]) * 0.8)].to_numpy().reshape(-1, 1), df_lnd_grouped["retail_recreation"][:int(len(df_lnd_grouped["retail_recreation"]) * 0.8)].to_numpy().reshape(-1, 1),)
print(linr.intercept_, linr.coef_[0])
make_dot(loss, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")

plt.figure(figsize=(16, 5))
plt.style.use("ggplot")
ax = plt.axes()
ax.scatter(df_cvd["newDeaths28DaysByDeathDate"], df_lnd_grouped["retail_recreation"],  # df_list.index(df)
            marker="o",
            color="black")
ax.plot(df_cvd["newDeaths28DaysByDeathDate"], linr.predict(np.log(df_cvd["newDeaths28DaysByDeathDate"]).to_numpy().reshape(-1, 1)),
        marker=".",
        color="red",
        label="Time Span")
# ax.plot(df_cvd["newDeaths28DaysByDeathDate"], model(torch.tensor(df_cvd["newDeaths28DaysByDeathDate"].values).detach().numpy()),
# ax.plot(df_cvd["newDeaths28DaysByDeathDate"], model(df_cvd["newDeaths28DaysByDeathDate"].values),
#         marker=".",
#         color="green",
#         label="Time Span")
plt.xlabel("Log of deaths of past 14 days")
plt.xlim(xmin=0)
plt.ylabel("Amount Traveled")
plt.title("retail_recreation")
plt.show()

print(model.parameters())