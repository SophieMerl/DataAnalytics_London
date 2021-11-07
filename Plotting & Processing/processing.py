import pandas as pd
import numpy as np
import io
import requests
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot


device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# You may want to turn this one on if you don't have a mac

url_lnd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/data/London_301021.csv"
download_lnd = requests.get(url_lnd).content
df_lnd = pd.read_csv(io.StringIO(download_lnd.decode('utf-8')))

url_cvd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/data/London_Covid_301021.csv"
download_cvd = requests.get(url_cvd).content
df_cvd = pd.read_csv(io.StringIO(download_cvd.decode('utf-8')))

# x_train = torch.tensor(df_lnd['retail_recreation_city'][:int(len(df_lnd['retail_recreation_city']) * 0.8)].values).float().to(device)
x_train = torch.tensor(df_lnd['retail_recreation_city'][:500].values).float().to(device)
x_test = torch.tensor(df_lnd['retail_recreation_city'][int(len(df_lnd['retail_recreation_city']) * 0.8):].values).float().to(device)

# y_train = torch.tensor(df_cvd['newCasesBySpecimenDate'][:int(len(df_cvd['newCasesBySpecimenDate']) * 0.8)].values).float().to(device)
y_train = torch.tensor(df_cvd['newCasesBySpecimenDate'][:500].rolling(min_periods=1, window=14).sum().values).float().to(device)
y_test = torch.tensor(df_cvd['newCasesBySpecimenDate'][:int(len(df_cvd['newCasesBySpecimenDate']) * 0.8)].values).float().to(device)


class ModelClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.a + self.b * np.log(x)


model = ModelClass().to(device)
lr = 1e-1
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
