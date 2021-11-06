import pandas as pd
import numpy as np
import io
import requests
import torch


device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# You may want to turn this one on if you don't have a mac

url_lnd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/data/London_301021.csv"
download_lnd = requests.get(url_lnd).content
df_lnd = pd.read_csv(io.StringIO(download_lnd.decode('utf-8')))

url_cvd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/data/London_Covid_301021.csv"
download_cvd = requests.get(url_cvd).content
df_cvd = pd.read_csv(io.StringIO(download_cvd.decode('utf-8')))

x_train = torch.from_numpy(df_lnd[1][:len(df_lnd[]) * 0.8]).float().to(device)
x_test = torch.from_numpy().float().to(device)

y_train = torch.from_numpy().float().to(device)
y_test = torch.from_numpy().float().to(device)


lr = 1.3e-1
n_epochs = 1000

a = torch.randn(1, dtype=torch.float).to(device)
b = torch.randn(1, dtype=torch.float).to(device)
a.requires_grad_()
b.requires_grad_()

for epoch in range(n_epochs):
    yhat = a + b * x_train
    error = y_train - yhat
    loss = (error ** 2).mean()
    loss.backward()
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
    a.grad.zero_()
    b.grad.zero_()
