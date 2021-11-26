import pandas as pd
import io
import requests
import torch
from torchviz import make_dot

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

# training Data is 80% of the data but for visualisation this can be ignored
# furthermore we don't need test data in visualisation
x_train = torch.tensor(df_cvd["newDeaths28DaysByDeathDate"][:int(len(df_cvd["newDeaths28DaysByDeathDate"]) * 0.8)].values).float().to(device)
y_train = torch.tensor(df_lnd_grouped["retail_recreation"][:int(len(df_lnd_grouped["retail_recreation"]) * 0.8)].values).float().to(device)

# creating the model manual for better visualisation
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
n_epochs = 1000
lr = 3e-2

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
make_dot(yhat).render("yhat_torchviz", format="png")
make_dot(error).render("error_torchviz", format="png")
make_dot(loss).render("loss_torchviz", format="png")
