import pandas as pd
import io
import requests
import torch

url_lnd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/data/London_301021.csv"
download_lnd = requests.get(url_lnd).content
df_lnd = pd.read_csv(io.StringIO(download_lnd.decode('utf-8')))
print(df_lnd.head())

url_cvd = "https://raw.githubusercontent.com/SophieMerl/DataAnaytics_London/master/data/London_Covid_301021.csv"
download_cvdurl_cvd = requests.get(url_cvd).content
df_cvdurl_cvd = pd.read_csv(io.StringIO(download_cvdurl_cvd.decode('utf-8')))
print(df_cvdurl_cvd.head())


x = torch.rand(5, 3)
print(x)


