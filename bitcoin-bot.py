import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
!pip install jovian --upgrade --quiet

import jovian
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm
import os
import torch
import pandas as pd
from torch.utils.data import TensorDataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms

project = "bitcoin-bot"

DATASET_URL = "/kaggle/input/btc-2012-to-2020/bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv"

data = pd.read_csv(DATASET_URL)

data.head()

df= data.dropna()
df

df['High'].plot(figsize=(12,8))

X = df[['Timestamp', 'Open', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']].values
y = df[['High']]


X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2)


from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")

# from statsmodels.tsa.stattools import adfuller

# def ad_test(df):
#      df = adfuller(df, autolag = 'AIC')
#      print("1. ADF : ",df[0])
#      print("2. P-Value : ", df[1])
#      print("3. Num Of Lags : ", df[2])
#      print("4. Num Of Observations Used For ADF Regression:", df[3])
#      print("5. Critical Values :")
#      for key, val in df[4].items():
#          print("\t",key, ": ", val)

# ad_test(df['High'])

fit = auto_arima(y_train, trace= True, supress_warnings= True)
fit.summary()




jovian.commit(project=project)
