import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('tips.csv')


# EDA
#print(df)

#print(df.dtypes)
#print(df.describe().T)
#print(df.isnull().sum())

#plt.subplots(figsize=(15,8))
#
#for i, col in enumerate(['total_bill', 'tip']):
#    plt.subplot(2,3, i + 1)
#    sb.displot(df[col])
#plt.tight_layout()
#plt.show()

#plt.subplots(figsize=(15,8))
#for i, col in enumerate(['total_bill', 'tip']):
#    plt.subplot(2,3, i + 1)
#    sb.boxplot(df[col])
#plt.tight_layout()
#plt.show()

#print(df.shape, df[(df['total_bill']<45) & (df['tip']<7)].shape)

df =df[(df['total_bill']<45) & (df['tip']<7)]

#feat = df.loc[:, 'sex':'size'].columns
#
#plt.subplots(figsize=(15,8))
#for i, col in enumerate(feat):
#    plt.subplot(2,3, i + 1)
#    sb.countplot(df[col])
#plt.tight_layout()
#plt.show()


# Data Tranforming
le = LabelEncoder()
  
for col in df.columns:
  if df[col].dtype == object:
    df[col] = le.fit_transform(df[col])


# Data Splitting
features = df.drop('tip', axis=1)
target = df['tip']
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.15, random_state=22) 
#print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# Model Training
models = [LinearRegression(), XGBRegressor(), RandomForestRegressor(), AdaBoostRegressor()]


for i in range(4):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  pred_train = models[i].predict(X_train)
  print('Trainnig Accuracy: ', 1-(mae(Y_train, pred_train)))

  pred_val = models[i].predict(X_val)
  print('Validation Accuracy: ', 1-(mae(Y_val, pred_val)))

# Accuracy is bad due to the small dataset