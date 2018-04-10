# Package
import os
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Data
path = 'C:\\Users\\xuefei.yang\\Documents\\DS_challenge\\Salary Prediction'
os.chdir(path)

df = pd.read_csv('Train_rev1.csv')
df.head()
df.shape

# EDA
df.apply(lambda x:x.dtype)

num_col = ['SalaryNormalized']
str_col = [col for col in df.columns if col not in num_col]

df[num_col].describe().T
df[str_col].isnull().sum()
df[str_col].nunique()

plt.hist(df['SalaryNormalized'])
plt.hist(np.log(df['SalaryNormalized']))

# Missing values
df = df.loc[(df['Title'].notnull()) & df['SourceName'].notnull()]

df['Company'] = df['Company'].replace(np.NAN, 'Unknown')
df['Company'].value_counts()

pd.crosstab(df['ContractType'], df['ContractTime'])
df['ContractType'] = df['ContractType'].replace(np.NAN, 'Unknown')
df['ContractTime'] = df['ContractTime'].replace(np.NAN, 'Unknown')

df[str_col].isnull().sum()
df[str_col].nunique()


plt.hist(df['SalaryNormalized'])
plt.hist(np.log(df['SalaryNormalized']))

# Feature Engineering
## one-hot encode
cat = df[['ContractTime', 'ContractType', 'Category']]
cat = pd.get_dummies(cat)

## tf-idf
vect = TfidfVectorizer(min_df=1, ngram_range=(1,3), max_features=24000000)
des = df['FullDescription']  + ' ' + df['Company'] + ' ' + df['LocationNormalized']
des = vect.fit_transform(des)

vect2 = TfidfVectorizer(min_df=1, ngram_range=(1,3), max_features=24000000)
titles = df['Title']
titles = vect2.fit_transform(titles)

## concatenate
merged = hstack((des,titles,cat))

y = np.log(df['SalaryNormalized'])

# Model
rr = linear_model.Ridge(alpha= 0.045)
rr.fit(merged, y)
pred_rr = rr.predict(merged)
mae_rr = mean_absolute_error(df['SalaryNormalized'], np.exp(pred_rr))
mae_rr

# https://github.com/Newmu/Salary-Prediction
# https://github.com/buma/kaggle-job-salary

