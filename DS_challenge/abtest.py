# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 16:42:01 2018

@author: xuefei.yang
"""

import pandas as pd

import calendar


# test table
test = pd.read_csv('test_table.csv')
test.head()
test.shape
len(test['user_id'].unique())

# user table
user = pd.read_csv('user_table.csv')
user.head()
user.shape
len(user['user_id'].unique())

# merge two tables
df = pd.merge(test, user, how='left', on=['user_id'])
df.head()
df.shape
df.isnull().sum()


# EDA
df.groupby(['country'])['conversion'].mean().plot(kind='bar')



# Missing values
df['ads_channel'].fillna('Unknown', inplace=True)
df.isnull().sum()
df.shape

df.dropna(inplace=True)
df.isnull().sum()
df.shape


# New feature
df.head()
df['date'] = pd.to_datetime(df['date'])
df['wod'] = df['date'].apply(lambda x:calendar.day_name[x.weekday()])


df_test = df.loc[df['country'] != 'Spain']
df_test.shape

df['conversion'].loc[df['test']==0].mean()
df['conversion'].loc[df['test']==1].mean()


# t test
from scipy import stats
stats.ttest_ind(df_test['conversion'].loc[df_test['test']==0], df_test['conversion'].loc[df_test['test']==1], equal_var = False)

df_test['conversion'].loc[df_test['test']==0].mean()
df_test['conversion'].loc[df_test['test']==1].mean()

# logistic regression
#X = df_test.drop(['conversion', 'user_id', 'date'], axis=1)
#y = df_test['conversion']
#
#import statsmodels.api as sm
#X = pd.get_dummies(X)
#logit_model=sm.Logit(y, X)
#result=logit_model.fit()
#print(result.summary())
