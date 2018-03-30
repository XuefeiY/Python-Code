# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:18:36 2018

@author: xuefei.yang
"""

# Import Package
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve


# Load data
df = pd.read_csv("conversion_data.csv")

# EDA
df.head()
df.isnull().sum()
df.describe()
df.shape

# Remove outlier
np.sort(df['age'].unique())
df.loc[df['age'] > 79]
df = df.loc[df['age'] < 100]
df.shape

# Visualization
df['converted'].mean()

df['country'].value_counts().plot(kind='bar')
df.groupby('country')['converted'].mean().plot(kind='bar')

df['source'].value_counts().plot(kind='bar')
df.groupby('source')['converted'].mean().plot(kind='bar')


df.groupby('age')['converted'].mean().plot(kind='bar')

plt.scatter(df.age, df.converted)

sns.pairplot(df, hue = 'converted')

# Data preprocess
X = df.loc[:, ['country', 'age', 'new_user', 'source', 'total_pages_visited']]
y = df['converted']

## one-hot encoding
X  = pd.get_dummies(X)

## train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42) 

y_train.mean()
y_test.mean()
y.mean()       # imbalanced data


# Classification
## Random Forest
clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42, max_features = int(np.sqrt(len(X.columns))))
clf.fit(X_train, y_train)

y_test_pred = clf.predict(X_test)
confusion_matrix(y_test_pred, y_test)

np.mean(y_test_pred == y_test)
clf.oob_score_


# Precision/Recall rate
precision, recall, _ = precision_recall_curve(y_test_pred, y_test)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.np.mean(y_test_pred == y_test))

importance = clf.feature_importances_
indices = np.argsort(importance)
importances = pd.DataFrame(importance, index = X.columns, columns=['Importance'])




### feature importance
plt.title('Feature Importances')
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), X.columns[indices])
plt.xlabel('Relative Importance')

## Logistic Regression
glm = LogisticRegression()
glm.fit(X_train, y_train)

ytest_glm = glm.predict(X_test)
np.mean(ytest_glm == y_test)
glm.coef_
coef = pd.DataFrame(np.reshape(glm.coef_, (10,)), index = X.columns, columns=['Coef'])
coef




df.loc[(df['total_pages_visited'] > 15) & (df['converted'] == 0)]



