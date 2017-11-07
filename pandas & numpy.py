import pandas as pd
import numpy as np


#==============================================================================
# Object Creation
#==============================================================================

# Creating a Series by passing a list of values, letting pandas create a default integer index
s = pd.Series([1,3,5,np.nan,6,8])
s

# Creating a DataFrame by passing a numpy array, with a datetime index and labeled columns.
dates = pd.date_range('20130101',periods=6)
dates
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
df

# Creating a DataFrame by passing a dict of objects that can be converted to series-like.
df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })
df2
df2.dtypes


#==============================================================================
# Viewing Data
#==============================================================================
df.head()
df.tail(3)
df.shape

df.index
df.columns
df.values
df.describe()
df.T
df.sort_index(axis=1, ascending=False)
df.sort(columns='B')


#==============================================================================
# Selection
#==============================================================================
# Getting
df['A']
df[0:3]
df['20130102':'20130104']

# Selection by Label
df.loc[dates[0]]
df.loc[:,['A','B']]
# df['A', 'B']   # ERROR
df.loc['20130102':'20130104',['A','B']]   # Showing label slicing, both endpoints are included
df.loc['20130102',['A','B']]
 df.loc[dates[0],'A']
 df.at[dates[0],'A']

# Selection by Position
df.iloc[3]    # 4th row
df.iloc[3:5,0:2]    # both endpoints are excluded
df.iloc[[1,2,4],[0,2
df.iloc[1:3,:]
df.iloc[:,1:3]
df.iloc[1,1]
df.iat[1,1]


# NumPy
np.arange(0, 2, 0.2)
np.arange(0, 2.2, 0.2)

a = np.arange(10)**3
a[2:5]
a[:6:2] = -1000    # equivalent to a[0:6:2] = -1000; from start to position 6, exclusive, set every 2nd element to -1000
a[ : :-1]          # reversed a