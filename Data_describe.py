# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:13:15 2022

@author: 17844
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

train = pd.read_csv('train.csv',header=0)
train = train.dropna()


train ['Lead']. replace ('Female', 1 , inplace = True )
train ['Lead']. replace ('Male', 0 , inplace = True )
#train.info ()
#print(train.shape)
#print(list(train.columns))
#train.head()
#print(train.describe ())
#train.columns
#train.dtypes
#train["Lead"].value_counts()


#sns.countplot(x='Lead',data=train)
#plt.show()
#plt.savefig('Count_plot')
