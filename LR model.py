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
#print(train.shape)
#print(list(train.columns))
#train.head()
#train.describe ()
#train.columns
#train.dtypes

train ['Lead']. replace ('Female', 1 , inplace = True )
train ['Lead']. replace ('Male', 0 , inplace = True )
#train.info ()

##Split the model

X = train.loc[:,'Number words female':'Age Co-Lead']
#X = train.drop(["Lead","Year"],axis =1)
X_reduced = X.iloc[:,[0,1,2,3,4,5,10,11]]
y = train['Lead']

##Split the training set into train set and test set
X_train,X_test,y_train,y_test = train_test_split (X , y , test_size =0.3 , random_state =0 )
#test size : train size = 3:7
#X_train,X_test,y_train,y_test = train_test_split (X , y , test_size =0.2 , random_state =0 )
#test size : train size = 2:8

models = []
## Basic Model
model_LR_base = LogisticRegression ( max_iter = 10000,random_state =0 )
model_LR_base.fit(X_train , y_train)

predictTrain_LR_base = model_LR_base.predict(X_train)
AccTrain_LR_base = np.mean( predictTrain_LR_base == y_train )
print ("The accuracy of split training set : {}".format(AccTrain_LR_base))
predictTest_LR_base = model_LR_base.predict ( X_test )
AccTest_LR_base = np.mean( predictTest_LR_base == y_test )
print ("The accuracy of split test set: {}".format(AccTest_LR_base))

## Parameter Tuning

grid_LR ={'solver':['lbfgs'],'penalty': ['l2'],'C' : [100 , 10 , 1.0 , 0.1 , 0.01],}
grid_search_LR = GridSearchCV (estimator = model_LR_base ,param_grid = grid_LR ,n_jobs =-1 ,cv=5 ,error_score =0)
grid_result_LR = grid_search_LR . fit ( X_train , y_train )
print (" Best score:{}: using {}". format ( grid_result_LR.best_score_,grid_result_LR . best_params_ ) )

## Applicate the preparameter we got
model_LR = LogisticRegression ( penalty = grid_result_LR.best_params_ ['penalty'],
C= grid_result_LR.best_params_ ['C'],
max_iter =10000 ,
random_state = 42 ,
class_weight = {0:0.45,1:0.55}
)
model_LR . fit ( X_train , y_train )
predictTrain_LR = model_LR . predict ( X_train )

AccTrain_LR = np.mean ( predictTrain_LR == y_train )
print ("The accuracy of split training set : {}". format ( AccTrain_LR ) )
predictTest_LR = model_LR . predict ( X_test )
AccTest_LR = np.mean ( predictTest_LR == y_test )
print ("The accuracy of split test set: {}". format ( AccTest_LR ))

y_pred = predictTest_LR
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"accuracy:{acc}")
print(f"precision:{pre}")
print(f"recall:{rec}")
print(f"F1-score:{f1}")

