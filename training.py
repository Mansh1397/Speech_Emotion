import warnings
warnings.filterwarnings('ignore')  
import joblib
import helper as hlp 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = hlp.load_data(test_size=0.25)
    
grid = [
{
'model': [KNeighborsClassifier()],   
'model__n_neighbors': np.arange(1,31,1),       
'model__algorithm':['auto', 'ball_tree','kd_tree', 'brute'],
'model__leaf_size': np.arange(1,10,1), 
'model__p': np.arange(1,10,1), 
'model__metric':['manhattan','euclidean','minkowski'],    
'model__n_jobs':[-1,1]
},
{
'model': [RandomForestClassifier()],           
'model__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],           #default=100  and 100 came to best.
'model__criterion':['gini','entropy'],               #default=mse and 
'model__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],                 #None
'model__min_samples_split':[2, 5, 7, 9, 10],              # 2
'model__min_samples_leaf':[1,2,4],               #1
'model__min_weight_fraction_leaf':[0.0],         #0.0
'model__max_features':['auto', 'sqrt', 'log2'],                  #auto
'model__max_leaf_nodes':[None,2,5,8],                  #None
'model__min_impurity_decrease':[0.0,1.0,2.0],           #0.0
'model__min_impurity_split':[None,0.01,0.1],             #None
'model__bootstrap':[True, False],                      #True
'model__oob_score':[True,False],                     #False
'model__n_jobs':[-1],                         #-1
'model__random_state':[None,0,42],  

},
{'model': [XGBClassifier()],
'model__learning_rate': np.arange(0,1,0.01),
'model__n_estimators': np.arange(10,500,20),
'model__subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
'model__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], 
'model__colsample_bytree': [0.5, 0.7, 0.9, 1],
'model__min_child_weight': [1, 2, 3, 4]
},
{'model': [lgb.LGBMClassifier()],
'model__num_leaves': [10,20,30,40,50,60,70,80,90,100,150,200],
'model__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
'model__learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
'model__n_estimators': [10, 30, 50, 70, 90, 100, 120, 150, 170, 200], 
'model__min_split_gain' : [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
'model__reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
'model__reg_lambda': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}]

pipe = Pipeline(steps=[('model', LogisticRegression())])
acc_scorer = make_scorer(accuracy_score)
rfs = RandomizedSearchCV(pipe, grid, random_state=0,scoring=acc_scorer, n_jobs=-1, n_iter=100, verbose=True, return_train_score = True, cv = 3)
rfs.fit(X_train, y_train)
joblib.dump(rfs, 'Model')
print("Model has been fitted")
