import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pandas as pd
import pylab as pl
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

feature_names = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity',
                 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
                 'Impulsive', 'SS']
# feature_names = ['Ns  core', 'Escore', 'Oscore', 'Ascore', 'Cscore',
#                  'Impulsive', 'SS']

origin = pd.read_excel('../drug_consumption.xls')
for f in feature_names:
    print(f'{f}: {len(origin[f].unique())} {origin[f].unique()}')

features = origin[feature_names].copy()

target_name = 'Nicotine'
target = origin[[target_name]].copy()

pca = PCA(n_components=2).fit(features)
pca_2d = pca.transform(features)

import xgboost as xgb

target_mapping = dict(CL0=.1, CL1=.2, CL2=.3, CL3=.4, CL4=.5, CL5=.6, CL6=.7, CL7=.8)
target[target_name] = target[target_name].map(target_mapping)

X, y = features, target

data_dmatrix = xgb.DMatrix(data=X, label=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective='reg:squarederror',
                          colsample_bytree=0.3,
                          learning_rate=0.1,
                          max_depth=10,
                          alpha=10,
                          n_estimators=10)
xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

params = {
    "objective": "reg:squarederror",
    'colsample_bytree': 0.3,
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 10
}

cv_results = xgb.cv(dtrain=data_dmatrix,
                    params=params,
                    nfold=3,
                    num_boost_round=50,
                    early_stopping_rounds=10,
                    metrics="rmse",
                    as_pandas=True,
                    seed=123)

print(cv_results.head())
print((cv_results["test-rmse-mean"]).tail(1))

xg_reg = xgb.train(params=params,
                   dtrain=data_dmatrix,
                   num_boost_round=10)

import matplotlib.pyplot as plt

# xgb.plot_tree(xg_reg, num_trees=0)
# # plt.rcParams['figure.figsize'] = [50, 10]
# plt.show()

xgb.plot_tree(xg_reg, num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()