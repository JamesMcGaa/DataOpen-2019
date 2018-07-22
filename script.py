import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

flight_traffic = pd.read_csv("C:\\Users\\James\\Documents\\ML\\Citadel Summer Invitation 2018 NYC\\flight_traffic.csv", na_values=[0])
stocks = pd.read_csv("C:\\Users\\James\\Documents\\ML\\Citadel Summer Invitation 2018 NYC\\stock_prices.csv", na_values=[0])

stocks['Date'] = pd.to_datetime(stocks['timestamp'])
flight_traffic["total_delay"] = flight_traffic["airline_delay"] + flight_traffic["weather_delay"] + flight_traffic["air_system_delay"] + flight_traffic["security_delay"] + flight_traffic["aircraft_delay"] 
flight_traffic = flight_traffic.assign(Date=pd.to_datetime(flight_traffic[['year', 'month', 'day']]))
flight_traffic.set_index(pd.to_datetime(flight_traffic['Date']), inplace=True)
merged = flight_traffic.merge(stocks,on='Date',how='left')

filtered = {}
for airline in merged.airline_id.unique():
    if airline in merged.columns:
        filtered[airline] = merged[merged["airline_id"] == airline]
        filtered[airline] = filtered[airline].interpolate()
        filtered[airline]["stock_price"] =  filtered[airline][airline]

        le = LabelEncoder()
        for feature in ['origin_airport',  'destination_airport']:
            filtered[airline][feature] = le.fit_transform(filtered[airline][feature])
            y = filtered[airline]['stock_price']
        #y['stock_price'] = (y['stock_price'] - y['stock_price'].mean()) / y['stock_price'].std(); #print(X['total_delay'])





mapes = {}
for airline in ['AA']: #	'UA',	'B6',	'OO',	'AS',	'NK',	'WN',	'DL',	'HA']:
    target = airline
    X = filtered[airline].drop(["stock_price"],axis=1)[['year', 'month', 'origin_airport', 'destination_airport', 'cancelled', 'diverted', 'total_delay']]
    X['total_delay'] = (X['total_delay'] - X['total_delay'].mean()) / X['total_delay'].std(); #print(X['total_delay'])
    y = filtered[airline]['stock_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test)
    lgb_valid = lgb.Dataset(X_valid, y_valid)
    
    bst = lgb.train(core_params, lgb_train, num_round, valid_sets=[lgb_valid])
    ypred = bst.predict(X_test, num_iteration=bst.best_iteration)
    mapes[airline] = mean_absolute_percentage_error(y_test,ypred)


core_params = {
    'boosting_type': 'gbdt', # GBM type: gradient boosted decision tree, rf (random forest), dart, goss.
    'objective': 'regression', # the optimization object: binary, regression, multiclass, xentropy.
    'learning_rate': 0.01, # the gradient descent learning or shrinkage rate, controls the step size.
    'num_leaves': 50, # the number of leaves in one tree.
    'nthread': 4, # number of threads to use for LightGBM, best set to number of actual cores.
    
    'metric': 'mape' # an additional metric to calculate during validation: area under curve (auc).
}

num_round = 1000



def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 1

print(mean_absolute_percentage_error(y_test,ypred))
import graphviz
bst.save_model('model.txt')
lgb.plot_tree(bst, figsize=(20, 20))
