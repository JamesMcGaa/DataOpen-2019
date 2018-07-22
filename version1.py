import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

flight_traffic = pd.read_csv("rip.csv", na_values=[0])
stocks = pd.read_csv("stock_prices.csv", na_values=[0])

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
            try:
                filtered[feature] = filtered.fit_transform(filtered[feature])
            except:
                print('Error encoding: '+ feature)


for airline in ["AA"]:
    target = airline 
    X = filtered[airline].drop(['stock_price'], axis=1)[['year', 'month', 'origin_airport', 'destination_airport', 'cancelled', 'diverted', 'total_delay']]
    y = filtered[airline]['stock_price']

param = {'num_leaves':31, 'num_trees':100, 'objective':'binary'}
param['metric'] = 'auc'

X_train, X_test, y_train, y_test = train_test_split(X,y)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_test, y_test)


num_round = 10
bst = lgb.train(param, lgb_train, num_round, valid_sets=[lgb_val])
