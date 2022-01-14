# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 22:04:10 2022

@author: laura
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from openpyxl import load_workbook

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.inspection import plot_partial_dependence, partial_dependence

import sys
import glob
import os
plt.style.use('ggplot')


## Set path and list files

export_path= os.path.abspath('C:\\Users\\laura\\OneDrive\\Documents\\Ingredion II\\Data\\')
comparisson_file = 'ingredion_can_2016_2021_final.csv'
newcogs_file = 'ingredion_can_2016_2021_chart_newcogs.csv'
p3sales_file = 'ingredion_can_2016_2021_chart_3psales.csv'
netrev_file = 'ingredion_can_2016_2021_chart_netrev.csv'
operating_file = 'ingredion_can_2016_2021_chart_operating.csv'


#Change column names
new_columns = ['date','company_name', 'company_code_id','commercial_name','BPC_customar','BPC_product', 'ship_to_party', 'material',
               'sales_qty_total_mt','3p_sales_qty_total_mt', 'gross_revenue_usd', 'discounts_usd',
               'new_net_revenue', 'n3p_net_revenue', 'net_corn', 'raw_material_other','utilities', 'waste', 'repair',
               'labor', 'ohmfg','supplies_and_packaging','supplies_indirect',   'depreciation','3p_freight_usd',
               'logistics', 'cos_other', 'new_cogs', 'freight_usd','intercompany_cost_elimination', 'gross_profit',
               'sga_toal', 'other_income_expense',   'operating_income', 'other_non_operating_income_loss','special_items',
               'interco_dividends', 'charge_back', 'exchange_gain_loss',  'intercompany_financing_cost', 'financing_costs',
               'fees_and_royalties','pbt', 'taxes_on_income', 'net_income', 'minority_income','adj_minority_income',
               'total_net_income', 'ing10000_ingr_net_income']

#Import and format Actuals
df_actuals = pd.read_csv('C:\\Users\\laura\\OneDrive\\Documents\\Ingredion II\\Data\\CAN\\ingredion_can_2016_2021_by_customer_and_product.csv')
df_actuals.insert(loc = 0,
           column = 'date',
           value = pd.to_datetime(df_actuals.year.astype(str) + '/' + df_actuals.month.astype(str) ).dt.to_period('M'))
df_actuals.insert(loc = 0,
          column = 'Type',
          value = 'Actuals')
df_actuals=df_actuals.drop(columns=['year', 'month','fiscal_year_period'])
df_actuals = df_actuals.groupby(['Type','date', 'ship_to_party','material']).sum().reset_index()


#Import and format Forcast
df_le0 = pd.read_csv('C:\\Users\\laura\\OneDrive\\Documents\\Ingredion II\\Data\\US\\ingredion_us_2016_2021_by_customer_and_product_flatLE0.csv')
df_le0=df_le0.drop(columns=['Unnamed: 0','year', 'month'])
df_le0.columns = new_columns[0:]
df_le0.insert(loc = 0,
          column = 'Type',
          value = 'LE0')
df_le0 = df_le0.groupby(['Type','date', 'ship_to_party','material']).sum().reset_index()
df_le0['date'] = pd.to_datetime(df_le0['date']).dt.to_period('M')

df_le3 = pd.read_csv('C:\\Users\\laura\\OneDrive\\Documents\\Ingredion II\\Data\\US\\ingredion_us_2016_2021_by_customer_and_product_flatLE0.csv')
df_le3=df_le3.drop(columns=['Unnamed: 0','year', 'month'])
df_le3.columns = new_columns[0:]
df_le3.insert(loc = 0,
          column = 'Type',
          value = 'LE3')
df_le3 = df_le3.groupby(['Type','date', 'ship_to_party','material']).sum().reset_index()
df_le3['date'] = pd.to_datetime(df_le3['date']).dt.to_period('M')


#Create Dataframe joined
df_general= pd.concat ([df_actuals, df_le0,df_le3] )
df_general= df_general[['Type','date', 'ship_to_party','material','3p_sales_qty_total_mt','n3p_net_revenue','new_cogs', 'operating_income']]


#Import and format models ML
df_xgboost = pd.read_csv('C:\\Users\\laura\\OneDrive\\Documents\\Ingredion II\\Jupyter Notebook\\Models CAN\\ingredion_results_xgboost_CAN.csv')
df_lasso = pd.read_csv('C:\\Users\\laura\\OneDrive\\Documents\\Ingredion II\\Jupyter Notebook\\Models CAN\\ingredion_results_lasso_CAN.csv')
df_gbm = pd.read_csv('C:\\Users\\laura\\OneDrive\\Documents\\Ingredion II\\Jupyter Notebook\\Models CAN\\ingredion_results_gbm_CAN.csv')

df_ml= pd.concat ([df_xgboost, df_lasso,df_gbm] )
df_ml.insert(loc = 0,
          column = 'Type',
          value = df_ml.model_type)
df_ml=df_ml.drop(columns=['Unnamed: 0', 'model_type'])

#Create forecast
#Dataset
df_ml_forecast=df_ml[(df_ml['date'] == '2019-12-01') | (df_ml['date'] == '2020-12-01')]
df_ml_forecast_c1 = df_ml_forecast.pivot_table(df_ml_forecast , index='feature_forecasted', columns=['Type','date', 'ship_to_party','material'] , aggfunc=np.sum, fill_value=0)
df_ml_forecast = df_ml_forecast_c1.T
df_ml_forecast2 = df_ml_forecast.reset_index() 
# month
df_ml_forecast2.insert(loc = 2,
          column = 'month',
          value = df_ml_forecast2.level_0.str[9:].astype(int))
# year
df_ml_forecast2.insert(loc = 3,
          column = 'year',
          value = df_ml_forecast2.date.str[:4].astype(int)+1)
df_ml_forecast2=df_ml_forecast2.drop(columns=['date','level_0'])
# date
df_ml_forecast2.insert(loc = 1,
               column = 'date',
               value = '')
df_ml_forecast2['date'] = pd.to_datetime(df_ml_forecast2['year'].astype(int).astype(str) +'-' +df_ml_forecast2['month'].astype(int).astype(str)).dt.to_period('M')
df_ml_forecast2=df_ml_forecast2.drop(columns=['month','year'])


#Create FULL df
df_general= pd.concat ([df_general, df_ml_forecast2] )
df_general.to_csv((export_path + comparisson_file))


#Create METRICS CHART
df_general2 = df_general.groupby(['Type','date']).sum().reset_index()

df_newcogs= df_general2 .drop(columns=['3p_sales_qty_total_mt','n3p_net_revenue', 'operating_income'])
df_newcogs_c1 = df_newcogs.pivot_table(df_newcogs , index='Type', columns=['date'] , aggfunc=np.sum, fill_value=0)
df_newcogs = df_newcogs_c1.T
df_newcogs = df_newcogs.reset_index() 

df_3psales= df_general2 .drop(columns=['n3p_net_revenue','new_cogs', 'operating_income'])
df_3psales_c1 = df_3psales.pivot_table(df_3psales , index='Type', columns=['date'] , aggfunc=np.sum, fill_value=0)
df_3psales = df_3psales_c1.T
df_3psales = df_3psales.reset_index() 

df_netrev= df_general2 .drop(columns=['3p_sales_qty_total_mt','new_cogs', 'operating_income'])
df_netrev_c1 = df_netrev.pivot_table(df_netrev , index='Type', columns=['date'] , aggfunc=np.sum, fill_value=0)
df_netrev = df_netrev_c1.T
df_netrev = df_netrev.reset_index() 

df_operating= df_general2 .drop(columns=['3p_sales_qty_total_mt','n3p_net_revenue','new_cogs'])
df_operating_c1 = df_operating.pivot_table(df_operating , index='Type', columns=['date'] , aggfunc=np.sum, fill_value=0)
df_operating = df_operating_c1.T
df_operating = df_operating.reset_index() 

#Create FULL df
df_newcogs.to_csv((export_path + newcogs_file))
df_3psales.to_csv((export_path + p3sales_file))
df_netrev.to_csv((export_path + netrev_file))
df_operating.to_csv((export_path + operating_file))