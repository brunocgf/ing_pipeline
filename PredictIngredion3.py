#   Ingredion Predicting

#   Libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from openpyxl import load_workbook
import time
from datetime import datetime
from datetime import timedelta

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

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__, ContentSettings
from io import StringIO, BytesIO
import pandas as pd
import azure.functions as func
import glob
import os, uuid

import pickle
import sys
import glob
import os
plt.style.use('ggplot')

from connection_helper import engine
conn = engine.connect()





#   US Prediction

def getIngredioData():
    # Import df from sql
    sql_query = 'SELECT * FROM [dbo].[pl_data_us_actuals]'
    df = pd.read_sql(sql_query, con = conn)
    # Change column names
    new_columns = ['file','date', 'company_name', 'company_code_id','commercial_name', 'BPC_customer','BPC_product', 'ship_to_party', 'material',
               'sales_qty_total_mt','3p_sales_qty_total_mt', 'gross_revenue_usd', 'discounts_usd',
               'new_net_revenue', 'n3p_net_revenue', 'net_corn', 'raw_material_other','utilities', 'waste', 'repair',
               'labor', 'ohmfg','supplies_and_packaging','supplies_indirect',   'depreciation','3p_freight_usd',
               'logistics', 'cos_other', 'new_cogs', 'freight_usd','intercompany_cost_elimination', 'gross_profit',
               'sga_toal', 'other_income_expense',   'operating_income', 'other_non_operating_income_loss','special_items',
               'interco_dividends', 'charge_back', 'exchange_gain_loss',  'intercompany_financing_cost', 'financing_costs',
               'fees_and_royalties','pbt', 'taxes_on_income', 'net_income', 'minority_income','adj_minority_income',
               'total_net_income', 'ing10000_ingr_net_income']
    df.columns = new_columns[0:]
    #Set date format
    df['date'] = pd.to_datetime(df['date']).dt.to_period('D')  
    #Drop, sort and group
    df=df.drop(columns=['file'])
    df=df.sort_values(['ship_to_party','material','date'], ascending=[True, True, True])
    df = df.groupby(['date', 'ship_to_party','material']).sum().reset_index()
    return df

def getVariablesToPredict():
    variables_costos = ['net_corn','raw_material_other', 'utilities','waste','repair','labor','ohmfg',
                    'supplies_and_packaging','supplies_indirect','depreciation','3p_freight_usd','logistics',
                    'cos_other','new_cogs']
    variables_pred = ['new_cogs', 'raw_material_other', 'net_corn', 'labor', 'depreciation', 'gross_profit', 'new_net_revenue', 'n3p_net_revenue']
    variables_tend = ['{0}_12'.format(feat) for feat in variables_pred]
    return variables_costos, variables_pred, variables_tend

def getTargetVariables(df, target):
    df_final = df.sort_values(['ship_to_party','material','date'])
    df_final['target_1_month'] = df_final.groupby(['ship_to_party','material'])[target].shift(-1)
    df_final['target_2_month'] = df_final.groupby(['ship_to_party','material'])[target].shift(-2)
    df_final['target_3_month'] = df_final.groupby(['ship_to_party','material'])[target].shift(-3)
    df_final['target_4_month'] = df_final.groupby(['ship_to_party','material'])[target].shift(-4)
    df_final['target_5_month'] = df_final.groupby(['ship_to_party','material'])[target].shift(-5)
    df_final['target_6_month'] = df_final.groupby(['ship_to_party','material'])[target].shift(-6)
    df_final['target_7_month'] = df_final.groupby(['ship_to_party','material'])[target].shift(-7)
    df_final['target_8_month'] = df_final.groupby(['ship_to_party','material'])[target].shift(-8)
    df_final['target_9_month'] = df_final.groupby(['ship_to_party','material'])[target].shift(-9)
    df_final['target_10_month'] = df_final.groupby(['ship_to_party','material'])[target].shift(-10)
    df_final['target_11_month'] = df_final.groupby(['ship_to_party','material'])[target].shift(-11)
    df_final['target_12_month'] = df_final.groupby(['ship_to_party','material'])[target].shift(-12)
    # df_final['date'] = pd.to_datetime(df_final.date, format='%Y%m')
    # df_final['date'] = pd.to_datetime(df_final.date, format='%Y-%m-%d')
    return df_final

def getShiftVariables(df_final):
    df_final['new_cogs_12'] = df_final.groupby(['ship_to_party','material'])['new_cogs'].shift(12)
    df_final['raw_material_other_12'] = df_final.groupby(['ship_to_party','material'])['raw_material_other'].shift(12)
    df_final['net_corn_12'] = df_final.groupby(['ship_to_party','material'])['net_corn'].shift(12)
    df_final['labor_12'] = df_final.groupby(['ship_to_party','material'])['labor'].shift(12)
    df_final['depreciation_12'] = df_final.groupby(['ship_to_party','material'])['depreciation'].shift(12)
    df_final['gross_profit_12'] = df_final.groupby(['ship_to_party','material'])['gross_profit'].shift(12)
    df_final['new_net_revenue_12'] = df_final.groupby(['ship_to_party','material'])['new_net_revenue'].shift(12)
    df_final['n3p_net_revenue_12'] = df_final.groupby(['ship_to_party','material'])['n3p_net_revenue'].shift(12)
    df_final['month'] = df_final.date.astype('str').map(lambda x: x[5:7])
    return df_final

def fillTendVariables(df_final, variables_tend):
    dict_tend = {}
    for feat in variables_tend:
        dict_tend[feat] = 0
    df_final.fillna(dict_tend, inplace = True)
    return df_final


def createRatioVariables(df_final, variables_pred):
    ratio_variables = ['ratio_{0}'.format(feat) for feat in variables_pred]
    for feat in variables_pred:
        df_final['ratio_{0}'.format(feat)] = df_final[feat] / (df_final['{0}_12'.format(feat)] + 0.001)
    return ratio_variables, df_final

def fillnanValues(df_final):
    target_variables = ['target_1_month', 'target_2_month', 'target_3_month', 'target_4_month',
        'target_5_month', 'target_6_month', 'target_7_month', 'target_8_month',
        'target_9_month', 'target_10_month', 'target_11_month',
        'target_12_month']
    pct_null = df_final.isnull().sum() / df_final.shape[0]
    vars_to_keep = pct_null[pct_null <= 0.3].index
    vars_to_keep = [feat for feat in vars_to_keep if feat.startswith('target_') == False]
    df_final = df_final[vars_to_keep + target_variables]
    df_final = df_final.fillna({'sales_qty_total_mt':0, 'n3p_net_revenue':0, '3p_freight_usd':0})
    df_final['3p_sales_qty_total_mt'] = np.where(df_final['3p_sales_qty_total_mt'].isnull(),
                                                df_final.sales_qty_total_mt,
                                                df_final['3p_sales_qty_total_mt'])
    return df_final

def createDummyProduct(df_final):
    list_products = ['030050-000', '020010-102', '026430-000', '06810107CA',
       '026550-000', '06442106CA', '32566109CA', '03401048CA', '06863108CA',
       '011420-000']
    df_final['c_commercial_name'] = np.where(df_final.material.isin(list_products), df_final.material, 'other')
    df_final = pd.get_dummies(df_final,prefix=['c_commercial_name'], columns = ['c_commercial_name'], drop_first=True)
    return df_final


def createDummyCustomer(df_final):
    list_customer = ['0000121058', '0000120570', '0000120728', '0000900670']
    df_final['c_ship'] = np.where(df_final.ship_to_party.isin(list_customer), df_final.ship_to_party, 'other')
    df_final = pd.get_dummies(df_final,prefix=['c_ship'], columns = ['c_ship'], drop_first=True)
    df_final = pd.get_dummies(df_final,prefix=['c_month'], columns = ['month'], drop_first=True)
    return df_final

def getDummyVariables(df_final):
    dummy_variables = [feat for feat in df_final.columns if feat.startswith('c_')]
    return dummy_variables

def predictIngredion(df_final, target, model_type, variables_costos, dummy_variables, ratio_variables, df_predict):
    if model_type == 'xgboost':
        for i in range(1,13):
            path_model = 'C:/Users/laura/OneDrive/Documents/Ingredion II/Jupyter Notebook/Modelos/'
            model = pickle.load(open(path_model + 'model_{0}_{1}_{2}_month'.format(model_type, target, str(i)), 'rb'))
            df_columns = ['sales_qty_total_mt','3p_sales_qty_total_mt'] + variables_costos + dummy_variables + ratio_variables
            matrix_final = xgb.DMatrix(df_final[df_columns])
            serie_predict = model.predict(matrix_final)
            df_predict['forecast_{0}'.format(str(i))] = serie_predict
#             predict_series = model.predict(matrix_final)
    else:
        for i in range(1,13):
            path_model = 'C:/Users/laura/OneDrive/Documents/Ingredion II/Jupyter Notebook/Modelos/'
            model = pickle.load(open(path_model + 'model_{0}_{1}_{2}_month'.format(model_type, target, str(i)), 'rb'))
            serie_predict = model.predict(df_final[['sales_qty_total_mt','3p_sales_qty_total_mt'] + variables_costos + dummy_variables + ratio_variables])
            df_predict['forecast_{0}'.format(str(i))] = serie_predict
#             predict_series = model.predict(df_final[['sales_qty_total_mt','3p_sales_qty_total_mt'] + variables_costos + dummy_variables + ratio_variables])
    return df_predict

#   US forecast

df = getIngredioData()
variables_costos, variables_pred, variables_tend = getVariablesToPredict()

# target_variables = ['sales_qty_total_mt','3p_sales_qty_total_mt','gross_revenue_usd','discounts_usd','new_net_revenue',
#                    'n3p_net_revenue','net_corn','raw_material_other','utilities','waste','repair','labor','ohmfg',
#                    'supplies_and_packaging', 'supplies_indirect','depreciation','3p_freight_usd','logistics','cos_other',
#                    'new_cogs','freight_usd', 'gross_profit', 'operating_income']
target_variables = ['3p_sales_qty_total_mt','n3p_net_revenue','new_cogs', 'operating_income']
models = ['lasso','gbm','xgboost']

df_general = df.sort_values(['ship_to_party','material','date'])
# df_general['date'] = pd.to_datetime(df_general.date, format='%Y%m')
# df_general['date'] = pd.to_timestamp(df_general.date, format='%Y-%m-%d')
dict_df = {'lasso':pd.DataFrame(), 'gbm':pd.DataFrame(), 'xgboost': pd.DataFrame()}
for target in target_variables:
    for model_type in models:
        print('')
        print('#################################################')
        print('#################################################')
        print('starting with: {0} for model {1}'.format(target, model_type))
        print('#################################################')
        print('#################################################')
        print('')
        df_final = getTargetVariables(df, target)
        df_final = getShiftVariables(df_final)
        df_final = fillTendVariables(df_final, variables_tend)
        ratio_variables, df_final = createRatioVariables(df_final, variables_pred)
        df_final = fillnanValues(df_final)
        df_final = createDummyProduct(df_final)
        df_final = createDummyCustomer(df_final)
        dummy_variables = getDummyVariables(df_final)
        pivot = predictIngredion(df_final, target, model_type,
                                     variables_costos, dummy_variables,
                                     ratio_variables, df_general)
        pivot['model_type'] = model_type
        pivot['feature_forecasted'] = target
        pivot = pivot[pivot.date > '2017-12-01']
        dict_df[model_type] = pd.concat([dict_df[model_type], pivot])

dict_df['xgboost']

features_forecast = [feat for feat in dict_df['lasso'].columns if feat.startswith('forecast_')]

#   To upload data to Azure container
connect_str = 'DefaultEndpointsProtocol=https;AccountName=storingrediondeveastus;AccountKey=hQR7K42hnJ+TKj1qzDlMVXw5gPOV9uLVBJ44WcXfU+voh+g0YHnNUmLZ8EsUylRiWIOOhnHN7gJ6oomN274ipg==;EndpointSuffix=core.windows.net'
container_name = 'ingredion-phase-2'

for modelo in models:
    df_lasso = dict_df[modelo][['date','ship_to_party','material','model_type', 'feature_forecasted'] + features_forecast]
    df_lasso = df_lasso.melt(id_vars=['date','ship_to_party','material','model_type', 'feature_forecasted'],
            var_name='window_prediction',
            value_name='forecast')
    df_lasso['window_prediction'] = df_lasso.window_prediction.map(lambda x: int(x.split('_')[-1]))
    df_parquet = df_lasso.to_parquet(engine='pyarrow')
    blob_name = 'forecast_MEX_{0}.parquet'.format(modelo)
    blob  = BlobClient.from_connection_string(connect_str, container_name = container_name, blob_name=blob_name)
    blob.upload_blob(data=df_parquet)
    df_lasso = 0
    df_parquet = 0
    print('model {0} Done'.format(modelo))

dict_df['lasso'][['date','ship_to_party','material','model_type', 'feature_forecasted'] + features_forecast].to_csv('C:\\Users\\laura\\OneDrive\\Documents\\Ingredion II\\Jupyter Notebook\\Modelos\\ingredion_results_lasso_MEX.csv')
dict_df['gbm'][['date','ship_to_party','material','model_type', 'feature_forecasted'] + features_forecast].to_csv('C:\\Users\\laura\\OneDrive\\Documents\\Ingredion II\\Jupyter Notebook\\Modelos\\ingredion_results_gbm_MEX.csv')
dict_df['xgboost'][['date','ship_to_party','material','model_type', 'feature_forecasted'] + features_forecast].to_csv('C:\\Users\\laura\\OneDrive\\Documents\\Ingredion II\\Jupyter Notebook\\Modelos\\ingredion_results_xgboost_MEX.csv')

