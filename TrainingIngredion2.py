#   Ingredion model train

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

import pickle
import sys
import glob
import os
plt.style.use('ggplot')


def add_days(date, days, format_input='%Y-%m-%d', format_output='%Y-%m-%d'):
    ''' Resta o suma dias a una fecha que se le indique en el argumento

        :type date: STRING
        :param date: Fecha a la que le queremos adicionar dias

        :type days: INT
        :param days: No. dias que queremos sumar, el numero puede ser negativo para restar dias

        :type format_input: STRING
        :param format_input: Formato en el que introducimos la fecha, fault = '%Y%m%d'

        :type format_output: STRING
        :param format_input: Formato en el que recibiremos la fecha, fault = '%Y%m%d'

        :raises: TelefonicaError
    '''
    final_date = str(datetime.strptime(date, format_input) + timedelta(days=days))[0:10]
    return datetime.strptime(final_date, '%Y-%m-%d').strftime(format_output)

def getIngredioData():
    # En esta parte debo anexar el codigo con el que obtendremos los datos consolidados
    # df = pd.read_csv('test_df.csv')
    df = pd.read_csv('C:\\Users\\laura\\OneDrive\\Documents\\Ingredion II\\Data\\US\\ingredion_us_2016_2021_by_customer_and_product.csv')
    # df.fiscal_year_period = df.fiscal_year_period.astype('str')
    # df['month'] = df.fiscal_year_period.map(lambda x: x.split('.')[0].rjust(2, '0'))
    # df['year'] = df.fiscal_year_period.map(lambda x: x.split('.')[1].ljust(4, '0'))
    df.insert(loc = 0,
               column = 'date',
               value = pd.to_datetime(df.year.astype(str) + '/' + df.month.astype(str) ))

    df['date'] = pd.to_datetime(df['date']).dt.to_period('D')
    df=df.sort_values(['ship_to_party','material','year', 'month'], ascending=[True, True, True, True])
    df=df.drop(columns=['year', 'month', 'fiscal_year_period'])
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
    df_final = df_final.fillna({'sales_qty_total_mt':0, 'n3p_net_revenue':0, '3p_freight_usd':0}) # Check code the original has "f_final" instead "df_final"
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

def filterByDate(df_final, today):
    day_max = add_days(today, 365, format_input = '%Y-%m-%d', format_output = '%Y-%m-%d')
    day_max[:7] + '-01'
    df_final = df_final[df_final.date < day_max]
    return df_final

def getDummyVariables(df_final):
    dummy_variables = [feat for feat in df_final.columns if feat.startswith('c_')]
    return dummy_variables

def trainingModel(df_final, variables_costos, dummy_variables, ratio_variables, model_type, target):
    dict_model = {'lasso':Lasso(alpha=0.8), 'gbm':GradientBoostingRegressor()}
    if model_type == 'xgboost':
        for i in range(1,13):
            X = df_final[df_final['target_{0}_month'.format(str(i))].isna() == False][['sales_qty_total_mt','3p_sales_qty_total_mt'] + variables_costos + dummy_variables + ratio_variables]
            y = df_final[df_final['target_{0}_month'.format(str(i))].isna() == False]['target_{0}_month'.format(str(i))]
            X_train , X_test ,y_train, y_test = train_test_split(X,y, random_state=123, test_size=0.2)
            matrix_train = xgb.DMatrix(X_train, label = y_train)
            matrix_test = xgb.DMatrix(X_test, label = y_test)
            model_xgb = xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'},
                                                  dtrain = matrix_train, num_boost_round = 500, early_stopping_rounds = 20,
                                                  evals = [(matrix_test,'test')])
            print('Train RMSE for {0} month: '.format(str(i)),model_xgb.eval(matrix_train))
            print('Test  RMSE for {0} month: '.format(str(i)),model_xgb.eval(matrix_test))
            print('Train  R2 Score : %.2f'%r2_score(y_train, model_xgb.predict(matrix_train)))
            print('Test R2 Score : %.2f'%r2_score(y_test, model_xgb.predict(matrix_test)))
            path_model = 'C:/Users/laura/OneDrive/Documents/Ingredion II/Jupyter Notebook/Modelos/'
            model_name = 'model_{0}_{1}_{2}_month'.format(model_type, target, str(i))
            pickle.dump(model_xgb, open(path_model + model_name, 'wb'))
    else:
        for i in range(1,13):
            X = df_final[df_final['target_{0}_month'.format(str(i))].isna() == False][['sales_qty_total_mt','3p_sales_qty_total_mt'] + variables_costos + dummy_variables + ratio_variables]
            y = df_final[df_final['target_{0}_month'.format(str(i))].isna() == False]['target_{0}_month'.format(str(i))]
            X_train , X_test ,y_train, y_test = train_test_split(X,y, random_state=123, test_size=0.2)
            model = dict_model[model_type]
            model.fit(X_train,y_train)
            print('Train RMSE for {0} month: '.format(str(i)), math.sqrt(mean_squared_error(y_train, model.predict(X_train))))
            print('Test  RMSE for {0} month: '.format(str(i)), math.sqrt(mean_squared_error(y_test, model.predict(X_test))))
            print('Train  R2 Score : %.2f'%r2_score(y_train, model.predict(X_train)))
            print('Test R2 Score : %.2f'%r2_score(y_test, model.predict(X_test)))
            path_model = 'C:/Users/laura/OneDrive/Documents/Ingredion II/Jupyter Notebook/Modelos/'
            model_name = 'model_{0}_{1}_{2}_month'.format(model_type, target, str(i))
            pickle.dump(model, open(path_model + model_name, 'wb'))

#   Model Training
df = getIngredioData()
variables_costos, variables_pred, variables_tend = getVariablesToPredict()

today = '2020-01-01'
# target_variables = ['sales_qty_total_mt','3p_sales_qty_total_mt','gross_revenue_usd','discounts_usd','new_net_revenue',
#                    'n3p_net_revenue','net_corn','raw_material_other','utilities','waste','repair','labor','ohmfg',
#                    'supplies_and_packaging', 'supplies_indirect','depreciation','3p_freight_usd','logistics','cos_other',
#                    'new_cogs','freight_usd', 'gross_profit', 'operating_income']
# target_variables = ['utilities','waste','repair','labor','ohmfg',
#                    'supplies_and_packaging', 'supplies_indirect','depreciation','3p_freight_usd','logistics','cos_other',
#                   'freight_usd', 'gross_profit', 'operating_income']
target_variables = ['3p_sales_qty_total_mt','n3p_net_revenue','new_cogs', 'operating_income']
models = ['lasso','gbm','xgboost']
for target in target_variables:
    for model_type in models:
        print('#################################################')
        print('#################################################')
        print('#################################################')
        print('starting with: {0} for model {1}'.format(target, model_type))
        print('#################################################')
        print('#################################################')
        print('#################################################')
        df_final = getTargetVariables(df, target)
        df_final = getShiftVariables(df_final)
        df_final = fillTendVariables(df_final, variables_tend)
        ratio_variables, df_final = createRatioVariables(df_final, variables_pred)
        df_final = fillnanValues(df_final)
        df_final = createDummyProduct(df_final)
        df_final = createDummyCustomer(df_final)
        df_final = filterByDate(df_final, today)
        dummy_variables = getDummyVariables(df_final)
        trainingModel(df_final, variables_costos, dummy_variables, ratio_variables, model_type, target)


df_final = getTargetVariables(df, target)
