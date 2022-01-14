# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 09:55:24 2021

@author: lfranco
"""

import os
import pandas as pd

csv_name = 'ingredion_us_2016_2021_by_customer_and_product_flatLE0.csv'

## Set path and list files
path= os.path.abspath('C:\\Users\\laura\\OneDrive\\Documents\\Ingredion II\\Data\\USA\\LE0\\')
os.chdir(path)
files = os.listdir(path)

export_path= os.path.abspath('C:\\Users\\laura\\OneDrive\\Documents\\Ingredion II\\Data\\')

## Creates empty df
df = pd.DataFrame()
## Loop to import and append files
for file in files:
 df = df.append(pd.read_excel(
    file,
    sheet_name= "Data",
    skiprows=range(0, 15),
    usecols = "G:BC" ))


## Format file
    #   Change column name
 df.columns.values[2] = 'company_code_id'
    #   Reset index
 df = df.reset_index(drop = True)
    #   Fill all NANs with 0
 df = df.fillna(0)
    #   Remove the "Result" line
 df = df[df['Company Code'] != 'Result']
    #   Split fiscal year and order data frame
dates = pd.DataFrame(data = df['Fiscal year/period'].astype(str).str.split(pat = '.', expand = True).astype(int).values, columns = ['month', 'year'])
    #  This is to have the correct value for years, have trouble with years that finish in 0 because it has been read as number.
dates.year[dates.year < 2016] = dates.year[dates.year < 2016]*10
# month
df.insert(loc = 1,
          column = 'month',
          value = dates.iloc[:, 0])
# year
df.insert(loc = 2,
          column = 'year',
          value = dates.iloc[:, 1])
# Sort
df=df.sort_values(['year', 'month'], ascending=[True, True])
df=df.drop(columns=['year', 'month'])


# Export to csv
df.to_csv((export_path + csv_name), mode = 'a', index = True , header = True)
df.to_csv((export_path + csv_name), mode = 'a', index = True , header = True)
