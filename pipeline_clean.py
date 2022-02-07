import datetime
import io, os, uuid, yaml, urllib
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from src.pipeline import utils, azure_tools
import sqlalchemy

def format_file(df):
    df_clean = df.copy()

    #   Change column name
    df_clean.columns.values[2] = 'company_code_id'
    #   Reset index
    df_clean = df_clean.reset_index(drop = True)
    #   Fill all NANs with 0
    df_clean = df_clean.fillna(0)
    #   Remove the "Result" line
    df_clean = df_clean[df_clean['Company Code'] != 'Result']
    #   Split fiscal year and order data frame
    dates = pd.DataFrame(data = df_clean['Fiscal year/period'].astype(str).str.split(pat = '.', expand = True).astype(int).values, columns = ['month', 'year'])
    df_clean.insert(
        loc = 0,
        column = 'date',
        value = dates.year.astype(str).str.ljust(4,'0')+dates.month.astype(str).str.rjust(2,'0')+"01")

    df_clean = df_clean.astype({
        'Company Code':str,
        'Commercial Name': str,
        'BPC: Customer': str,
        'BPC: Product': str,
        'Ship to party': str,
        'Material': str
    })

    return df_clean

credentials = utils.load_credentials("blob_storage")
container_input = "ingredion-data"
container_client = azure_tools.get_container_client(container_input,credentials['conn_string'])
blob_list = azure_tools.get_blob_list(container_client)
blob_list_coun = [i for i in blob_list if "US/LE2" in i]
with open("credentialsdb.yml") as stream:
    cred = yaml.safe_load(stream)
params = urllib.parse.quote_plus(f"DRIVER={cred['DRIVER']};"
                                 f"SERVER={cred['SERVER']};"
                                 f"DATABASE={cred['DATABASE']};"
                                 f"UID={cred['UID']};"
                                 f"PWD={cred['PWD']}")
#engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
for f in blob_list_coun:
# for f in ['MEX/LE0/02-2018 MEX LE0 P&L by Customer & Product(ZBW_ZANDTM001_Q00NI).xlsm',
#           'MEX/LE0/02-2019 Mex LE0 P&L by Customer & Product(ZBW_ZANDTM001_Q00NI).xlsm']:

    country, type, file_name = f.split("/")
    stream = container_client.download_blob(f)
    file = io.BytesIO()
    stream.readinto(file)
    df = pd.read_excel(file,
                       sheet_name="Data",
                       skiprows=range(0, 15),
                       usecols = "G:BC")
    print("Limpiando: ", file_name)
    df = format_file(df)
    df.insert(loc=0, column="Country", value=country)
    df.insert(loc=0, column="Type", value=type)
    df.insert(loc=0, column="File", value=file_name)

    table_name = "pl_data_" + type.lower()
    print("Subiendo a: ", table_name)
    engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)
    df.to_sql(table_name, con=engine, schema='clean', if_exists='append', index=False)
    print("Listo: ", file_name)
    print("-----", datetime.datetime.now(),'-----')