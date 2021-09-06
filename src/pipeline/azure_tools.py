import io, re
from logging import error
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

def get_container_client(container_name, conn_string):
    """
    Return blob service client using a connection string
    """
    # Get Blob Client
    blob_client = BlobServiceClient.from_connection_string(conn_string)

    return blob_client.get_container_client(container_name)

def get_blob_list(container_client):
    """
    Get blobs in a container
    """

    blob_list = []
    for blob in container_client.list_blobs():
        file_name = blob.name
        blob_list.append(file_name)

    return blob_list

def read_xlfile(file_name, container, credentials, **pd_args):
    """
    Read Excel file from Azure Blob Storage
    """

    container_client = get_container_client(container, credentials)

    # Get streaming object
    stream = container_client.download_blob(file_name)

    # Save in dataframe
    file = io.BytesIO()
    stream.readinto(file)
    df = pd.read_excel(file, **pd_args)

    return df

def upload_df_parquet(df, name, container, credentials):
    """
    Upload DataFrame as parquet to Blob Storage
    """

    blob  = BlobClient.from_connection_string(credentials, container_name = container, blob_name=name)

    parquet_file = io.BytesIO()
    df.to_parquet(parquet_file, engine='pyarrow')
    parquet_file.seek(0)
    blob.upload_blob(data=parquet_file)

    print("Uploded file: ", name)

def transform_pl_files(blob_list, container_input, container_output, credentials, country="US"):
    """
    Transform PL files for a giver country
    """
    # Select only country needed
    r = re.compile(f".*{country}.*")

    filtered_list = list(filter(r.match, blob_list))

    if len(filtered_list) ==0:
        raise ValueError("Country not founded")

    # filtered_list.sort()

    # Read and append files
    #df_total = pd.DataFrame()
    for file in filtered_list:
        print("Processing: ", file)
        df = read_xlfile(file, container_input, credentials, sheet_name="Data", skiprows=15)
        df = df.iloc[:,6:]
        df = df[df["Company Code"] != "Result"]
        name = f"{country}/"+file[:file.find(country)-1]+".parquet"
        print("Uploading: ", file)
        upload_df_parquet(df, name, container_output, credentials)
        #df_total = pd.concat([df_total, df])

    #return df_total

def read_transform_pl(container, credentials, country="US"):

    container_client = get_container_client(container, credentials)
    blob_list = get_blob_list(container_client)

    r = re.compile(f".*{country}.*")
    filtered_list = list(filter(r.match, blob_list))

    df_list=[]
    for file in filtered_list:
        stream = container_client.download_blob(file)
        file = io.BytesIO()
        stream.readinto(file)
        df_f = pd.read_parquet(file)
        df_list.append(df_f)

    return(pd.concat(df_list))



    



