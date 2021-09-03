import io, re
from logging import error
import numpy as np
import pandas as pd
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

def read_xlfile(file_name, container_client, **pd_args):
    """
    Read Excel file from Azure Blob Storage
    """

    # Get streaming object
    stream = container_client.download_blob(file_name)

    # Save in dataframe
    file = io.BytesIO()
    stream.readinto(file)
    df = pd.read_excel(file, **pd_args)

    return df

def transform_pl_files(blob_list, container_client, country="US"):
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
    df = pd.DataFrame()
    for file in filtered_list:
        print("Processing: ", file)
        pivot = read_xlfile(file, container_client, sheet_name="Data", skiprows=15)
        pivot = pivot.iloc[:,6:]
        pivot = pivot[pivot["Company Code"] != "Result"]
        df = pd.concat([df, pivot])

    return df






