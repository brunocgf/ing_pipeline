import io, os, uuid, yaml
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from src.pipeline import utils, azure_tools

if __name__ == "__main__":
    # Read credentials
    credentials = utils.load_credentials("blob_storage")

    # Files in raw container
    container_input = "reports-ingredion"
    container_client = azure_tools.get_container_client(container_input,credentials['conn_string'])
    blob_list = azure_tools.get_blob_list(container_client)

    # Tranform and save
    container_output = "intermidiate-ingredion"
    azure_tools.transform_pl_files(blob_list, container_input, container_output, credentials['conn_string'])