import argparse, io, os, uuid, yaml
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from src.pipeline import utils, azure_tools

my_parser = argparse.ArgumentParser(description="Files to upload. In None, all files are upload")
my_parser.add_argument('--file',
default = None)

args = my_parser.parse_args()
file_input = args.file

if __name__ == "__main__":
    # Read credentials
    credentials = utils.load_credentials("blob_storage")
    container_input = "reports-ingredion"
    container_output = "intermediate-ingredion"

    if file_input is None:
         # Files in raw container
        print("Processing all data")
        container_client = azure_tools.get_container_client(container_input,credentials['conn_string'])
        blob_list = azure_tools.get_blob_list(container_client)

    else:
        blob_list = [file_input]

    # Tranform and save

    azure_tools.transform_pl_files(blob_list, container_input, container_output, credentials['conn_string'], country="US")
    azure_tools.transform_pl_files(blob_list, container_input, container_output, credentials['conn_string'], country="Brazil")
