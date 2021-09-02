import os, uuid, yaml
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


with open("./credentials.yaml","r") as c:
    credentials = yaml.safe_load(c)['blob_storage']

blob_service_client = BlobServiceClient.from_connection_string(credentials['conn_string'])
 
# name of the existing container
container_name = "reports-ingredion"
container_client = blob_service_client.get_container_client(container_name)

blob_list = []
blob_it = container_client.list_blobs()
for blob in blob_it:
    file_name = blob.name
    print(file_name)
    blob_list.append(file_name)
