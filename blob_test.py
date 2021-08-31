import os, uuid, yaml
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

try:
    print("Azure Blob Storage v" + __version__ + " - Python quickstart sample")

except Exception as ex:
    print('Exception:')
    print(ex)

with open("./credentials.yaml","r") as c:
    credentials = yaml.safe_load(c)['blob_storage']

# blob_service_client = BlobServiceClient(
#     account_url="https://<my_account_name>.blob.core.windows.net",
#     credential=token_credential
# )

blob_service_client = BlobServiceClient.from_connection_string(credentials['conn_string'])
 
# name of the existing container
container_name = "ingredion-data"
container_client = blob_service_client.get_container_client(container_name)

# blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)


# list_of_blobs = []
blob_list = container_client.list_blobs()
print(blob_list)
for blob in blob_list:
    # blobs_dict = dict()
    # dataset = blob.container
    # file_name = blob.name
    # size = blob.size
    # blobs_dict['Filename'] = file_name
    print(blob)
    # blobs_dict['Azure Location'] = blob.name
    # blobs_dict['File Size'] = round(size/1024, 2)
    # blobs_dict['Folder'] = dataset

    # list_of_blobs.append(blobs_dict)
