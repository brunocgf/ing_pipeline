import numpy as np
import pandas as pd
#import azureml.core
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Dataset

# Load the workspace from the saved config file
#forced_interactive_auth = InteractiveLoginAuthentication(tenant_id="cef74873-15b3-47e9-979d-ceb99ef2b632", force=True)
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# Previously set datastore linked to ingredion as default
default_ds = ws.get_default_datastore()

# Load datastore
tab_data_set = Dataset.Tabular.from_delimited_files(
    path=(default_ds, 'CAN/Consolidated Data/ingredion_can_2016_2021_by_customer_and_product_flat.csv')
)

df = tab_data_set.take(3).to_pandas_dataframe()
print(df)