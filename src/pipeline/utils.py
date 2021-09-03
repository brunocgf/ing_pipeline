import yaml

# Read credentials from file
def load_credentials(credential, file = "./credentials.yaml"):
    
    with open(file,"r") as c:
        credentials = yaml.safe_load(c)[credential]

    return credentials