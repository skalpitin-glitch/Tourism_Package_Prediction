from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Specifying the HF space details for storing the dataset 
repo_id = "skalpitin/Tourism-Package-Prediction"
repo_type = "dataset"

# Initializing HF API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Checking if the HF space exists, if not then creating the space else using it. 
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

#Uploading the dataset from local folder to HF space. 
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
