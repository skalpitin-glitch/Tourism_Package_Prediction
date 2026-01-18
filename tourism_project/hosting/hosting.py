from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
# Uploading the deployment files to HF Space from the local folder.
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="skalpitin/Tourism-Package-Prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
