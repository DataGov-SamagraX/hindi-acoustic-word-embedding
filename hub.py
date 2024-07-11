from huggingface_hub import HfApi

root_path='/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_dataset'
HF_TOKEN="..."

api = HfApi(endpoint="https://huggingface.co", token=HF_TOKEN)
api.create_repo(repo_id='SamagraDataGov/hindi-acoustic-embedding-dataset', repo_type="dataset", exist_ok=True)
api.upload_folder(
            folder_path=root_path,
            repo_id='SamagraDataGov/hindi-acoustic-embedding-dataset',
            repo_type="dataset",
        )