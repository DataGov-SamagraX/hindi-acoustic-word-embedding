import ray
import pandas as pd
import numpy as np
import librosa
import os
from tqdm import tqdm
from utils import _load_config
import json 

config_file=_load_config()

if ray.is_initialized():
    ray.shutdown()
ray.init()

@ray.remote
def compute_mfcc(audio_path, n_mfcc=13):
    try:
        y, sr = librosa.load(audio_path)
        n_fft = min(2048, len(y))
        hop_length = n_fft // 4
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr,
                                     n_mfcc=n_mfcc,
                                     n_fft=n_fft,
                                     hop_length=hop_length)
        
        width = min(9, mfccs.shape[1])
        if width < 3:
            width = 3
        
        width = min(width, mfccs.shape[1])
        
        if width % 2 == 0:
            width -= 1
        
        delta1 = librosa.feature.delta(mfccs, order=1, width=width)
        delta2 = librosa.feature.delta(mfccs, order=2, width=width)
        
        mfccs_combined = np.concatenate((mfccs, delta1, delta2), axis=0)
        return mfccs_combined.shape[1]
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return 0    
   
def get_max_mfcc_length(dataset_path, n_mfcc=13, batch_size=8000,flag="dev"):
    df = pd.read_csv(dataset_path)
    dir_path = os.path.dirname(dataset_path)
    
    audio_paths = [os.path.join(dir_path, str(path)) for path in df["audio_path"]]
    
    max_len = 0
    num_batches = len(audio_paths) // batch_size + (1 if len(audio_paths) % batch_size != 0 else 0)
    
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(audio_paths))
        batch_paths = audio_paths[start_idx:end_idx]
        
        mfcc_length_ids = [compute_mfcc.remote(path, n_mfcc) for path in batch_paths]
        mfcc_lengths = ray.get(mfcc_length_ids)
        
        batch_max = max(mfcc_lengths)
        max_len = max(max_len, batch_max)
    
    print(f"Max MFCC length: {max_len}")

    if flag=="train":
        config_file["max_mfcc_train"]=max_len
    else:
        config_file["max_mfcc_dev"]=max_len
    
    with open("config.json", 'w') as json_file:
        json.dump(config_file, json_file,indent=4)


    return max_len


ray.shutdown()

if __name__=='__main__':
    root_path='/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_dataset'
    dev_csv='sample_bhashini_dev.csv'
    train_csv='sample_bhashini_train.csv'
    get_max_mfcc_length(os.path.join(root_path,train_csv),flag="train")
    get_max_mfcc_length(os.path.join(root_path,dev_csv),flag="dev")