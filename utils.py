import torch 
import subprocess 
import json 
import numpy as np
import os


def load_audio():
    pass 

def average_precision():
    pass 

def save_checkpoint(state,filename):
    print("=> saving checkpoint")

    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    filepath = os.path.join(checkpoint_dir, filename)
    print(f"=> Saving checkpoint to {filepath}")
    torch.save(state, filepath)
    
def load_checkpoint(checkpoint,model,optimizer):
    
    print("=> loading checkpoint")
    
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step=checkpoint["step"]
    
    return step 

def _load_config():
    with open("config.json",'r') as f:
        config_file=json.load(f)
    
    return config_file 

def _load_vocab():
    with open("vocab.json",'r') as f:
        json_file=json.load(f)
    
    return dict(json_file)


def extract_root_path(file_path):
    directory = os.path.dirname(file_path)
    
    parts = directory.split(os.path.sep)
    
    root_directory_path = os.path.join(parts[0], parts[1])
    
    return root_directory_path

