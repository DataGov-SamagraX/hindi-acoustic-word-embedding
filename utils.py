import torch 
import subprocess 
import json 
import numpy as np

def load_audio():
    pass 

def average_precision():
    pass 

def save_checkpoint(state,filename="my_checkpoint.pth.tar"):
    print("=> saving checkpoint")
    
    torch.save(state,filename)
    
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

