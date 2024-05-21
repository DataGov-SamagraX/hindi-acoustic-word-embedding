import numpy as np 
import random, time, operator 
import os 
import torch 
from utils import _load_vocab
from torch.utils.data import Dataset
import librosa
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import librosa.display

VOCAB_DICT=_load_vocab()

def collate_fn(batch):
    mfccs,one_hot=zip(*batch)

    max_mfcc_len=max(len(mfcc) for mfcc in mfccs)
    max_one_hot_len=max(len(oh) for oh in one_hot)

    mfccs=torch.nn.utils.rnn.pad_sequence(mfccs,batch_first=True)
    one_hot=torch.nn.utils.rnn.pad_sequence(one_hot,batch_first=True,padding_value=0)

    return mfccs,one_hot

class AudioDataset(Dataset):

    def __init__(self,csv_file,n_mfcc=13):

        if isinstance(csv_file,str):
            self.data=pd.read_csv(csv_file)
        self.data=csv_file
        self.n_mfcc=n_mfcc
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):

        audio_path = self.data['file_path'][idx]
        y,sr=librosa.load(audio_path)

        mfccs=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=self.n_mfcc)
        delta1=librosa.feature.delta(mfccs,order=1)
        delta2=librosa.feature.delta(mfccs,order=2)

        mfccs_combined=np.concatenate((mfccs,delta1,delta2),axis=0)

        return mfccs_combined
    
    def visualize_mfcc(self,idx):
        audio_path = self.data['file_path'][idx]
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(mfccs, x_axis='time', sr=sr, hop_length=512, cmap='viridis')
        plt.colorbar()
        plt.title('MFCCs')
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficients')
        plt.tight_layout()
        plt.show()

class CharacterDataset(Dataset):

    def __init__(self,csv_file):

        if isinstance(csv_file,"str"):
            self.data=pd.read_csv(csv_file)
        self.data=csv_file
        self.vocab_dict=VOCAB_DICT
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        transcript=self.data['transcript'][idx]
        one_hot=torch.zeros(len(transcript),len(self.vocab_dict))
        for i,char in enumerate(transcript):
            one_hot[i,self.vocab_dict[char]]=1 
        
        return one_hot

class MultiViewDataset(Dataset):

    def __init__(self,csv_file,n_mfcc=13):
        
        if isinstance(csv_file,"str"):
            self.data=pd.read_csv(csv_file)
        self.data=csv_file 
        self.n_mfcc=n_mfcc
        self.audio_dataset=AudioDataset(self.data,self.n_mfcc)
        self.char_dataset=CharacterDataset(self.data)

    def __len__(self):
        return len(self.data) 
    def __getitem__(self, idx):
        
        mfccs=self.audio_dataset[idx]
        one_hot=self.char_dataset[idx]

class MultiviewDevDataset(Dataset):

    def __init__():
        pass 
    def __len__():
        pass 
    def __getitem__():
        pass 


#hard_negative examples
#one hard negative sampler for our train dataset
#one hard negative sampler for our dev dataset
#dataloader {view1,view2,x2/c2}
#it will provide three vectors depending on the type of x2/c2 
# 
