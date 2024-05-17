import numpy as np 
import random, time, operator 
import os 
import torch 
from torch.utils.data import Dataset
import librosa
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import librosa.display


def Padding(data):

    lengths=[]
    for matrix in data:
        lengths.append(matrix.shape[0])
    max_len=np.max(lengths)
    Pdata=[]
    for matrix in data:
        Pdata.append(np.pad(matrix,((0,max_len-matrix.shape[0]),(0,0)),mode='constant',constant_values=0))
    
    return np.asarray(Pdata), lengths

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

