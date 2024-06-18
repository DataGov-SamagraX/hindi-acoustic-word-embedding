import numpy as np 
import random, time, operator 
import os 
import torch 
from utils import _load_vocab
from utils import load_audio
from torch.utils.data import Dataset,DataLoader
import librosa
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import librosa.display
import ast 

VOCAB_DICT=_load_vocab()

class MultiViewDataset(Dataset):
    def __init__(self,csv_file,loss_fn,n_mfcc=13):
        self.data=pd.read_csv(csv_file)
        self.dir_path=os.path.dirname(csv_file)
        self.vocab_dict=VOCAB_DICT
        self.n_mfcc=n_mfcc
        self.loss_fn=loss_fn 

    def __len__(self):
        return len(self.data)
    
    def char_to_idx(self,transcript):
        
        one_hot=torch.zeros(len(transcript),len(self.vocab_dict))
        for i,char in enumerate(transcript):
            one_hot[i,self.vocab_dict[char]]=1 
        
        return one_hot
    
    def compute_mfcc(self,audio_path):

        y,sr=librosa.load(audio_path)

        n_fft = min(2048, len(y))
        hop_length = n_fft // 4

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=n_fft, hop_length=hop_length)

        width = min(9, mfccs.shape[1])
        if width < 3:
            width = 3
        
        width = min(width, mfccs.shape[1])

        if width % 2 == 0:
            width -= 1

        
        delta1=librosa.feature.delta(mfccs,order=1,width=width)
        delta2=librosa.feature.delta(mfccs,order=2,width=width)

        mfccs_combined=np.concatenate((mfccs,delta1,delta2),axis=0)

        return mfccs_combined
    
    def __getitem__(self,idx):
        
        audio_path_x1=self.data["audio_path"][idx]
        audio_path_x1=os.path.join(self.dir_path,str(audio_path_x1))
        transcript_c1=self.data["transcript"][idx]

        sample_list=ast.literal_eval(self.data["negative_samples"][idx])
        
        one_hot_c2=None
        if 0 in self.loss_fn or 1 in self.loss_fn:
            transcript_c2=random.choice(sample_list)[1]
            one_hot_c2=self.char_to_idx(transcript_c2)
        
        mfccs_x2=None
        if 2 in self.loss_fn or 3 in self.loss_fn:
            audio_path_x2=random.choice(sample_list)[0]
            audio_path_x2=os.path.join(self.dir_path,str(audio_path_x2))

            mfccs_x2=self.compute_mfcc(audio_path_x2)
        
        #computing mfcc
        mfcc_x1=self.compute_mfcc(audio_path_x1) 
        
        #computiing one hot for transcript 
        one_hot_c1=self.char_to_idx(transcript_c1)

        output_tensors= [torch.tensor(mfcc_x1), one_hot_c1]

        if 0 in self.loss_fn or 1 in self.loss_fn:
            output_tensors.append(one_hot_c2)
    
        if 2 in self.loss_fn or 3 in self.loss_fn:
            output_tensors.append(torch.tensor(mfccs_x2))
    
        return output_tensors

def pad_mfccs(mfccs, max_len):
    padded_mfccs = []
    for mfcc in mfccs:
        # Padding to the right with zeros
        pad_width = max_len - mfcc.shape[1]
        padded_mfcc = torch.nn.functional.pad(mfcc, (0, pad_width), 'constant', 0)
        padded_mfccs.append(padded_mfcc)
    return torch.stack(padded_mfccs)

def pad_sequence(sequences, batch_first=False, padding_value=0):
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)

def collate_fn(batch):

    mfccs_x1= []
    one_hot_c1=[]
    one_hot_c2=[]
    mfccs_x2=[]

    for item in batch:
        mfcc_x1,oh_c1 = item[0], item[1]
        mfccs_x1.append(mfcc_x1)
        one_hot_c1.append(oh_c1)

        if len(item) == 4:
            oh_c2,mfcc_x2= item[2], item[3]
            one_hot_c2.append(oh_c2)
            mfccs_x2.append(mfcc_x2)
        
        elif len(item) == 3:
            if item[2].shape[1] == len(VOCAB_DICT):
                oh_c2=item[2]
                one_hot_c2.append(oh_c2)
            else:
                mfcc_x2=item[2]
                mfccs_x2.append(mfcc_x2)
    
    max_mfcc_len_x1=max(mfcc.shape[1] for mfcc in mfccs_x1)
    max_mfcc_len_x2=max(mfcc.shape[1] for mfcc in mfccs_x2) if mfccs_x2 else 0

    mfccs_x1=pad_mfccs(mfccs_x1,max_mfcc_len_x1)
    one_hot_c1=pad_sequence(one_hot_c1, batch_first=True)

    result={'view1_x1':mfccs_x1,'view2_c1':one_hot_c1}

    if one_hot_c2:
        one_hot_c2=pad_sequence(one_hot_c2,batch_first=True)
        result['view2_c2']=one_hot_c2
    
    if mfccs_x2:
        mfccs_x2=pad_mfccs(mfccs_x2, max_mfcc_len_x2)
        result['view1_x2']=mfccs_x2
    
    return result 

def get_loader(csv_file,batch_size,loss_fn):

    dataset=MultiViewDataset(csv_file=csv_file,loss_fn=loss_fn)

    loader=DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    return loader




#TODO: Dev and Test DataLoader        
class MultiviewDevDataset(Dataset):

    def __init__():
        pass 
    def __len__():
        pass 
    def __getitem__():
        pass 