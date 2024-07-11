import numpy as np 
import os 
import torch 
from utils import _load_vocab
from torch.utils.data import Dataset,DataLoader
import librosa
import numpy as np 
import pandas as pd 
import librosa.display
import ast 
from sampler import edit_distance
from abc import ABC,abstractmethod


VOCAB_DICT=_load_vocab()

# Dataloader class
class MultiViewDataset(Dataset,ABC):
    def __init__(self, csv_file,max_mfcc_len,n_mfcc=13):
        self.data=pd.read_csv(csv_file)
        self.dir_path=os.path.dirname(csv_file)
        self.vocab_dict=VOCAB_DICT
        self.n_mfcc=n_mfcc
        self.max_mfcc_len=max_mfcc_len
        self.max_seq_len=self.compute_max_seq_len()

    def __len__(self):
        return len(self.data)
    
    def compute_max_seq_len(self):
        self.data['length']=self.data['transcript'].apply(len)

        return max(self.data['length'])
    def char_to_idx(self, transcript):
        one_hot = torch.zeros(len(transcript), len(self.vocab_dict))
        for i, char in enumerate(transcript):
            one_hot[i, self.vocab_dict[char]] = 1
        return one_hot
    
    def compute_mfcc(self, audio_path):
        y, sr = librosa.load(audio_path)
        n_fft = min(2048, len(y))
        hop_length = n_fft // 4

        mfccs = librosa.feature.mfcc(y=y, sr=sr, 
                                     n_mfcc=self.n_mfcc, 
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
        return mfccs_combined
    
    @abstractmethod
    def __getitem__(self, idx):
        pass

class MultiViewTrainDataset(MultiViewDataset):
    def __init__(self,csv_file,loss_fn,max_mfcc_len,n_mfcc=13):
        super().__init__(csv_file, max_mfcc_len=max_mfcc_len,n_mfcc=n_mfcc)
        self.loss_fn=loss_fn
    
    def __getitem__(self,idx):
        
        audio_path_x1=self.data["audio_path"][idx]
        audio_path_x1=os.path.join(self.dir_path,str(audio_path_x1))
        transcript_c1=self.data["transcript"][idx]

        sample_list=ast.literal_eval(self.data["negative_samples"][idx])
        
        one_hot_c2=None
        if 0 in self.loss_fn or 1 in self.loss_fn:
            transcript_c2=sample_list[0]
            one_hot_c2=self.char_to_idx(transcript_c2)
            lev_distance=edit_distance(transcript_c1,transcript_c2)
        
        mfccs_x2=None
        if 2 in self.loss_fn or 3 in self.loss_fn:
            audio_path_x2=sample_list[1]
            audio_path_x2=os.path.join(self.dir_path,str(audio_path_x2))

            mfccs_x2=self.compute_mfcc(audio_path_x2)
        
        #computing mfcc
        mfcc_x1=self.compute_mfcc(audio_path_x1) 
        
        #computiing one hot for transcript 
        one_hot_c1=self.char_to_idx(transcript_c1)

        output_tensors= [torch.tensor(mfcc_x1), one_hot_c1]

        if 0 in self.loss_fn or 1 in self.loss_fn:
            output_tensors.append(one_hot_c2)
            output_tensors.append(lev_distance)
    
        if 2 in self.loss_fn or 3 in self.loss_fn:
            output_tensors.append(torch.tensor(mfccs_x2))
    
        return output_tensors

class MultiViewDevDataset(MultiViewDataset):
    def __init__(self,csv_file,max_mfcc_len,n_mfcc=13):
        super().__init__(csv_file,max_mfcc_len=max_mfcc_len,n_mfcc=n_mfcc)
    
    def __getitem__(self,idx):
        
        audio_path_x1=self.data["audio_path"][idx]
        audio_path_x1=os.path.join(self.dir_path,str(audio_path_x1))

        #mfcc 
        audio_mfcc=self.compute_mfcc(audio_path_x1)

        sample_dict=ast.literal_eval(self.data["sampled_words"][idx])
        lev_scores=[]
        for score in sample_dict.keys():
            for _ in range(len(sample_dict[score])):
                lev_scores.append(score)

        one_hot=[]
        for transcripts in sample_dict.values():
            for transcript in transcripts:
                one_hot.append(self.char_to_idx(transcript))
        
        transcript_c1=self.data['transcript'][idx]
        one_hot_c1=self.char_to_idx(transcript_c1)

        #sampling negative
        transcript_c2=ast.literal_eval(self.data["negative_samples"][idx])[0]
        one_hot_c2=self.char_to_idx(transcript_c2)
        audio_path_x2=ast.literal_eval(self.data["negative_samples"][idx])[1]
        audio_path_x2=os.path.join(self.dir_path,str(audio_path_x2))
        mfcc_x2=self.compute_mfcc(audio_path_x2)
        
        loss_lev_distance=edit_distance(transcript_c1,transcript_c2)

        output_tensor=[torch.tensor(audio_mfcc),one_hot,
                       torch.tensor(lev_scores),one_hot_c1,one_hot_c2,
                       torch.tensor(mfcc_x2),loss_lev_distance]

        return output_tensor

# colate functions 
def train_collate_fn(batch,max_mfcc_len,max_seq_len):

    mfccs_x1= []
    one_hot_c1=[]
    one_hot_c2=[]
    mfccs_x2=[]
    lev_distances=[]

    for item in batch:
        mfcc_x1,oh_c1 = item[0], item[1]
        mfccs_x1.append(mfcc_x1)
        one_hot_c1.append(oh_c1)

        if len(item) == 5:
            oh_c2,lev_distance,mfcc_x2= item[2], item[3],item[4]
            one_hot_c2.append(oh_c2)
            mfccs_x2.append(mfcc_x2)
            lev_distances.append(lev_distance)
        elif len(item)==4:
            if item[2].shape[1]==len(VOCAB_DICT):
                oh_c2,lev_distance=item[2],item[3]
                one_hot_c2.append(oh_c2)
                lev_distances.append(lev_distance)
        elif len(item) == 3:
            mfcc_x2=item[2]
            mfccs_x2.append(mfcc_x2)
    
    #max_mfcc_len = batch[0].dataset.max_mfcc_len

    mfccs_x1=pad_mfccs(mfccs_x1,max_mfcc_len)
    one_hot_c1=pad_sequence(one_hot_c1, max_seq_length=max_seq_len, batch_first=True)

    result={'view1_x1':mfccs_x1,'view2_c1':one_hot_c1}

    if one_hot_c2 and lev_distances:
        one_hot_c2=pad_sequence(one_hot_c2,max_seq_length=max_seq_len,batch_first=True)
        result['view2_c2']=one_hot_c2
        result['edit_distance']=lev_distances

    if mfccs_x2:
        mfccs_x2=pad_mfccs(mfccs_x2, max_mfcc_len)
        result['view1_x2']=mfccs_x2
    
    return result 
def dev_collate_fn(batch,max_mfcc_len,max_seq_len):
    
    mfccs_x1=[]
    one_hot=[]
    lev_scores=[]
    one_hot_c1=[]
    one_hot_c2=[]
    mfccs_x2=[]
    loss_lev_distances=[]

    for item in batch:
        mfcc_x1,oh,lev_score,oh_c1,oh_c2,mfcc_x2,loss_lev_dis=item[0],item[1],item[2],item[3],item[4],item[5],item[6]
        mfccs_x1.append(mfcc_x1)
        one_hot.append(oh)
        lev_scores.append(lev_score)
        one_hot_c1.append(oh_c1)
        one_hot_c2.append(oh_c2)
        mfccs_x2.append(mfcc_x2)
        loss_lev_distances.append(loss_lev_dis)

    
    #max_mfcc_len=batch[0].dataset.max_mfcc_len
    mfccs_x1=pad_mfccs(mfccs_x1,max_mfcc_len)
    mfccs_x2=pad_mfccs(mfccs_x2,max_mfcc_len)

    one_hot=pad_batch_sequence(one_hot,max_seq_length=max_seq_len)

    one_hot_c1=pad_sequence(one_hot_c1,max_seq_length=max_seq_len,batch_first=True)
    one_hot_c2=pad_sequence(one_hot_c2,max_seq_length=max_seq_len,batch_first=True)

    results={"mfcc":mfccs_x1,
             "sampled_one_hot":one_hot,
             "lev_scores":torch.stack(lev_scores),
             "ground_truth":one_hot_c1,
             "one_hot_c2":one_hot_c2,
             "mfcc_x2":mfccs_x2,
             "edit_distance":loss_lev_distances}

    return results

# Padding functions
def pad_mfccs(mfccs, max_len):
    padded_mfccs = []
    for mfcc in mfccs:
        # Padding to the right with zeros
        pad_width = max_len - mfcc.shape[1]
        padded_mfcc = torch.nn.functional.pad(mfcc, (0, pad_width), 'constant', 0)
        padded_mfccs.append(padded_mfcc)
    return torch.stack(padded_mfccs)

def pad_sequence(sequences,max_seq_length, batch_first=False, padding_value=0):
    padded_sequences = [torch.cat([seq, torch.zeros(max_seq_length - seq.size(0), seq.size(1), device=seq.device)], dim=0) for seq in sequences]

    return torch.nn.utils.rnn.pad_sequence(padded_sequences, batch_first=batch_first, padding_value=padding_value)

def pad_batch_sequence(batch, max_seq_length, padding_value=0):
    padded_batch = []

    max_num_sequences = max(len(sequences) for sequences in batch)
    
    for sequences in batch:
        padded_sequences = [
            torch.nn.functional.pad(torch.tensor(seq), 
                                    (0, 0, 0, max_seq_length - len(seq)), 
                                    'constant', padding_value)
            for seq in sequences
        ]
        
        padded_sequences = pad_sequence(padded_sequences, batch_first=True, padding_value=padding_value,max_seq_length=max_seq_length)
        
        if len(padded_sequences) < max_num_sequences:
            padding = torch.full((max_num_sequences - len(padded_sequences), 
                                  max_seq_length, 
                                  padded_sequences.shape[2]), 
                                  padding_value)
            padded_sequences = torch.cat((padded_sequences, padding), dim=0)
        
        padded_batch.append(padded_sequences)
    
    stacked_tensor = torch.stack(padded_batch)

    return stacked_tensor

# loaders
def get_train_loader(csv_file,batch_size,loss_fn,max_mfcc_len):

    dataset=MultiViewTrainDataset(csv_file=csv_file,loss_fn=loss_fn,max_mfcc_len=max_mfcc_len)

    loader=DataLoader(dataset, batch_size=batch_size, 
                      collate_fn=lambda batch: train_collate_fn(batch,dataset.max_mfcc_len,dataset.max_seq_len))

    return loader

def get_dev_loader(csv_file,batch_size,max_mfcc_len):
    
    dev_dataset=MultiViewDevDataset(csv_file=csv_file,max_mfcc_len=max_mfcc_len)

    dev_loader=DataLoader(dev_dataset, batch_size=batch_size, 
                          collate_fn=lambda batch: dev_collate_fn(batch, dev_dataset.max_mfcc_len,dev_dataset.max_seq_len))

    return dev_loader
    