import pandas as pd 
import librosa 
import IPython.display 
import os 
from tqdm import tqdm 
import torch 
import numpy as np
import json  
from scipy.io.wavfile import write

from data_preparation.model import Model 
from data_preparation.audio import load_audio,SAMPLE_RATE
from data_preparation.utils import Point,Segment
from tqdm import tqdm 

from data_preparation.alignment import compose_graph, backtrack, _load_model,force_align

root_path='/root/suyash/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_dataset'
train_csv='/root/suyash/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_dataset/sample_bhashini_train.csv'
dev_csv='/root/suyash/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_dataset/sample_bhashini_dev.csv'

df_1=pd.read_csv(os.path.join(root_path,train_csv))
df_2=pd.read_csv(os.path.join(root_path,dev_csv))
df_1=df_1.drop(columns=['Unnamed: 0.4', 'Unnamed: 0.3', 'Unnamed: 0.2', 'Unnamed: 0.1',
       'Unnamed: 0', 'level_0', 'index'])
df_2=df_2.drop(columns=['Unnamed: 0.3', 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'level_0',
       'index','sampled_words'])
df=pd.concat([df_1,df_2])
df=df.reset_index()
correct_df=df[df['transcript']==df['bhashini_transcript']]
duplet_idx_list=[]
for i in range(len(df)):
    if len(df['bhashini_transcript'][i].split(' '))==2:
        duplet_idx_list.append(i)

filtered_df=df.iloc[duplet_idx_list]
filtered_df=filtered_df.reset_index()

model=_load_model()

def compute_t_j(df):

    t_list=[]
    j_list=[]

    for i in tqdm(range(len(df))):
        audio_path=os.path.join(root_path,df['audio_path'][i])
        transcript=df['bhashini_transcript'][i]
        audio=load_audio(audio_path)
        token_ids=model.tokenize(transcript)
        #preprocessed_transcript=transcript.replace(" ","|")

        emission=model.inference(audio)
        graph=compose_graph(emission,token_ids)

        t=graph.size(0)-1
        j=graph.size(1)-1

        t_list.append(t)
        j_list.append(j)
    
    return t_list,j_list

t_list,j_list=compute_t_j(filtered_df)

filtered_df['token_length'],filtered_df['seq_length']=t_list,j_list

filtered_df.to_csv(os.path.join(root_path,'filtered_df.csv'))