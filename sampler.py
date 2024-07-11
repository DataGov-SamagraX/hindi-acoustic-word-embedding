import Levenshtein
import pandas as pd 
import random 
import os 
from tqdm import tqdm

def edit_distance(word1,word2):
    distance=Levenshtein.distance(word1,word2)
    return distance

def sample_negative_transcript(curr_transcript,all_transcripts,audio_dict):
    other_transcripts = [t for t in all_transcripts if t != curr_transcript]
    new_transcript = random.choice(other_transcripts)
    audio_path=random.choice(audio_dict[new_transcript])
    return (new_transcript,audio_path)


def sample_negative(csv_file):
    if isinstance(csv_file,str):
        df=pd.read_csv(csv_file)
    else:
        df=csv_file
    
    transcripts = df['transcript'].unique().tolist()
    audio_dict = df.groupby('transcript')['audio_path'].apply(list).to_dict()
    curr_transcripts=df['transcript'].tolist()

    negatives=[sample_negative_transcript(cur_transcript,transcripts,audio_dict) for cur_transcript in curr_transcripts]

    df['negative_samples']=negatives

    df.to_csv(csv_file)

    print("Sampling Complete")

def sample_words_with_lev_scores(df, lev_score, word):
    filtered_words = df[df.apply(lambda row: edit_distance(row['transcript'], word), axis=1) == lev_score]
    
    if len(filtered_words) >= 2:
        sampled_words = random.sample(list(filtered_words['transcript']), 2)
    else:
        sampled_words = list(filtered_words['transcript'])
    
    return sampled_words

def sample_dev_words(path):
    df=pd.read_csv(path)
    sampled_list=[]

    for i in tqdm(range(len(df))):
        lev_dict={}
        for j in range(len(df)):
            lev_distance=edit_distance(df['transcript'][i],df['transcript'][j])
            if lev_distance in lev_dict:
                lev_dict[lev_distance].append(df['transcript'][j])
            else:
                lev_dict[lev_distance]=[df['transcript'][j]]
        
        sampled_list.append(lev_dict)
    
    output_list=[]
    for i in tqdm(range(len(sampled_list))):
        lev_dict=sampled_list[i]
        out_dict={}
        element_count=0
        for lev_score in [0,1,2]:
            if lev_score in lev_dict.keys():
                if len(lev_dict[lev_score])>2:
                    out_dict[lev_score]=lev_dict[lev_score][:2]
                else:
                    out_dict[lev_score]=lev_dict[lev_score]
                element_count+=len(out_dict[lev_score])
            else:
                continue
        
        #out_dict={0:[],1:[],2:[]}
        scores=[score for score in lev_dict.keys()]
        lev_score=3
        while element_count<20:
            if lev_score not in scores:
                lev_score += 1
                continue
            if lev_score in scores:
               words_to_add = lev_dict[lev_score][:min(20- element_count, len(lev_dict[lev_score]))]
               if lev_score not in out_dict:
                   out_dict[lev_score]=[]
               out_dict[lev_score]=words_to_add
               element_count+= len(words_to_add) 
            lev_score+=1
        
        output_list.append(out_dict)
                
    df['sampled_words']=output_list

    root_path=os.path.dirname(path)
    save_path=os.path.join(root_path,"sampled_metadata.csv")

    df.to_csv(path)
    print("dev_sampling complete")

    return save_path

def calculate_lev_scores(df):
    lev_scores = {}  
    
    for i in tqdm(range(len(df))):
        word_i = df['transcript'][i]
        
        for j in range(len(df)):
            word_j = df['transcript'][j]
            lev_distance = edit_distance(word_i, word_j)
                
            if lev_distance not in lev_scores:
                    lev_scores[lev_distance] = []
            lev_scores[lev_distance].append(word_j)
    
    return lev_scores

def sample_dev_words_optimized(path):
    df = pd.read_csv(path)
    
    lev_scores = calculate_lev_scores(df)
    
    sampled_words = []
    total_word_count = 0
    
    for lev_score in [0, 1, 2]:  
        if len(lev_score)>2:
            sampled_words.extend(words_to_add)
    
    lev_score = 3
    while total_word_count < 20:
        if lev_score not in lev_scores:
            lev_score += 1
            continue
        
        if lev_score in lev_scores:
            words_to_add = lev_scores[lev_score][:min(20 - total_word_count, len(lev_scores[lev_score]))]
            sampled_words.extend(words_to_add)
            total_word_count += len(words_to_add)
        lev_score += 1
        
    
    sampled_df = pd.DataFrame({'sampled_words': sampled_words})
    root_path = path.split('dataset/')[0] + 'dataset/'
    save_path = os.path.join(root_path, "sampled_devset.csv")
    sampled_df.to_csv(save_path, index=False)
    print("Dev sampling complete")

    return sampled_df

def sample_dev_words_initial(path):
    df=pd.read_csv(path)
    sampled_list=[]
    # for now lets sample words for every other words in the dev dataset 

    for i in range(len(df)):
        out_dict={}
        for j in range(len(df)):
            lev_distance=edit_distance(df['transcript'][i],df['transcript'][j])
            if lev_distance in out_dict:
                out_dict[lev_distance].append(df['transcript'][j])
            else:
                out_dict[lev_distance]=[df['transcript'][j]]
        
        sampled_list.append(out_dict)

    """for i in range(len(df)):
        score=0
        out_dict={}
        count_list=[]
        while len(count_list)<8:
            lev_score_i_words = sample_words_with_lev_scores(df, score, df['transcript'][i])
            out_dict[score]=lev_score_i_words
            
            count_list+=lev_score_i_words
            score+=1

        sampled_list.append(out_dict)"""

    df['sampled_words']=sampled_list

    root_path=path.split('dataset/')[0] + 'dataset/'
    save_path=os.path.join(root_path,"sampled_devset(2).csv")

    df.to_csv(save_path)

    return df 

if __name__=='__main__':
    #sample_negatives_optimized('/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_dataset/train_reduced_data.csv')
    #sample_dev_words('/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_dataset/reduced_dev_data.csv')
    sample_negative('/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_dataset/sample_bhashini_train.csv')
    sample_dev_words('/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_dataset/sample_bhashini_dev.csv')
    sample_negative('/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_dataset/sample_bhashini_dev.csv')



