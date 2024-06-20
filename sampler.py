import Levenshtein
import pandas as pd 
import random 
import os 

def edit_distance(word1,word2):
    distance=Levenshtein.distance(word1,word2)
    return distance

def sample_negatives(path):

    df=pd.read_csv(path)
    negatives=[]

    for i in range(len(df)):
        word_negatives=[]
        for j in range(len(df)):
            if edit_distance(df["transcript"][i],df["transcript"][j])>2:
                word_negatives.append((df["audio_path"][j],df["transcript"][j]))
        negatives.append(word_negatives)

    df["negative_samples"]=negatives

    df.to_csv("sampled_metadata.csv")
    
    return df
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
    save_path=os.path.join(root_path,"sampled_devset.csv")

    df.to_csv(save_path)

    return df 

if __name__=='__main__':
    sample_negatives('/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/dataset/sampled_metadata.csv')