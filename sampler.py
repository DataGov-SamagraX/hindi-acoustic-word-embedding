import Levenshtein
import pandas as pd 

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

if __name__=='__main__':
    sample_negatives('/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/dataset/sampled_metadata.csv')