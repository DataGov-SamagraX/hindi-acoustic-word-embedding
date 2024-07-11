import pandas as pd 
from scipy.io.wavfile import write
from alignment import force_align
from audio import SAMPLE_RATE
from tqdm import tqdm 
import os 
def align_dataset(dataset_path,output_path):
    
    df=pd.read_csv(dataset_path)
    os.makedirs(output_path,exist_ok=True)
    parent_dir=os.path.dirname(dataset_path)
    output_metadat_path=os.path.join(output_path,"metadata.csv")
    aligned_list=[]
    
    for i in tqdm(range(len(df))):
        audio_name=df['audio_name'][i]
        transcript=df['transcript'][i]
        audio_path=os.path.join(parent_dir,audio_name)

        audio_segments=force_align(audio_path,transcript)
        #savinf audio segments is the key 
        for i in range(len(audio_segments)):
            word_label=audio_segments[i][0]
            duration=audio_segments[i][1]
            wave_form=audio_segments[i][2]

            segment_name=audio_name.replace('.wav',f'_seg_{i}.wav')
            aligned_list.append((segment_name,word_label,duration))
            seg_path=os.path.join(output_path,segment_name)

            write(seg_path,SAMPLE_RATE,wave_form)
    
    meta_df=pd.DataFrame(aligned_list,columns=["audio_path","transcrip","duration"])
    meta_df.to_csv(output_metadat_path,index=False, encoding='utf-8')

    print("Alignment complete")

if __name__=="__main__":
    dataset_path="/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/data_preparation/train_dataset/metadata.csv"
    output_path="train_aligned_dataset"

    align_dataset(dataset_path,output_path)

