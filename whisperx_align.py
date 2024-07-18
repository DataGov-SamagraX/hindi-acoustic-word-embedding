import whisperx 
import pandas as pd 
import gc
import os 
from scipy.io.wavfile import write
from tqdm import tqdm 

DEVICE="cuda"
BATCH_SIZE=16
COMPUTE_TYPE="float16"


def align_whisperx(dataset_path,output_path):

    model=whisperx.load_model("whisper-tiny-ct2", DEVICE, compute_type=COMPUTE_TYPE)

    df=pd.read_csv(dataset_path)
    df=df[:10000]
    os.makedirs(output_path,exist_ok=True)
    root_dir=os.path.dirname(dataset_path)
    output_metadat_path=os.path.join(output_path,"metadata.csv")

    aligned_list=[]

    for i in tqdm(range(len(df))):

        audio_name=df['audio_name'][i]
        audio_path=os.path.join(root_dir,audio_name)

        audio=whisperx.load_audio(audio_path)

        result=model.transcribe(audio,batch_size=BATCH_SIZE)

        model_a, metadata = whisperx.load_align_model(language_code='hi', device=DEVICE)

        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)

        segments=result['segments'][0]['words']

        for i,seg in enumerate(segments):
            #print(i)
            #print(seg)
            word_label=seg['word']
            if 'start' in seg.keys():
                start_frame=int(seg['start']*16000)
                end_frame=int(seg['end']*16000)

                audio_seg=audio[start_frame:end_frame]

                save_path=os.path.join(output_path,f'{audio_name}_seg{i}.wav')
                write(save_path,16000,audio_seg)

                dict={'audio_path':f'{audio_name}_seg{i}.wav','transcript':word_label}

                aligned_list.append(dict)
    
    
    meta_df=pd.DataFrame(aligned_list)

    meta_df.to_csv(output_metadat_path)

    print("Aligned sucessfully")

if __name__=='__main__':

    dataset_path='/root/suyash/acoustic_stuff/hindi-acoustic-word-embedding/train_dataset/metadata.csv'
    output_path='/root/suyash/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_whisper'

    align_whisperx(dataset_path=dataset_path,output_path=output_path)











