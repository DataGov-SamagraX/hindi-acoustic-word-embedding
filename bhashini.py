import pandas as pd
import asyncio
import aiohttp
import os
import json
import time

root_path = '/root/suyash/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_dataset/'
df = pd.read_csv(os.path.join(root_path,'left_out_audios.csv'))
batch_size = 500
output_dir = '/root/suyash/acoustic_stuff/hindi-acoustic-word-embedding/bhashini_transcripts'

os.makedirs(output_dir, exist_ok=True)

def get_batches(df, batch_size):
    num_batches = len(df) // batch_size + int(len(df) % batch_size != 0)
    for i in range(num_batches):
        yield df[i * batch_size:(i + 1) * batch_size]

async def post_audio_file(session, audio_path):
    url = 'https://ai-tools.dev.bhasai.samagra.io/asr/bhashini_nisai/'
    data = {
        'text_read': '[]',
        'match_status': '[]',
        'fuzz_match': 'false',
        'distance_cutoff': '1',
        'input_lang': 'hi'
    }

    with open(audio_path, 'rb') as f:
        form = aiohttp.FormData()
        form.add_field('file', f, filename=os.path.basename(audio_path))
        for key, value in data.items():
            form.add_field(key, value)

        async with session.post(url, data=form) as response:
            if response.status == 200 and response.content_type == 'application/json':
                result = await response.json()
                return {'file': os.path.basename(audio_path), 'transcription': result.get('transcription', '')}
            else:
                error_text = await response.text()
                return {'file': os.path.basename(audio_path), 'error': error_text, 'status': response.status}

async def get_tasks(session, batch_df):
    tasks = []
    for i in range(len(batch_df)):
        audio_path = os.path.join(root_path, batch_df.iloc[i]['audio_path'])
        task = asyncio.create_task(post_audio_file(session, audio_path))
        tasks.append(task)
    return tasks

async def process_batch(batch_df, batch_number):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = await get_tasks(session, batch_df)
        responses = await asyncio.gather(*tasks)

        for response in responses:
            if 'error' in response:
                print(f"Error processing file '{response['file']}': {response['error']}")
                return

        results.extend(responses)

    json_str = json.dumps(results, indent=4)

    json_file_path = os.path.join(output_dir, f'batch_{batch_number}.json')
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_str)

    print(f"Batch {batch_number} processed and saved.")

async def process_all_batches(df, batch_size):
    batches = get_batches(df, batch_size)
    batch_number = 0
    for batch_df in batches:
        await process_batch(batch_df, batch_number)
        batch_number += 1

start = time.time()
asyncio.run(process_all_batches(df, batch_size))
end = time.time()

print(f"Total processing time: {end - start} seconds")

