import pandas as pd
import numpy as np
import openai
import multiprocessing
import jsonlines
import argparse
import requests
import time
import os

SYSTEM_PROMPT = "You are a helpful assistant that creates prompts for Large Language Model fine-tuning."
USER_PROMPT = """Below will be the lyrics of a song.  Consider the lyrics, themes, musical styles, and cultural aspects. Generate a one paragraph condensed prompt that requests that song in an active voice, specifying the artist style to mimic. For example: "Write me a song in the style of [ARTIST] that [insert song characteristics]..." """


def get_user_prompt(lyrics, artist):
    return f"{USER_PROMPT}\n\n[ARTIST]: {artist}\n\n[SONG]:\n{lyrics}\n\n[PROMPT]:\n"

def generate_using_openai(system_prompt, prompt, model_name, api_key):
    #return "its a pretty cool song imo"
    system = {
        "role": "system",
        "content": system_prompt
    }

    message = {
        "role": "user",
        "content": prompt
    }
    response = openai.ChatCompletion.create(
        model=model_name,  
        messages=[system, message],
        temperature=0.7,
        api_key=api_key,
        request_timeout=100,
    )
    return response.choices[0].message['content']

def get_process_name():
    return multiprocessing.current_process().name

def generate(system_prompt, prompt, model, api_key):

    while True:
        try:
            return generate_using_openai(system_prompt, prompt, model, api_key)
        except requests.RequestException as e:
            print(f"Error during generation from the text generation interface: {e}. Retrying in 10 seconds.")
            time.sleep(10)
        except openai.InvalidRequestError as e:
            print(f"WARNING, PROMPT SKIPPED: {e}")
            return False
        except openai.error.ServiceUnavailableError as e:
            print(f"WARNING ServiceUnavailableError: {e}. Retrying in 10 seconds.")
            time.sleep(10)
        except openai.error.APIError as e:
            print(f"WARNING {e}. Retrying in 10 seconds.")
            time.sleep(10)
        except openai.error.RateLimitError as e:
            pause = np.random.randint(5, 15)
            print(f"Error from OpenAI API: {e}. Retrying in {pause} seconds and rotating key.")
            time.sleep(pause)
        except Exception as e:
            print(f"WARNING: Unexpected exception in generate function {e}")
            return False

def pool_process(inputs):
    song_data, api_key, out_dir, model = inputs

    artist = song_data['artist']

    lyrics = song_data['lyrics']

    prompt = get_user_prompt(lyrics, artist)

    out_text = generate(SYSTEM_PROMPT, prompt, model, api_key)

    if not out_text:
        print(f"Unknown error generating output for song with id: {song_data['id']}")
        return False
    
    out_json = dict(id=int(song_data['id']), song=song_data['song'], artist=artist, lyrics=lyrics, prompt=out_text)

    with jsonlines.open(f"{out_dir}/temp-{get_process_name()}.jsonl", 'a') as writer:
        writer.write(out_json)

    print(f"Generated prompt for song id: {song_data['id']}")


def main(args):

    os.mkdir(args.out_dir)

    with open(args.key_file, 'r') as kf:
        keys = [key.replace('\n', '') for key in kf.readlines()]

    songs = pd.read_csv(args.data_path)

    def data_iter():
        max_num = args.num_data_points if args.num_data_points is not None else len(songs)
        count = 0
        for i in range(args.start_row, len(songs)):
            if count < max_num:
                row = songs.loc[i]
                count += 1

                yield row, keys[count % len(keys)], args.out_dir, args.model
            else:
                break
        print(f"PROCESSING {count} RESPONSES")
        return
        

    print(f"Spawning {args.num_processes} processes...")
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        pool.map(pool_process, data_iter(), chunksize=1)
    
    print(f"Combining Files")

    datas = []
    for f in os.listdir(args.out_dir):
        if os.path.splitext(f)[1] == ".jsonl" and f.startswith("temp-"):
            pth = os.path.join(args.out_dir, f)
            datas.append(pd.read_json(pth, lines=True))
        
    data = pd.concat(datas)
    data.to_json(f'{args.out_dir}/output.jsonl', orient="records", lines=True)

    print(f"Safe Cleanup")
    for f in os.listdir(args.out_dir):
        if os.path.splitext(f)[1] == ".jsonl" and f.startswith("temp-"):
            pth = os.path.join(args.out_dir, f)
            os.remove(pth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Prompts')
    parser.add_argument('--data-path', type=str, help='Data path', required=True)
    parser.add_argument('--start_row', type=int, default=0, help='Row number to start processing from (default: 0).')
    parser.add_argument('--num-processes', type=int, default=64, help='Number of parallel process to spawn')
    parser.add_argument('--num_data_points', '-n', type=int, help='Number of datapoints to process')
    parser.add_argument('--out-dir', type=str, help='Out data path', required=True)
    parser.add_argument('--key-file', type=str, help='OpenAI keys', required=True)
    parser.add_argument('--model', type=str, help='Which OpenAI model to use', default='gpt-4-1106-preview')

    args = parser.parse_args()
    
    main(args)