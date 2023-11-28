import pandas as pd
import numpy as np
import openai
from datasets import load_dataset
import argparse

def main(args):

    df = pd.read_json(args.input_path, orient="records")

    lyric_list = df['model_response'].to_list()

    embedding_map = dict()

    count = 0
    for id, input in enumerate(lyric_list):
        if count > args.max_num:
            break
        try:
            resp = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=[input]
                )
            embedding_map[id] = resp.data[0].embedding

            #embedding_map[id] = np.random.random(5).tolist()
        except Exception as e:
            print(f"Failed to embed song with id {id} due to error {e}")
        
        count += 1
    
    rows = pd.Series(embedding_map)
    out_df = rows.to_frame().reset_index()
    out_df.columns = ['question_id', 'embedding']
    out_df.to_parquet(args.output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--max-num', '-n', type=int, default=999999)

    args = parser.parse_args()

    main(args)
