import argparse
import pandas as pd


def main(args):

    data = pd.read_json(args.data_path, lines=True)
    data['conversations'] = pd.Series(list(zip(data['prompt'], data['lyrics']))).map(lambda t: [{'from': 'human', 'value': t[0]}, {'from': 'gpt', 'value': t[1]}])
    data[['id', 'conversations']].to_json(args.out_path, orient='records', indent=1)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Prompts')
    parser.add_argument('--data-path', type=str, help='Data path', required=True)
    parser.add_argument('--out-path', type=str, help='Out data path', required=True)


    args = parser.parse_args()
    
    main(args)