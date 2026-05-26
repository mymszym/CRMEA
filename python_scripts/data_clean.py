import json
import argparse
import os
import random

def playground():
    train_data_file_path = 'data/DBP15K_DE_EN_V1/training_data=DBP15K_DE_EN_V1=zs=train-20=5==ptm-5-desc-sparql=pmm=gpt3.5=0=0=100.json'
    with open(train_data_file_path, 'r') as f:
        data = json.load(f)

    keys_to_remove = []
    for key,value in data.items():
        #remove null values
        if not key or not value or None in value:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del data[key]


    print('Data loaded successfully!')

def split_data(train_data_path, dev_data_path, raw_data_path, seed=42, ratio=0.8):
    print('Splitting train and dev data ... %s' % raw_data_path)

    with open(raw_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # split data randomly
    keys = list(data.keys())
    random.seed(seed)
    random.shuffle(keys)
    train_keys = keys[:int(ratio * len(keys))]
    dev_keys = keys[int(ratio * len(keys)):]
    train_data = {k: data[k] for k in train_keys}
    dev_data = {k: data[k] for k in dev_keys}

    # Save the split data
    with open(train_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(dev_data_path, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=4)

    print('Generated split data ... %s; %s' % (train_data_path, dev_data_path))


def main():
    parser = argparse.ArgumentParser(description='Process and clean data.')
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Path to the training data file')
    parser.add_argument('--dev_data_path', type=str, required=True,
                        help='Path to the development data file')
    parser.add_argument('--raw_data_path', type=str, required=True,
                        help='Path to the raw data file')
    parser.add_argument('--func', type=str, required=True,
                        help='Function to perform on the data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random number generator')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='Ratio of training data to total data')

    args = parser.parse_args()

    # Check which function to call based on the input
    if args.func == "split":
        split_data(args.train_data_path, args.dev_data_path, args.raw_data_path, args.seed, args.ratio)
    else:
        print(f"Function '{args.func}' not recognized.")

if __name__ == "__main__":
    # main()
    dataset_name = 'DBP15K_DE_EN_V1'
    raw_data_path = os.path.join(
        os.getcwd(), '..', 'output', 'training_data', dataset_name,
        'training_data_distribution=zs=train-20=rand-5==ptm-5-desc-sparql=mpnet-round-2=gpt3.5=0=5=0=2700.json')

    train_data_path = os.path.join(
        os.getcwd(), '..', 'output', 'training_data', dataset_name,
        'split_train_test=zs=train-20=rand-5==ptm-5-desc-sparql=mpnet-round-2=gpt3.5=0=5=0=2700.json')
    dev_data_path = os.path.join(
        os.getcwd(), '..', 'output', 'training_data', dataset_name,
        'split_dev_test=zs=train-20=rand-5==ptm-5-desc-sparql=mpnet-round-2=gpt3.5=0=5=0=2700.json')
    seed = 42
    ratio = 0.8
    split_data(train_data_path, dev_data_path, raw_data_path, seed, ratio)
