import argparse
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0x66ccff)
    
    args = parser.parse_args()
    return args


def main(args):
    files = os.listdir(args.path)
    # Shuffle files
    random.seed(args.seed)
    random.shuffle(files)
    # Split files
    split = int(len(files) * args.ratio)
    train_files = files[:split]
    test_files = files[split:]
    # Write files
    with open('train.txt', 'w') as f:
        f.write('\n'.join(train_files))
    with open('test.txt', 'w') as f:
        f.write('\n'.join(test_files))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    