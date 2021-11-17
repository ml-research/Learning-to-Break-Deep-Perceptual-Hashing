# Based on Code of Asuhariet Ygvar, added various modifications
#
# Copyright 2021 Asuhariet Ygvar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

import sys

sys.path.insert(0,'/code')

import argparse
import os
import pathlib
from os.path import isfile, join

import numpy as np
import pandas as pd
import torch
from models.neuralhash import NeuralHash
from PIL import Image
from tqdm import tqdm

from utils.hashing import compute_hash, load_hash_matrix


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Perform neural collision attack.')
    parser.add_argument('--source', dest='source', type=str,
                        default='data/imagenet_test', help='image folder to compute hashes for')
    args = parser.parse_args()

    # Load images
    if os.path.isfile(args.source):
        images = [args.source]
    elif os.path.isdir(args.source):
        datatypes = ['png', 'jpg', 'jpeg']
        images = [os.path.join(path, name) for path, subdirs, files in os.walk(
            args.source) for name in files]
    else:
        raise RuntimeError(f'{args.source} is neither a file nor a directory.')

    # Load pytorch model and hash matrix
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = NeuralHash()
    model.load_state_dict(torch.load('./models/model.pth'))
    model.to(device)
    seed = load_hash_matrix()
    seed = torch.tensor(seed).to(device)

    # Prepare results
    result_df = pd.DataFrame(columns=['image', 'hash_bin', 'hash_hex'])
    for img_name in tqdm(images):
        # Preprocess image
        try:
            img = Image.open(img_name).convert('RGB')
        except:
            continue
        img = img.resize([360, 360])
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])
        arr = torch.tensor(arr).to(device)

        # Compute hashes
        outputs_unmodified = model(arr)
        hash_bin = compute_hash(
            outputs_unmodified, seed, binary=True, as_string=True)
        hash_hex = compute_hash(
            outputs_unmodified, seed, binary=False)

        result_df = result_df.append(
            {'image': img_name, 'hash_bin': hash_bin, 'hash_hex': hash_hex}, ignore_index=True)
    os.makedirs('./dataset_hashes', exist_ok=True)
    if os.path.isfile(args.source):
        result_df.to_csv(f'./dataset_hashes/{args.source}_hashes.csv')
    elif os.path.isdir(args.source):
        path = pathlib.PurePath(args.source)
        result_df.to_csv(f'./dataset_hashes/{path.name}_hashes.csv')


if __name__ == '__main__':
    main()

