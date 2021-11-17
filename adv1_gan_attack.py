import argparse
import csv
import os
from random import randint

import torch

from models.neuralhash import NeuralHash
from losses.hinge_loss import hinge_loss
from utils.hashing import compute_hash, load_hash_matrix
from utils.image_processing import load_and_preprocess_img, save_images
from utils.load_generator import load_generator
import torchvision

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Perform neural collision attack.')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-2,
                        type=float, help='step size of PGD optimization step')
    parser.add_argument('--optimizer', dest='optimizer', default='Adam',
                        type=str, help='kind of optimizer')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default='preimage_GAN_attack', type=str, help='name of the experiment and logging file')
    parser.add_argument('--output_folder', dest='output_folder',
                        default='gan_attack_outputs', type=str, help='folder to save optimized images in')
    parser.add_argument('--target_hashset', dest='target_hashset',
                        type=str, help='Target hashset csv file path')
    parser.add_argument('--pkl_file', dest='pkl_file', type=str, help='StyleGAN2 weights used to generate images')
    parser.add_argument('--threads', dest='num_threads',
                        default=4, type=int, help='Number of parallel threads')
    parser.add_argument('--check_interval', dest='check_interval',
                        default=10, type=int, help='Hash change interval checking')
    args = parser.parse_args()

    # Load StyleGan-2-Ada generator
    generator = load_generator(args.pkl_file)
    generator.eval()

    # Load and prepare components
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = load_hash_matrix()
    seed = torch.tensor(seed).to(device)
    id = randint(1, 1000000)
    temp_img = f'curr_image_{id}'
    model = NeuralHash()
    model.load_state_dict(torch.load('./models/model.pth'))
    model.to(device)

    # Read target hashset
    target_hash_dict = dict()
    bin_hex_hash_dict = dict()
    target_hashes = []
    with open(args.target_hashset, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip header
        for row in reader:
            hash = [int(b) for b in list(row[2])]
            hash = torch.tensor(hash).unsqueeze(0).to(device)
            target_hash_dict[row[3]] = [row[2], row[1]]
            target_hash = torch.tensor(
                [int(b) for b in list(row[2])]).unsqueeze(0)
            bin_hex_hash_dict[row[2]] = row[3]
            target_hashes.append(target_hash)
    target_hashes = torch.cat(target_hashes, dim=0).to(device)

    # Prepare output folder
    if args.output_folder != '':
        try:
            os.mkdir(args.output_folder)
        except:
            if not os.listdir(args.output_folder):
                print(
                    f'Folder {args.output_folder} already exists and is empty.')
            else:
                print(
                    f'Folder {args.output_folder} already exists and is not empty.')

    # define loss function
    resize = torchvision.transforms.Resize((360, 360))

    for sample in range(10):

        z = torch.randn(1, generator.z_dim).to(device)
        c = None
        w = generator.mapping(z, c, truncation_psi=0.0, truncation_cutoff=0)
        w.requires_grad = True
        optimizer = torch.optim.Adam(params=[w], lr=args.learning_rate, betas=(0.1, 0.1))
        target_hash = target_hashes[0]
        target_hash_str = ''.join(
                    ['1' if it > 0 else '0' for it in target_hash.tolist()])
        target_hash_hex = bin_hex_hash_dict[target_hash_str]
        for i in range(1000):
            img = generator.synthesis(w, noise_mode='const', force_fp32=True)
            img_resized = resize(img)
            outputs = model(img_resized)
            target_loss = hinge_loss(outputs, target_hash, seed)
            optimizer.zero_grad()
            target_loss.backward()
            optimizer.step()
            # Check for hash changes
            if i % args.check_interval == 0:
                with torch.no_grad():
                    save_images(img_resized, './temp', temp_img)
                    current_img = load_and_preprocess_img(
                        f'./temp/{temp_img}.png', device)
                    check_output = model(current_img)
                    source_hash_hex = compute_hash(check_output, seed)
                    print(
                        f'Iteration {i+1}: \tTarget Loss {target_loss.detach():.4f}')
                    print(
                        f'\t\tOriginal Hash: {target_hash_hex}, Current Hash: {source_hash_hex}')

                    # Log results and finish if hash has changed
                    if source_hash_hex == target_hash_hex:
                        if args.output_folder != '':
                            save_images(current_img, args.output_folder,
                                        f'gan_opt_{sample}')
                        print(
                            f'Finishing after {i+1} steps')
                        break


if __name__ == "__main__":
    main()
