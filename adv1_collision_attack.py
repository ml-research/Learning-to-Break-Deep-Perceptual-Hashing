import argparse
import concurrent.futures
import csv
import os
import threading
from os.path import isfile, join
from random import randint

import torch
import torchvision.transforms as T
from skimage import feature

from models.neuralhash import NeuralHash
from losses.hinge_loss import hinge_loss
from losses.quality_losses import ssim_loss
from utils.hashing import compute_hash, load_hash_matrix
from utils.image_processing import load_and_preprocess_img, save_images
from utils.logger import Logger
import threading

def optimization_thread(url_list, device, seed, loss_fkt, logger, args, target_hashes, bin_hex_hash_dict, target_hash_dict):
    print(f'Process {threading.get_ident()} started')

    id = randint(1, 1000000)
    temp_img = f'curr_image_{id}'
    model = NeuralHash()
    model.load_state_dict(torch.load('./models/model.pth'))
    model.to(device)

    # Start optimizing images
    while(url_list != []):
        img = url_list.pop(0)
        # Store and reload source image to avoid image changes due to different formats
        source = load_and_preprocess_img(img, device)
        input_file_name = img.rsplit(sep='/', maxsplit=1)[1].split('.')[0]
        if args.output_folder != '':
            save_images(source, args.output_folder, f'{input_file_name}')
        source_orig = source.clone()
        # Compute original hash and hamming distances
        with torch.no_grad():
            outputs_unmodified = model(source)
            unmodified_hash_bin = compute_hash(
                outputs_unmodified, seed, binary=True)
            hamming_dist = torch.norm(
                unmodified_hash_bin - target_hashes.float(), p=1, dim=1) / unmodified_hash_bin.shape[0]
            _, idx = torch.min(hamming_dist, dim=0)
            target_hash = target_hashes[idx.item()]
            target_hash_str = ''.join(
                ['1' if it > 0 else '0' for it in target_hash.tolist()])
            target_hash_hex = bin_hex_hash_dict[target_hash_str]
            if args.edges_only:
                # Compute edge mask
                transform = T.Compose(
                    [T.ToPILImage(), T.Grayscale(), T.ToTensor()])
                image_gray = transform(source.squeeze()).squeeze()
                image_gray = image_gray.cpu().numpy()
                edges = feature.canny(image_gray, sigma=3).astype(int)
                edge_mask = torch.from_numpy(edges).to(device)

        # Apply attack
        source.requires_grad = True
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                params=[source], lr=args.learning_rate)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                params=[source], lr=args.learning_rate)
        else:
            raise RuntimeError(
                f'{args.optimizer} is no valid optimizer class. Please select --optimizer out of [Adam, SGD]')
        for i in range(10000):
            outputs_source = model(source)[0]
            target_loss = loss_fkt(
                outputs_source, target_hash, seed)
            visual_loss = -1 * args.ssim_weight * \
                ssim_loss(source_orig, source)
            optimizer.zero_grad()
            total_loss = target_loss + visual_loss
            total_loss.backward()
            if args.edges_only:
                optimizer.param_groups[0]['params'][0].grad *= edge_mask
            optimizer.step()
            with torch.no_grad():
                source.data = torch.clamp(source.data, min=-1, max=1)

            # Check for hash changes
            if i % args.check_interval == 0:
                with torch.no_grad():
                    save_images(source, './temp', temp_img)
                    current_img = load_and_preprocess_img(
                        f'./temp/{temp_img}.png', device)
                    check_output = model(current_img)
                    source_hash_hex = compute_hash(check_output, seed)

                    # Log results and finish if hash has changed
                    if source_hash_hex == target_hash_hex:
                        optimized_file = f'{args.output_folder}/{input_file_name}_opt'
                        if args.output_folder != '':
                            save_images(source, args.output_folder,
                                        f'{input_file_name}_opt')
                        # Compute metrics in the [0, 1] space
                        l2_distance = torch.norm(
                            ((current_img + 1) / 2) - ((source_orig + 1) / 2), p=2)
                        linf_distance = torch.norm(
                            ((current_img + 1) / 2) - ((source_orig + 1) / 2), p=float("inf"))
                        ssim_distance = ssim_loss(
                            (current_img + 1) / 2, (source_orig + 1) / 2)
                        print(
                            f'Finishing after {i+1} steps - L2 distance: {l2_distance:.4f} - L-Inf distance: {linf_distance:.4f} - SSIM: {ssim_distance:.4f}')

                        logger_data = [img, optimized_file + '.png', l2_distance.item(),
                                       linf_distance.item(), ssim_distance.item(), i+1, target_hash_dict[target_hash_hex][1]]
                        logger.add_line(logger_data)
                        break


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Perform neural collision attack.')
    parser.add_argument('--source', dest='source', type=str,
                        default='inputs/source.png', help='image to manipulate')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-3,
                        type=float, help='step size of PGD optimization step')
    parser.add_argument('--optimizer', dest='optimizer', default='Adam',
                        type=str, help='kind of optimizer')
    parser.add_argument('--ssim_weight', dest='ssim_weight', default=100,
                        type=float, help='weight of ssim loss')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default='preimage_attack', type=str, help='name of the experiment and logging file')
    parser.add_argument('--output_folder', dest='output_folder',
                        default='collision_attack_outputs', type=str, help='folder to save optimized images in')
    parser.add_argument('--target_hashset', dest='target_hashset',
                        type=str, help='Target hashset csv file path')
    parser.add_argument('--edges_only', dest='edges_only',
                        action='store_true', help='Change only pixels of edges')
    parser.add_argument('--sample_limit', dest='sample_limit',
                        default=10000000, type=int, help='Maximum of images to be processed')
    parser.add_argument('--threads', dest='num_threads',
                        default=4, type=int, help='Number of parallel threads')
    parser.add_argument('--check_interval', dest='check_interval',
                        default=10, type=int, help='Hash change interval checking')
    args = parser.parse_args()

    # Load and prepare components
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = load_hash_matrix()
    seed = torch.tensor(seed).to(device)

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

    # Prepare logging
    logging_header = ['file', 'optimized_file', 'l2',
                      'l_inf', 'ssim', 'steps']
    logger = Logger(args.experiment_name, logging_header, output_dir='./logs')
    logger.add_line(['Hyperparameter', args.source, args.learning_rate,
                     args.optimizer, args.ssim_weight, args.edges_only])

    # define loss function
    loss_function = hinge_loss

    # Load images
    if os.path.isfile(args.source):
        images = [args.source]
    elif os.path.isdir(args.source):
        images = [join(args.source, f) for f in os.listdir(
            args.source) if isfile(join(args.source, f))]
        images = sorted(images)
    else:
        raise RuntimeError(f'{args.source} is neither a file nor a directory.')
    images = images[:args.sample_limit]

    threads_args = (images, device, seed, loss_function,
                     logger, args, target_hashes, bin_hex_hash_dict, target_hash_dict)

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        for t in range(args.num_threads):
            executor.submit(lambda p: optimization_thread(*p), threads_args)

    logger.finish_logging()


if __name__ == "__main__":
    os.makedirs('./temp', exist_ok=True)
    main()
