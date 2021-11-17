import argparse
import os
from os.path import isfile, join
from random import randint

import torch
import torchvision.transforms as T
from onnx import load_model
from skimage import feature
from skimage.color import rgb2gray
from tqdm import tqdm

from models.neuralhash import NeuralHash
from losses.mse_loss import mse_loss
from losses.quality_losses import ssim_loss
from utils.hashing import compute_hash, load_hash_matrix
from utils.image_processing import load_and_preprocess_img, save_images
from utils.logger import Logger
from metrics.hamming_distance import hamming_distance
import threading
import concurrent.futures
from itertools import repeat
import copy
import time


def optimization_thread(url_list, device, seed, loss_fkt, logger, args):
    # Store and reload source image to avoid image changes due to different formats
    id = randint(1, 10000000)
    temp_img = f'curr_image_{id}'
    model = NeuralHash()
    model.load_state_dict(torch.load('./models/model.pth'))
    model.to(device)
    while(url_list != []):
        img = url_list.pop(0)
        print('Thread working on ' + img)
        if args.optimize_original:
            resize = T.Resize((360, 360))
            source = load_and_preprocess_img(img, device, resize=False)
        else:
            source = load_and_preprocess_img(img, device, resize=True)
        input_file_name = img.rsplit(sep='/', maxsplit=1)[1].split('.')[0]
        if args.output_folder != '':
            save_images(source, args.output_folder, f'{input_file_name}')
        orig_image = source.clone()
        # Compute original hash
        with torch.no_grad():
            if args.optimize_original:
                outputs_unmodified = model(resize(source))
            else:
                outputs_unmodified = model(source)
            unmodified_hash_bin = compute_hash(
                outputs_unmodified, seed, binary=True)
            unmodified_hash_hex = compute_hash(
                outputs_unmodified, seed, binary=False)

        # Compute edge mask
        if args.edges_only:
            transform = T.Compose(
                [T.ToPILImage(), T.Grayscale(), T.ToTensor()])
            image_gray = transform(source.squeeze()).squeeze()
            image_gray = image_gray.cpu().numpy()
            edges = feature.canny(image_gray, sigma=3).astype(int)
            edge_mask = torch.from_numpy(edges).to(device)

        # Set up optimizer
        source.requires_grad = True
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                params=[source], lr=args.learning_rate)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params=[source], lr=args.learning_rate)
        else:
            raise RuntimeError(
                f'{args.optimizer} is no valid optimizer class. Please select --optimizer out of [Adam, SGD]')

        # Optimization cycle
        print(f'\nStart optimizing on {img}')
        for i in range(10000):
            with torch.no_grad():
                source.data = torch.clamp(source, min=-1, max=1)
            if args.optimize_original:
                outputs_source = model(resize(source))
            else:
                outputs_source = model(source)
            target_loss = - \
                loss_fkt(outputs_source, unmodified_hash_bin, seed)
            visual_loss = -ssim_loss(orig_image, source)
            optimizer.zero_grad()
            total_loss = target_loss + 0.99**i * args.ssim_weight * visual_loss
            total_loss.backward()
            if args.edges_only:
                optimizer.param_groups[0]['params'][0].grad *= edge_mask
            optimizer.step()

            # Check for hash changes
            if i % args.check_interval == 0:
                with torch.no_grad():
                    save_images(source, './temp', temp_img)
                    current_img = load_and_preprocess_img(
                        f'./temp/{temp_img}.png', device, resize=True)
                    check_output = model(current_img)
                    source_hash_hex = compute_hash(check_output, seed)
                    source_hash_bin = compute_hash(
                        check_output, seed, binary=True)

                    # Log results and finish if hash has changed
                    if source_hash_hex != unmodified_hash_hex:
                        if hamming_distance(source_hash_bin.unsqueeze(0), unmodified_hash_bin.unsqueeze(0)) >= args.hamming:
                            optimized_file = f'{args.output_folder}/{input_file_name}_opt'
                            if args.output_folder != '':
                                save_images(source, args.output_folder,
                                            f'{input_file_name}_opt')
                            # Compute metrics in the [0, 1] space
                            l2_distance = torch.norm(
                                ((current_img + 1) / 2) - ((orig_image + 1) / 2), p=2)
                            linf_distance = torch.norm(
                                ((current_img + 1) / 2) - ((orig_image + 1) / 2), p=float("inf"))
                            ssim_distance = ssim_loss(
                                (current_img + 1) / 2, (orig_image + 1) / 2)
                            print(
                                f'Finishing after {i+1} steps - L2 distance: {l2_distance:.4f} - L-Inf distance: {linf_distance:.4f} - SSIM: {ssim_distance:.4f}')

                            logger_data = [img, optimized_file + '.png', l2_distance.item(),
                                           linf_distance.item(), ssim_distance.item(), i+1, target_loss.item()]
                            logger.add_line(logger_data)
                            break
    os.remove(f'./temp/{temp_img}.png')


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
    parser.add_argument('--ssim_weight', dest='ssim_weight', default=5,
                        type=float, help='weight of ssim loss')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default='change_hash_attack', type=str, help='name of the experiment and logging file')
    parser.add_argument('--output_folder', dest='output_folder',
                        default='evasion_attack_outputs', type=str, help='folder to save optimized images in')
    parser.add_argument('--edges_only', dest='edges_only',
                        action='store_true', help='Change only pixels of edges')
    parser.add_argument('--optimize_original', dest='optimize_original',
                        action='store_true', help='Optimize resized image')
    parser.add_argument('--sample_limit', dest='sample_limit',
                        default=10000000, type=int, help='Maximum of images to be processed')
    parser.add_argument('--hamming', dest='hamming',
                        default=0.00001, type=float, help='Minimum Hamming distance to stop')
    parser.add_argument('--threads', dest='num_threads',
                        default=4, type=int, help='Number of parallel threads')
    parser.add_argument('--check_interval', dest='check_interval',
                        default=1, type=int, help='Hash change interval checking')
    args = parser.parse_args()

    # Create temp folder
    os.makedirs('./temp', exist_ok=True)

    # Load and prepare components
    start = time.time()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = load_hash_matrix()
    seed = torch.tensor(seed).to(device)

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
                      'l_inf', 'ssim', 'steps', 'target_loss', 'Hamming']
    logger = Logger(args.experiment_name, logging_header, output_dir='./logs')
    logger.add_line(['Hyperparameter', args.source, args.learning_rate,
                     args.optimizer, args.ssim_weight, args.edges_only, args.hamming])

    # define loss function
    loss_function = mse_loss

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

    # Start threads
    def thread_function(x): return optimization_thread(
        images, device, seed, loss_function, logger, args)
    threads_args = [(images, device, seed, loss_function,
                     logger, args) for i in range(args.num_threads)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        executor.map(thread_function, threads_args)

    logger.finish_logging()
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()
