import argparse
import math
import os
from os.path import isfile, join
from random import randint

import numpy as np
import torch
from onnx import load_model
from torchvision.transforms.functional import resize
from tqdm import tqdm

from models.neuralhash import NeuralHash
from losses.mse_loss import mse_loss
from losses.quality_losses import ssim_loss
from utils.hashing import compute_hash, load_hash_matrix
from utils.image_processing import load_and_preprocess_img, save_images
from utils.logger import Logger


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Perform neural collision attack.')
    parser.add_argument('--source', dest='source', type=str,
                        default='inputs/source.png', help='image to manipulate')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1.0,
                        type=float, help='step size of PGD optimization step')
    parser.add_argument('--optimizer', dest='optimizer', default='Adam',
                        type=str, help='kind of optimizer')
    parser.add_argument('--steps', dest='steps', default=15,
                        type=int, help='number of optimization steps per setting')
    parser.add_argument('--max_pixels', dest='max_pixels', default=150,
                        type=int, help='maximal number of pixels to modify')
    parser.add_argument('--optimize_resized', dest='opt_resized',
                        default=True, type=bool, help='optimize the resized 360x360 image')
    parser.add_argument('--ssim_weight', dest='ssim_weight', default=0,
                        type=float, help='weight of ssim loss')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default='change_hash_attack_few_pixels', type=str, help='name of the experiment and logging file')
    parser.add_argument('--output_folder', dest='output_folder',
                        default='few_pixels_attack_outputs', type=str, help='folder to save optimized images in')
    parser.add_argument('--sample_limit', dest='sample_limit',
                        default=1000000, type=int, help='Maximum of images to be processed')
    args = parser.parse_args()

    # Create temp folder
    os.makedirs('./temp', exist_ok=True)

    # Load model and source image
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = load_hash_matrix()
    seed = torch.tensor(seed).to(device)
    id = randint(1, 100000)
    temp_img = f'curr_image_{id}'
    model = NeuralHash()
    model.load_state_dict(torch.load('./models/model.pth'))
    model.to(device)

    # Prepare output folder
    try:
        os.mkdir(args.output_folder)
    except:
        if not os.listdir(args.output_folder):
            print(f'Folder {args.output_folder} already exists and is empty.')
        else:
            print(
                f'Folder {args.output_folder} already exists and is not empty.')

    # Prepare logging
    logging_header = ['file', 'optimized_file', 'l2',
                      'l_inf', 'ssim', 'steps', 'target_loss', 'num_pixels']
    logger = Logger(args.experiment_name, logging_header, output_dir='./logs')
    logger.add_line(['Hyperparameter', args.source, args.learning_rate,
                    args.optimizer, args.ssim_weight, args.steps, args.max_pixels])
    model.to(device)

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

    # define loss function
    loss_function = mse_loss

    # Start optimizing images
    for img in tqdm(images):
        # Store and reload source image to avoid image changes due to different formats
        source = load_and_preprocess_img(img, device)
        input_file_name = img.rsplit(sep='/', maxsplit=1)[1].split('.')[0]
        save_images(source, args.output_folder, f'{input_file_name}')
        source = load_and_preprocess_img(
            f'{args.output_folder}/{input_file_name}.png', device)
        orig_image = source.clone()

        # Compute original hash
        with torch.no_grad():
            outputs_unmodified = model(source)
            unmodified_hash_bin = compute_hash(
                outputs_unmodified, seed, binary=True)
            unmodified_hash_hex = compute_hash(
                outputs_unmodified, seed, binary=False)

        # Compute set of pixel locations and gradient mask
        pixel_locations = set()
        grad_mask = torch.zeros_like(source)
        for i in range(args.max_pixels):

            # Set up optimizer
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

            step_size_up = math.floor(args.steps/2)
            step_size_down = math.ceil(args.steps/2)

            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.5)

            # Compute pixels with the largest gradient (l1 norm)
            if args.opt_resized:
                outputs_source = model(source)
            else:
                outputs_source = model(resize(source, (360, 360)))
            target_loss = - \
                loss_function(outputs_source, unmodified_hash_bin, seed, c=1)
            total_loss = target_loss
            total_loss.backward()
            grad = source.grad

            # Identify pixels with largest gradient norm
            grad_norm = torch.norm(grad, p=1, dim=1)
            indices = grad_norm.flatten().topk(k=args.max_pixels)[1]
            topk_indices = np.array(np.unravel_index(
                indices.cpu().numpy(), grad_norm.shape)).T[:, 1:]
            for k in range(args.max_pixels):
                pixel_tuple = tuple(topk_indices[k])
                if pixel_tuple not in pixel_locations:
                    pixel_locations.add(pixel_tuple)
                    break

            # Compute new grad mask
            for x, y in pixel_locations:
                grad_mask[:, :, x, y] = 1

            print(f'optimizing {int(grad_mask.sum().cpu().numpy()/3)} pixels')
            for j in range(args.steps):
                with torch.no_grad():
                    source.data = torch.clamp(source.data, min=-1, max=1)
                source.requires_grad = True
                if args.opt_resized:
                    outputs_source = model(source)
                else:
                    outputs_source = model(resize(source, (360, 360)))
                target_loss = -mse_loss(outputs_source,
                                        unmodified_hash_bin, seed)
                visual_loss = -ssim_loss(orig_image, source)
                optimizer.zero_grad()
                total_loss = target_loss + args.ssim_weight * visual_loss
                total_loss.backward()
                optimizer.param_groups[0]['params'][0].grad *= grad_mask
                optimizer.step()
                scheduler.step()
                print(
                    f'Iteration {j+1}: \tTarget Loss {target_loss.detach():.4f}, Visual Loss {visual_loss.detach():.4f}')

            # Check for hash changes after optimizing pixels
            with torch.no_grad():
                save_images(source, './temp', temp_img)
                current_img = load_and_preprocess_img(
                    f'./temp/{temp_img}.png', device)
                check_output = model(current_img)
                source_hash_hex = compute_hash(check_output, seed)
                print(
                    f'Optimizing {i+1} pixels: Original Hash: {unmodified_hash_hex}, Current Hash: {source_hash_hex}')
                # Log results and finish if hash has changed
                if source_hash_hex != unmodified_hash_hex:
                    optimized_file = f'{args.output_folder}/{input_file_name}_opt'
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
                                   linf_distance.item(), ssim_distance.item(), (i*args.max_pixels) + j+1, target_loss.item(), i+1]
                    logger.add_line(logger_data)
                    break
            if source_hash_hex != unmodified_hash_hex:
                print(
                    f'Finishing optimization after {i+1} iterations and {int(grad_mask.sum().cpu().numpy()/3)} optimized pixels.')
                break
    logger.finish_logging()


if __name__ == "__main__":
    main()