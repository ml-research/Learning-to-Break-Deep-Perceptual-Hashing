import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from typing import Union


def load_hash_matrix(path='models/coreml_model/neuralhash_128x96_seed1.dat'):
    """
    Loads the output hash matrix multiplied with the network logits.
    """
    seed1 = open(path, 'rb').read()[128:]
    seed1 = np.frombuffer(seed1, dtype=np.float32)
    seed1 = seed1.reshape([96, 128])
    return seed1


def compute_hash(logits, seed=None, binary=False, as_string=False):
    """
    Computes the final hash based on the network logits.
    """
    if seed is None:
        seed = load_hash_matrix()
    if type(seed) is torch.Tensor and type(logits) is torch.Tensor:
        seed = seed.to(logits.device)
        outputs = logits.squeeze().unsqueeze(1)
        hash_output = torch.mm(seed, outputs).flatten()
    else:
        if type(logits) is torch.Tensor:
            logits = logits.detach().cpu().numpy()
        if type(seed) is torch.Tensor:
            seed = seed.cpu().numpy()
        hash_output = seed.dot(logits.flatten())
    hash_bits = ''.join(['1' if it >= 0 else '0' for it in hash_output])
    hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)
    if binary:
        if as_string:
            return hash_bits
        hash_bits = torch.tensor([int(b) for b in hash_bits])
        hash_bits = hash_bits.to(logits.device)
        return hash_bits
    else:
        return hash_hex


def get_hashes_of_dataset(dataset: Dataset, model: torch.nn.Module, seed: torch.Tensor,
                          device: Union[str, torch.device] = 'cuda:0', matmul: bool = True, binarization: bool = True,
                          batch_size: int = 64, num_workers: int = 8):
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False)
    with torch.no_grad():
        hashes = []
        hash_classes = []
        for x, y in tqdm(dataloader, desc='Getting Neural Hashes', leave=False):
            x = x.to(device)

            hash = model(x).squeeze().unsqueeze(2)
            if matmul:
                hash = torch.matmul(seed.repeat(len(x), 1, 1), hash)
            if binarization:
                hash = torch.sign(hash)

            hashes.append(hash.view(len(x), -1).cpu())
            hash_classes.append(y.cpu())
    hashes = torch.cat(hashes)
    hash_classes = torch.cat(hash_classes)

    hash_dict = {}
    for label in range(len(dataset.classes)):
        indices = (hash_classes == label)
        hash_dict[label] = hashes[indices]

    min_num_examples_in_class = min([len(x) for x in hash_dict.values()])
    generated_hashes = torch.stack(
        [hash_dict[i][:min_num_examples_in_class] for i in range(len(dataset.classes))])
    hash_target_classes = torch.tensor(
        [[i for _ in range(min_num_examples_in_class)] for i in range(len(dataset.classes))])

    return generated_hashes, hash_target_classes
