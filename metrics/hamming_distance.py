import torch


def hamming_distance(x: torch.tensor, y: torch.tensor, normalize: bool = True):
    dist = torch.norm(x.float() - y.float(), p=1, dim=-1)
    if normalize:
        dist = dist / x.shape[-1]
    return dist
