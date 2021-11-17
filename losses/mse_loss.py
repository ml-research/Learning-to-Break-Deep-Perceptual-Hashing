import torch


def mse_loss(outputs, target_hash_bin, seed, c=5):
    outputs = outputs.squeeze().unsqueeze(1)
    hash_output = torch.mm(seed, outputs).flatten()
    hash_output = torch.nn.functional.normalize(hash_output, dim=0)
    hash_output = torch.sigmoid(hash_output * c)
    loss = torch.nn.functional.mse_loss(hash_output, target_hash_bin.float())
    return loss
