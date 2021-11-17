import torch


def hinge_loss(outputs, target_hash_bin, seed, distance=0):
    """
    Loss is based in the Hinge loss for binary classification.
    It uses the signs y in {-1, 1} of the targets and the model logit x for the source as inputs.
    Distance d specifies the minimal distance to the boundary one wants to achieve.
    By setting distance=0, only the signs are targeted to match.
    It then computes -min{0, d - x * y} as the loss function.
    """
    target_signs = torch.sign(target_hash_bin - 0.5)
    outputs = outputs.squeeze().unsqueeze(1)
    hash_outputs = torch.mm(seed, outputs).flatten()
    product = torch.mul(hash_outputs, target_signs)
    loss = torch.max(torch.zeros_like(target_signs), distance - product).mean()
    return loss
