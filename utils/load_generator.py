import sys
sys.path.insert(0,'/code/stylegan2-ada-pytorch')

import pickle

def load_generator(filepath):
    """Load pre-trained generator using the running average of the weights ('ema').
    Args:
        filepath (str): Path to .pkl file
    Returns:
        torch.nn.Module: G_ema from pickle
    """
    with open(filepath, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    return G