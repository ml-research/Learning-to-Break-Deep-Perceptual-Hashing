import os

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image


def load_and_preprocess_img(img_path, device, resize=True, normalize=True):
    image = Image.open(img_path).convert('RGB')
    if resize:
        image = image.resize([360, 360])
    arr = np.array(image).astype(np.float32) / 255.0
    if normalize:
        arr = arr * 2.0 - 1.0
    arr = arr.transpose(2, 0, 1)
    tensor = torch.tensor(arr).unsqueeze(0)
    tensor = tensor.to(device)
    return tensor


def save_images(img: torch.tensor, folder, filename):
    img = img.detach().cpu()
    img = (img + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)
    img = img.squeeze()
    path = os.path.join(folder, f'{filename}.png')
    save_image(img, path)
