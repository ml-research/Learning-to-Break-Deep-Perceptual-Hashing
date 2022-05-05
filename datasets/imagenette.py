import hashlib
import os

from typing import Optional, Callable
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, check_integrity


class ImageNette(ImageFolder):
    """`ImageNette <https://github.com/fastai/imagenette>`_ Dataset
    Inspired by CIFAR10 class of TorchVision.
    """
    base_folder = 'imagenette2-320'
    url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
    filename = 'imagenette2-320.tgz'
    tgz_md5 = '3df6f0d01a2c9592104656642f5e78a3'
    extracted_folder_md5 = '52be8ad908708ed40b424b25d0d0e738'

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        self.root = root
        self.train = train

        if download:
            self.download()
        elif not self._check_integrity():
            raise RuntimeError(
                'Dataset not found or corrupted. You can use download=True to download it')

        image_folder_path = os.path.join(root, self.base_folder, 'train' if self.train else 'val')
        super(ImageNette, self).__init__(image_folder_path, transform=transform, target_transform=target_transform)


    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _check_integrity(self):
        # check the integrity of the zip
        if not check_integrity(os.path.join(self.root, self.filename)):
            return False

        # check the integrity of the extracted image folder
        if not self._check_folder_integrity(os.path.join(self.root, self.base_folder), self.extracted_folder_md5):
            return False

        return True

    def _check_folder_integrity(self, folder_name, md5):
        """
        Inspired by https://stackoverflow.com/a/24937710. Special thanks to
        Andy <https://stackoverflow.com/users/189134/andy>.
        """
        md5sum = hashlib.md5()
        if not os.path.exists(folder_name):
            return False

        for root, dirs, files in os.walk(folder_name):
            dirs.sort()
            for fnames in sorted(files):
                fpath = os.path.join(root, fnames)
                try:
                    f = open(fpath, 'rb')
                except:
                    # if the file cannot be opened just continue
                    f.close()
                    continue

                for chunk in iter(lambda: f.read(4096), b''):
                    md5sum.update(hashlib.md5(chunk).digest())
                f.close()

        return md5sum.hexdigest() == md5

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
