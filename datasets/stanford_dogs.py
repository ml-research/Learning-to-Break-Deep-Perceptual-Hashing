from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from typing import Callable, Optional
import hashlib
import os
import scipy.io as sio
import numpy as np


class StanfordDogs(ImageFolder):
    """`Stanford Dogs Dataset <http://vision.stanford.edu/aditya86/ImageNetDogs/main.html>`_ Dataset.
    This class is based on the CIFAR10-Class from torchvision
    """
    base_folder = 'stanford_dogs'
    url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    filename = "images.tar"
    tar_md5 = '1bb1f2a596ae7057f99d7d75860002ef'
    image_folder_md5 = 'c24ab0237efb75419885481a3b2dcacf'
    split_file_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"
    split_filename = "lists.tar"
    split_tar_md5 = 'edbb9f16854ec66506b5f09b583e0656'
    split_files = {
        'train': ['train_list.mat', 'd37f459eacccfa4d299373dffba9648d'],
        'test': ['test_list.mat', '66f60c285efbc3ce2fb7893bd26c6b80']
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        all_data: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ) -> None:
        self.base_folder = os.path.join(root, self.base_folder)
        self.train = train

        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # load the images as ImageFolder
        super(StanfordDogs, self).__init__(
            os.path.join(self.base_folder, "Images"), transform=transform, target_transform=target_transform
        )

        # load the split file
        split_fname, checksum = self.split_files['train' if self.train else 'test']
        loaded_split_mat = sio.loadmat(
            os.path.join(self.base_folder, split_fname))
        self.file_list = [
            os.path.join(self.base_folder, "Images", item[0]) for item in loaded_split_mat['file_list'].flatten()
        ]

        # filter the imgs, the samples, and the targets to contain only the files in the file list
        file_mask = []
        for img_path in np.array(self.imgs)[:, 0]:
            if img_path in self.file_list or all_data:
                file_mask.append(1)
            else:
                file_mask.append(0)

        sample_indices = np.where(np.array(file_mask) == 1)
        self.imgs = np.array(self.imgs)[sample_indices].tolist()
        self.samples = np.array(self.samples)[sample_indices].tolist()
        self.targets = np.array(self.targets)[sample_indices].tolist()

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _check_integrity(self):
        # check the integrity of the image tar
        if not check_integrity(os.path.join(self.base_folder, self.filename)):
            return False
        # check the integrity of the lists tar
        if not check_integrity(os.path.join(self.base_folder, self.split_filename)):
            return False

        # check the integrity of the extracted image folder
        if not self._check_folder_integrity(os.path.join(self.base_folder, "Images"), self.image_folder_md5):
            return False

        # check the integrity of the train list
        split_file_name, split_file_md5 = self.split_files['train' if self.train else 'test']
        if not check_integrity(os.path.join(self.base_folder, split_file_name), split_file_md5):
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
        download_and_extract_archive(
            self.url, self.base_folder, filename=self.filename, md5=self.tar_md5)
        download_and_extract_archive(
            self.split_file_url, self.base_folder, filename=self.split_filename, md5=self.split_tar_md5
        )
