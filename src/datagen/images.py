import torch
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
"""This is identical to  https://github.com/EmilienDupont/neural-processes/blob/master/datasets.py,
Many thanks to them.

"""

def mnist(path_to_data, batch_size=16, size=28):
    """MNIST dataloader.
    Parameters
    ----------
    batch_size : int
    size : int
        Size (height and width) of each image. Default is 28 for no resizing.
    path_to_data : string
        Path to MNIST data files.
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def celeba(path_to_data, batch_size=16, size=32, crop=89,
           shuffle=True):
    """CelebA dataloader.
    Parameters
    ----------
    batch_size : int
    size : int
        Size (height and width) of each image.
    crop : int
        Size of center crop. This crop happens *before* the resizing.
    path_to_data : string
        Path to CelebA data files.
    """
    transform = transforms.Compose([
        transforms.CenterCrop(crop),
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    celeba_data = CelebADataset(path_to_data,
                                transform=transform)
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=shuffle)
    return celeba_loader


class CelebADataset(Dataset):
    """CelebA dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        path_to_data : string
            Path to CelebA data files.
        subsample : int
            Only load every |subsample| number of images.
        transform : torchvision.transforms
            Torchvision transforms to be applied to each image.
        """
        self.img_paths = glob.glob(path_to_data + '/*.jpg')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = Image.open(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0