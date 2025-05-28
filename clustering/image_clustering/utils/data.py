from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import Imagenette, CIFAR10, CIFAR100
import glob
from pathlib import Path
import os
from PIL import Image
from typing import Callable, Optional

class ImageFolderDataset(Dataset):
    """
    A custom PyTorch Dataset class to load images from a folder and its subfolders.
    
    Args:
        root: The root directory containing your images
        transform: An optional transformation to apply to the images
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform

        # Collect all image paths recursively
        files = glob.glob(root+'/**/*', recursive=True)
        self.image_paths = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        self.image_paths.sort(key=lambda x: os.path.getmtime(x))
    
    def __len__(self) -> int:
        """
        Returns the number of images in the dataset
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        """
        Loads and returns an image at the given index
        """
        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert('RGB')  # Ensure RGB format
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image

    def get_image_paths(self) -> list:
        """
        Returns the list of all image paths in the dataset
        """
        return self.image_paths


def create_imagenette_dataloader(root_path: str, batch_size: int = 64, transform=None, shuffle: bool = False):
    """
    Creates a DataLoader for the Imagenette dataset with appropriate transformations.

    Args:
        root_path (str): The root directory containing your Imagenette images.
        batch_size (int): Number of samples per batch to load.
        shuffle (bool): If True, shuffles the data at every epoch.

    Returns:
        DataLoader: A PyTorch DataLoader object for iterating over the Imagenette dataset.
    """
    
    # Initialize the ImageFolderDataset with transformations
    dataset = Imagenette(root=root_path, transform=transform, split='train', size='320px', download=True)
    
    # Create a DataLoader from the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader



def create_cifar100_dataloader(root_path: str, batch_size: int = 64, transform=None, split='Train',shuffle: bool = False):
    """
    Creates a DataLoader for the CIFAR100 dataset with appropriate transformations.

    Args:
        root_path (str): The root directory containing your CIFAR100 images.
        batch_size (int): Number of samples per batch to load.
        shuffle (bool): If True, shuffles the data at every epoch.

    Returns:
        DataLoader: A PyTorch DataLoader object for iterating over the CIFAR100 dataset.
    """
    
    # Initialize the ImageFolderDataset with transformations
    dataset = CIFAR100(root=root_path, train=(split=='Train'), download=True, transform=transform)
    
    # Create a DataLoader from the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def create_cifar10_dataloader(root_path: str, batch_size: int = 64, transform=None, split='Train',shuffle: bool = False):
    """
    Creates a DataLoader for the CIFAR10 dataset with appropriate transformations.

    Args:
        root_path (str): The root directory containing your CIFAR10 images.
        batch_size (int): Number of samples per batch to load.
        shuffle (bool): If True, shuffles the data at every epoch.

    Returns:
        DataLoader: A PyTorch DataLoader object for iterating over the CIFAR10 dataset.
    """
    
    # Initialize the ImageFolderDataset with transformations
    dataset = CIFAR10(root=root_path, train=(split=='Train'), download=True, transform=transform)
    
    # Create a DataLoader from the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def create_dataloader_from_path(path: str, batch_size: int = 64, shuffle: bool = False,
                                transform: Optional[Callable] = None):
    """
    Creates a DataLoader from images found at the specified path.
    
    Args:
        path (str): The root directory containing your images
        batch_size (int): Number of samples per batch to load
        shuffle (bool): If True, shuffles the data at every epoch
        transform (Optional[Callable]): An optional transformation to apply to the images

    Returns:
        DataLoader: A PyTorch DataLoader object for iterating over the dataset.
    """
    
    # Initialize the custom ImageFolderDataset with transformations if provided
    dataset = ImageFolderDataset(root=path, transform=transform)
    
    # Create a DataLoader from the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def get_transforms(size=(224, 224)):
    """
    Returns a composed transformation pipeline including resizing,
    conversion to tensor, and normalization with ImageNet statistics.
    
    Returns:
        transforms.Compose: A transformation pipeline for image preprocessing.
    """
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])