from .gan import GAN
from .cgan import CGAN

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms 

__all__ = [
    'GAN',
    'CGAN',
    'load_mnist'
]

def load_mnist(path, batch_size):
    return DataLoader(
        MNIST(
            path, 
            download = True, 
            transform = transforms.ToTensor()
        ),
        batch_size = batch_size,
        shuffle=True
    )