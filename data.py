import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import IMG_SIZE, BATCH_SIZE
import os

YCbCr_MEAN = (0.5023, 0.5023, 0.5023)
YCbCr_STD = (0.25, 0.25, 0.25)

def rgb_to_ycbcr(img):
    return img.convert("YCbCr")

def load_data():
    transform_train = transforms.Compose([
        transforms.Lambda(rgb_to_ycbcr),  
        transforms.RandomCrop(IMG_SIZE, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
        transforms.RandAugment(num_ops=3, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(YCbCr_MEAN, YCbCr_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    transform_test = transforms.Compose([
        transforms.Lambda(rgb_to_ycbcr),
        transforms.ToTensor(),
        transforms.Normalize(YCbCr_MEAN, YCbCr_STD),
    ])

    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 2
    num_workers = max(1, cpu_count // 2)

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader
