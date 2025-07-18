import torchvision
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import IMG_SIZE, BATCH_SIZE

def load_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(   
        root='./data', train = True, download = True, transform = transform_train)

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train = False, download = True, transform = transform_test)

    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = os.cpu_count() // 2)
    test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = os.cpu_count() // 2)

    return train_loader, test_loader
