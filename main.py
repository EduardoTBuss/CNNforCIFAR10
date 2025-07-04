# main.py

import torch
from model import CNNModel
from data import load_data
from train import train_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    train_loader, test_loader = load_data()
    model = CNNModel().to(device)

    train_model(model, train_loader, test_loader, device)

if __name__ == '__main__':
    main()
