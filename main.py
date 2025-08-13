import torch
from model import LightCNNImproved, train_model 
from data import load_data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    train_loader, test_loader = load_data()

    model = LightCNNImproved().to(device)

    train_model(model, train_loader, test_loader, device)

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
