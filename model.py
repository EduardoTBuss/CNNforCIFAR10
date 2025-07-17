import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self ,num_classes = 10):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3 , 32 , kernel_size = 3 , padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32 , 32 , kernel_size = 3 , padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32 , 64 , kernel_size = 3 , padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64 , 64 , kernel_size = 3 , padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64 , 128 , kernel_size = 3 , padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128 , 128 , kernel_size = 3 , padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128 , 256 , kernel_size = 3 , padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256 , 256 , kernel_size = 3 , padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        
        self.classifiel = nn.sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2 , 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512 ,num_classes)
        )
  
    def forward(self , x):
      x = self.features(x)
      return self.classifier(x)
  