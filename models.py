import torch
import torch.nn as nn
    
class DASNet(nn.Module):
    def __init__(self): 
        super(DASNet, self).__init__()
        
        self.branch1 = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((112,112)),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((56,56)),
            
            )
        
        self.branch2 = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            )
        
        self.combined = nn.Sequential(
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 2),
        )
        
    def forward(self, x):
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        
        features = torch.concat((b1, b2), dim=1)
        
        combined = self.combined(features)
        
        output = self.fc_layers(combined)
        return output
