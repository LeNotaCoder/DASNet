import torch
import torch.nn as nn
    
class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels=[1, 2, 4], pool_type='max'):
        super(SpatialPyramidPooling, self).__init__()
        self.levels = levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size()
        features = []
        for level in self.levels:
            pooling = F.adaptive_max_pool2d if self.pool_type == 'max' else F.adaptive_avg_pool2d
            out = pooling(x, output_size=(level, level))
            features.append(out.view(num, -1))
        return torch.cat(features, dim=1)

class DASNet(nn.Module):
    def __init__(self):
        super(DASNet, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((112, 112)),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((56, 56))
        )

        self.combined = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.spp = SpatialPyramidPooling(levels=[1, 2, 4], pool_type='max')

        self.fc_layers = nn.Sequential(
            nn.Linear(10752, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        features = torch.cat((b1, b2), dim=1)
        combined = self.combined(features)
        pooled = self.spp(combined)
        output = self.fc_layers(pooled)
        return output
