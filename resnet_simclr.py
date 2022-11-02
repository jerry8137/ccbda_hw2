import torch.nn as nn
import torchvision.models as models

class ResNetSimCLR(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Identity()

        # add projection head
        self.projection = nn.Sequential(nn.Linear(512, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        feature = self.resnet(x)
        out = self.projection(feature)
        return out

