import torch.nn as nn
import torchvision.models as models

class ResNetSimCLR(nn.Module):
    def __init__(self, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet = models.resnet18(pretrained=False, num_classes=out_dim)
        # add projection head
        self.resnet.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                       nn.ReLU(),
                                       self.resnet.fc)

    def forward(self, x):
        return self.resnet(x)

