import torch.nn as nn
import torchvision.models as models

class ResNetSimCLR(nn.Module):
    def __init__(self, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet = models.resnet18(pretrained=False, num_classes=512)

        # add projection head
        self.projection = nn.Sequential(nn.BatchNorm1d(512),
                                        nn.ReLU(),
                                        nn.Linear(512, out_dim))

    def forward(self, x):
        embedding = self.resnet(x)
        out = self.projection(embedding)
        return embedding, out

