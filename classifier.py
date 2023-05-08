import torch
from torch import nn
import neuralnet

class EmotionClassifier(nn.Module):
    def __init__(self,name=None,drop=None,freeze=None,mlp=None,dropout=None,activations=None):
        super().__init__()
        self.net=neuralnet.BaseNet(name=name,drop=drop,freeze_level=freeze)
        self.fc_layers=neuralnet.MLP(mlp,dropout,activations)
        #self.logits=nn.Sequential(self.net, self.fc)#neuralnet.BaseNet(name=name,drop=1),neuralnet.MLP(mlp,dropout,activations))

    def forward(self,x):
        x=self.net(x)
        return self.fc_layers(x)
