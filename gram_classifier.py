#gram classifier

import torch
from torch import nn
import gramnet

class EmotionGramClassifier(nn.Module):
    def __init__(self,name=None,drop=None,freeze=None,mlp=None,dropout=None,activations=None):
        super().__init__()
        self.net=gramnet.BaseNet(name=name,drop=drop,freeze_level=freeze)
        self.pca=gramnet.PCA()
        self.fc_layers=gramnet.MLP(mlp,dropout,activations)

    def forward(self,x):
        x,y=self.net(x)
        x=self.pca(x).squeeze(-1)    
        feat=torch.cat((x,y),dim=1)
        return self.fc_layers(feat)

    def collect_feat(self,x):
        x,y=self.net(x)
        x=self.pca(x).squeeze(-1) 
        feat=torch.cat((x,y),dim=1)
        return feat

    def collect_hidden_embedding(self,x):
        x,y=self.net(x)
        x=self.pca(x).squeeze(-1) 
        feat=torch.cat((x,y),dim=1)
        return list(self.fc_layers.fc.children())[0](feat)

