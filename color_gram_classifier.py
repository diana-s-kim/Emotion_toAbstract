#gram classifier

import torch
from torch import nn
import colornet

class EmotionColorGramClassifier(nn.Module):
    def __init__(self,name=None,drop=None,freeze=None,mlp=None,dropout=None,activations=None):
        super().__init__()
        self.net=colornet.BaseNet(name=name,drop=drop,freeze_level=freeze)
        self.pca=colornet.PCA()
        self.fc_layers=colornet.MLP(mlp,dropout,activations)

    def forward(self,x):
        x,y,z=self.net(x[:,:3,:,:],x[:,3:,:,:])#color and gray
        print(x.shape)
        print(y.shape)
        print(z.shape)

        x,y,z=self.pca(x),self.pca(y),self.pca(z)
        feat=torch.cat((x,y,z),dim=1).squeeze(-1) #texture,comp,color
        return self.fc_layers(feat)

    def collect_feat(self,x):
        x,y,z=self.net(x[:,:3,:,:],x[:,3:,:,:])
        x,y,z=self.pca(x),self.pca(y),self.pca(z)
        feat=torch.cat((x,y,z),dim=1).squeeze(-1)
        return feat

    def collect_hidden_embedding(self,x):
        x,y,z=self.net(x[:,:3,:,:],x[:,3:,:,:])
        x,y,z=self.pca(x),self.pca(y),self.pca(z)
        feat=torch.cat((x,y,z),dim=1).squeeze(-1)
        return list(self.fc_layers.fc.children())[0](feat)

