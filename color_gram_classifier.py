#gram classifier

import torch
from torch import nn
import colornet

class EmotionColorGramClassifier(nn.Module):
    def __init__(self,name=None,drop=None,freeze=None,mlp=None,dropout=None,activations=None,factors=None,level=None,d_model=None,nhead=None,dim_feedforward=None,num_layers=None):
        super().__init__()
        self.net=colornet.BaseNet(name=name,drop=drop,freeze_level=freeze,level=level)
        self.pca=colornet.PCA()
        self.pca_color=colornet.PCA_color()
        self.fc_layers=colornet.MLP(mlp,dropout,activations)
        self.factors=factors

    def forward(self,x):
        x,y,z=self.net(x[:,:3,:,:],x[:,3:,:,:])#color and gray
        x,y,z=self.pca(x),self.pca(y),self.pca_color(z,hidden_dim=y.size()[1])#texture,comp,color
        feat=torch.empty(x.size()[0],1,1).to("cuda:0")
        if "texture" in self.factors:
            feat=torch.cat((feat,x),dim=1)
        if "composition" in self.factors:
            feat=torch.cat((feat,y),dim=1)
        if "color" in self.factors:
            feat=torch.cat((feat,z),dim=1)
        print(feat.size())
        return self.fc_layers(feat[:,1:,:].squeeze(-1))
        
    def collect_feat(self,x):
        x,y,z=self.net(x[:,:3,:,:],x[:,3:,:,:])
        x,y,z=self.pca(x),self.pca(y),self.pca_color(z)
        feat=torch.cat((x,y,z),dim=1).squeeze(-1)
        return feat

    def collect_hidden_embedding(self,x):
        x,y,z=self.net(x[:,:3,:,:],x[:,3:,:,:])
        x,y,z=self.pca(x),self.pca(y),self.pca_color(z)
        feat=torch.cat((x,y,z),dim=1).squeeze(-1)
        return list(self.fc_layers.fc.children())[0](feat)

