#gram classifier

import torch
from torch import nn
import colornet

class EmotionColorGramTransformerClassifier(nn.Module):
    def __init__(self,name=None,drop=None,freeze=None,mlp=None,dropout=None,activations=None,factors=None,level=None,d_model=None,nhead=None,dim_feedforward=None,num_layers=None):
        super().__init__()
        self.net=colornet.BaseNet(name=name,drop=drop,freeze_level=freeze,level=level)
        self.pca=colornet.PCA()
        self.pca_color=colornet.PCA_color()
        self.fc_layers=colornet.MLP(mlp,dropout,activations)
        self.transformer=colornet.TransformerEncoder_tc1c2(d_model,nhead,dim_feedforward,num_layers)#colornet.MLP(mlp,dropout,activations)
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
        print("feat_size:",feat.size())
        y=self.transformer(feat[:,1:,:].squeeze(-1))#all the feat
        print(y.size())# batch * 64*3
        return self.fc_layers(y)#last fully connected
        
    def collect_feat(self,x):
        x,y,z=self.net(x[:,:3,:,:],x[:,3:,:,:])
        x,y,z=self.pca(x),self.pca(y),self.pca_color(z,hidden_dim=y.size()[1])
        feat=torch.empty(x.size()[0],1,1).to("cuda:0")
        if "texture" in self.factors:
            feat=torch.cat((feat,x),dim=1)
        if "composition" in self.factors:
            feat=torch.cat((feat,y),dim=1)
        if "color" in self.factors:
            feat=torch.cat((feat,z),dim=1)
        feat=feat.squeeze(-1)
        return feat

    def collect_tensors(self,x):
        tensor_gray=self.net.basenet(x[:,3:,:,:])
        tensor_color_conv3,tensor_color_conv4=self.net.colornet(x[:,:3,:,:])
        return tensor_gray,tensor_color_conv4

    def collect_transformer_embedding(self,x):
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
        y=self.transformer(feat[:,1:,:]).squeeze(-1)#all the feat
        return y

    def collect_hidden_embedding(self,x):
        x,y,z=self.net(x[:,:3,:,:],x[:,3:,:,:])
        x,y,z=self.pca(x),self.pca(y),self.pca_color(z)
        feat=torch.cat((x,y,z),dim=1).squeeze(-1)
        return list(self.fc_layers.fc.children())[0](feat)

