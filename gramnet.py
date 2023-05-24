#several kinds of neural net back bone
import torch
from torch import nn
from torchvision import models

class BaseNet(nn.Module):
    def __init__(self,name=None,drop=None,freeze_level=None):
        super().__init__()
        backbones={'resnet34':models.resnet34,
                  'vgg16':models.vgg16,
                  'vit_l_16':models.vit_l_16}
        self.name=name
        self.drop=drop
        self.freeze_level=freeze_level
        self.backbone=backbones[self.name](weights='IMAGENET1K_V1')
        self.basenet=nn.Sequential(*list(self.backbone.children())[:-drop])
        self.flat=nn.Flatten()
        for p in self.basenet.parameters():# all gradient freeze
            p.requires_grad = False
        
    def forward(self,x):
        x=self.basenet(x)
        x=x.view(*x.size()[:-2],-1)#batch x comp (7*7) x 512
        gram_matrix_texture=torch.matmul(x,torch.transpose(x,1,2))
        #gram_matrix_comp=torch.mean(x,dim=1,keepdim=False)
        gram_matrix_comp=torch.matmul(torch.transpose(x,1,2),x)
        
        return gram_matrix_texture,gram_matrix_comp

    def unfreeze(self):#from the level to end
        all_layers = list(self.basenet.children())
        for l in all_layers[self.freeze_level:]:
            for p in l.parameters():
                p.requires_grad=True


class PCA(nn.Module):#operational-layer
    def __init__(self):
        super().__init__()
    def forward(self,x):
        print(x.size())
        u,s,_=torch.linalg.svd(x,full_matrices=True)#,driver="gesvda")
        s=s[:,:3]/torch.sum(s[:,:3],dim=1,keepdim=True)
        u=u[:,:,:3] #no slice temporary
        s=s.unsqueeze(-1)
        simplex=torch.matmul(u,s)
        return simplex

class MLP(nn.Module):
    def __init__(self,layers=None,dropout=None,activations=None):
        super().__init__()
        fc=[]
        for layer,drop,relu in zip(layers,dropout,activations):
            linear_layer=nn.Linear(layer[0],layer[1])
            fc.append(linear_layer)
            if drop is not None:
                dropout_layer=nn.Dropout(p=drop)
                fc.append(dropout_layer)
            if relu is not None:
                relu_layer=nn.ReLU(inplace=True)
                fc.append(relu_layer)
        self.fc=nn.Sequential(*fc)
    def forward(self,x):
        return self.fc(x)
                    


#img_encoder = ResnetEncoder('resnet34', adapt_image_size=1).unfreeze(level=7, verbose=True)
#print(BaseNet("resnet34",1).)
