#color net
import torch
from torch import nn
from torchvision import models

class ColorNet(nn.Module):
    #not-trainable, opeational block
    def __init__(self,name=None,drop=None,freeze_level=None):
        super().__init__()
        self.filter1=torch.div(torch.ones(1,1,7,7),1*1*7*7).to("cuda:0")
        self.filter2=torch.div(torch.ones(1,1,3,3),1*1*3*3).to("cuda:0")
        self.filter3=torch.ones(1,1,1,1).to("cuda:0")#for down sample
        
    def forward(self,x):#224
            x=nn.functional.conv2d(x,self.filter1,stride=(2,2),padding=(3,3))#(0)->112
            x=nn.functional.max_pool2d(x,kernel_size=3,stride=2,padding=1,dilation=1)#(3)->56
            
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))#(4)-0
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))

            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))#(4)-1
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))

            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))#(4)-2
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))

            x=nn.functional.conv2d(x,self.filter2,stride=(2,2),padding=(1,1))#(5)-0 #28
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))
#            x=nn.functional.conv2d(x,self.filter3,stride=(2,2))#down #14

            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))#(5)-1
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))

            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))#(5)-2
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))

            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))#(5)-3
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))

            x=nn.functional.conv2d(x,self.filter2,stride=(2,2),padding=(1,1))#(6)-0 #7
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))
#            x=nn.functional.conv2d(x,self.filter3,stride=(2,2))#down #4

            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))#(6)-1
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))

            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))#(6)-2
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))

            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))#(6)-3
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))

            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))#(6)-4
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))

            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))#(6)-5
            x=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))

            return x

class ColorNetRGB(nn.Module):
    def __init__(self,name=None,drop=None,freeze_level=None):
        super().__init__()
        self.colornet=ColorNet()

    def forward(self,x):
        r=torch.unsqueeze(x[:,0,:,:],axis=1)
        g=torch.unsqueeze(x[:,1,:,:],axis=1)
        b=torch.unsqueeze(x[:,2,:,:],axis=1)

        r=self.colornet(r)
        g=self.colornet(g)
        b=self.colornet(b)

        return torch.cat((r,g,b),axis=1)



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
        print(self.basenet)
        self.flat=nn.Flatten()
        for p in self.basenet.parameters():# all gradient freeze
            p.requires_grad = False
        self.colornet=ColorNetRGB()
        
    def forward(self,img_color,img_gray): #mean vector not-considered yet
        x=self.basenet(img_gray)#[32, 256, 14, 14]
        y=self.colornet(img_color)#[32, 3, 14, 14]
        z=torch.cat((x,y),axis=1) #[32, 259, 14, 14]    

        x=x.view(*x.size()[:-2],-1)
        y=y.view(*y.size()[:-2],-1)
        z=z.view(*z.size()[:-2],-1)
        
        gram_matrix_texture=torch.matmul(x,torch.transpose(x,1,2))
        gram_matrix_color=torch.matmul(y,torch.transpose(y,1,2))
        gram_matrix_comp=torch.matmul(torch.transpose(z,1,2),z) #colo + texture

        return gram_matrix_texture,gram_matrix_comp,gram_matrix_color

    def unfreeze(self):#from the level to end
        all_layers = list(self.basenet.children())
        for l in all_layers[self.freeze_level:]:
            for p in l.parameters():
                p.requires_grad=True


class PCA(nn.Module):#operational-layer
    def __init__(self):
        super().__init__()
    def forward(self,x):
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
                    



#BaseNet("resnet34",3)
