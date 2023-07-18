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
            conv_3=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))

            x=nn.functional.conv2d(conv_3,self.filter2,stride=(2,2),padding=(1,1))#(6)-0 #7
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
            conv_4=nn.functional.conv2d(x,self.filter2,stride=(1,1),padding=(1,1))
            
            return conv_3,conv_4

class ColorNetRGB(nn.Module):
    def __init__(self,name=None,drop=None,freeze_level=None):
        super().__init__()
        self.colornet=ColorNet()

    def forward(self,x):
        r=torch.unsqueeze(x[:,0,:,:],axis=1)
        g=torch.unsqueeze(x[:,1,:,:],axis=1)
        b=torch.unsqueeze(x[:,2,:,:],axis=1)

        r_conv3,r_conv4=self.colornet(r)
        g_conv3,g_conv4=self.colornet(g)
        b_conv3,b_conv4=self.colornet(b)

        return torch.cat((r_conv3,g_conv3,b_conv3),axis=1),torch.cat((r_conv4,g_conv4,b_conv4),axis=1)



class BaseNet(nn.Module):
    def __init__(self,name=None,drop=None,freeze_level=None,level=None):
        super().__init__()
        backbones={'resnet34':models.resnet34,
                  'vgg16':models.vgg16,
                  'vit_l_16':models.vit_l_16}
        self.name=name
        self.drop=drop
        self.level=level
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
        y_conv3,y_conv4=self.colornet(img_color)#[32, 3, 14, 14]
        if self.level=="conv3":
            y=y_conv3
        if self.level=="conv4":
            y=y_conv4
        z=torch.cat((x,y),axis=1) #[32, 259, 14, 14]#conv4    
        x=x.view(*x.size()[:-2],-1)#conv3
        y=y_conv3.view(*y_conv3.size()[:-2],-1)
        z=z.view(*z.size()[:-2],-1)
        
        gram_matrix_texture=torch.matmul(x,torch.transpose(x,1,2))
        gram_matrix_color=torch.matmul(y,torch.transpose(y,1,2))
        gram_matrix_comp=torch.matmul(torch.transpose(z,1,2),z)# color + general channels
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
        u,s,_=torch.linalg.svd(x,full_matrices=True)
        s=s[:,:3]/torch.sum(s[:,:3],dim=1,keepdim=True)
        u=u[:,:,:3] #no slice temporary
        s=s.unsqueeze(-1)
        simplex=torch.matmul(u,s)
        return simplex

class PCA_color(nn.Module): #encode color population
    def __init__(self):
        super().__init__()
    def forward(self,x,hidden_dim=None):
        u,s,_=torch.linalg.svd(x,full_matrices=True)
        scale =s[:,0]/(1.0*1.0*1.0*hidden_dim)
        scaled_first_u=u[:,:,0]*scale.unsqueeze(-1)
        return scaled_first_u.unsqueeze(-1)#better capture color characteristics


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
                    
class TransformerEncoder_tc1c2(nn.Module): #(256,196,3) to find interelationship between texture, composition, color
    def __init__(self,d_model=None,nhead=None,dim_feedforward=None,num_layers=None):
        super().__init__()
        self.d_model=d_model
        self.make_emb_texture=nn.Linear(256,d_model)
        self.make_emb_composition=nn.Linear(196,d_model)
        self.make_emb_color=nn.Linear(3,d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self,x):
        texture=x[:,:256]
        composition=x[:,256:256+196]
        color=x[:,-3:]

        texture_emb=torch.unsqueeze(self.make_emb_texture(texture),dim=1)
        composition_emb=torch.unsqueeze(self.make_emb_composition(composition),dim=1)
        color_emb=torch.unsqueeze(self.make_emb_color(color),dim=1)
        return torch.reshape(self.transformer_encoder(torch.cat((texture_emb,composition_emb,color_emb),dim=1)),(-1,self.d_model*3))
        
    
        


#BaseNet("resnet34",3)
